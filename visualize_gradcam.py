import argparse
import glob
import os
import random
from datetime import datetime
from typing import List

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T
from transformers import AutoImageProcessor, Swinv2ForImageClassification
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["angry", "happy", "relaxed", "sad"]
MEAN = STD = [0.5, 0.5, 0.5]  # assumed from training

# ------------------- CONFIG -------------------
DEFAULT_MULTI_LAYERS = 2  # how many deepest blocks to use when --multi


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


class SwinV2Wrapper(torch.nn.Module):
    """Wrap Swinv2 backbone + classifier for CAM."""

    def __init__(self, model: Swinv2ForImageClassification):
        super().__init__()
        self.model = model
        self.backbone = model.swinv2  # Swinv2Model
        self.classifier = model.classifier
        self.classes = CLASSES

    def forward(self, x):  # type: ignore[override]
        outputs = self.model(pixel_values=x)
        return outputs.logits


# ------------------------------------------------------------

def build_transforms():
    infer_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    vis_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    return infer_tf, vis_tf


def load_model(ckpt_path: str, model_id: str) -> SwinV2Wrapper:
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Swinv2ForImageClassification.from_pretrained(
        model_id,
        num_labels=len(CLASSES),
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)},
        ignore_mismatched_sizes=True,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return SwinV2Wrapper(model).to(DEVICE).eval()


def pick_images(img_dir: str, img_path: str | None = None, random_state: int | None = None) -> List[str]:
    if random_state:
        random.seed(random_state)
    if img_path:
        return [img_path]
      
    # pick one random per DogEmotion class
    paths = []
    for cls in CLASSES:
        pool = glob.glob(os.path.join(img_dir, cls, "*.jpg"))
        if pool:
            paths.append((random.choice(pool), cls))
    return paths


def get_patch_conv(backbone):
    """Return first conv layer used for patch embedding."""
    # Swin v1: backbone.patch_embed.proj
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "proj"):
        return backbone.patch_embed.proj
    # Swin v2 HF impl: backbone.embeddings.patch_embeddings.projection
    if hasattr(backbone, "embeddings"):
        emb = backbone.embeddings
        for attr in ["patch_embeddings", "patch_embedding"]:
            if hasattr(emb, attr):
                pe = getattr(emb, attr)
                if hasattr(pe, "projection"):
                    return pe.projection
                if isinstance(pe, torch.nn.Module):
                    # maybe direct conv
                    return pe
    # as fallback, use first conv found in backbone children
    for m in backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise AttributeError("Could not locate patch embedding conv layer in Swinv2 backbone")


def get_last_stage(backbone):
    """Return the deepest stage/module inside Swin(v2) backbone."""
    for attr in ["layers", "stages", "encoder.layers", "encoder.stages"]:
        parts = attr.split('.')
        obj = backbone
        ok = True
        for p in parts:
            if not hasattr(obj, p):
                ok = False; break
            obj = getattr(obj, p)
        if ok and isinstance(obj, (list, torch.nn.ModuleList)):
            return obj[-1]
    raise AttributeError("Could not find stages in backbone")


def get_target_layers(backbone, multi: bool = False, num_layers: int = DEFAULT_MULTI_LAYERS):
    """Return list of target layers for CAM."""
    if not multi:
        return [get_patch_conv(backbone)]

    last_stage = get_last_stage(backbone)
    # pick last `num_layers` blocks norm2 (or norm) layers as targets
    if hasattr(last_stage, "blocks"):
        blocks = last_stage.blocks
    else:
        # if stage itself is a list of blocks
        blocks = last_stage
    selected = []
    for blk in list(blocks)[-num_layers:]:
        if hasattr(blk, "norm2"):
            selected.append(blk.norm2)
        elif hasattr(blk, "norm"):
            selected.append(blk.norm)
        else:
            # fallback to block itself
            selected.append(blk)
    return selected


def compute_cam(wrapper: SwinV2Wrapper, img_path: str, infer_tf, vis_tf, multi_layers=False):
    # preprocessing
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = infer_tf(img_pil).unsqueeze(0).to(DEVICE)
    vis_img = vis_tf(img_pil).permute(1, 2, 0).cpu().numpy()

    target_layers = get_target_layers(wrapper.backbone, multi_layers)
    # GradCAM can accept list of layers directly
    # cam = GradCAM(model=wrapper, target_layers=target_layers)
    from pytorch_grad_cam import GradCAMPlusPlus
    cam = GradCAMPlusPlus(model=wrapper, target_layers=target_layers)

    # forward for probabilities
    with torch.no_grad():
        logits = wrapper(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # compute weighted combination of CAMs across classes & layers
    cams_weighted = np.zeros((224, 224), dtype=np.float32)
    for cls_idx, prob in enumerate(probs):
        layer_cams = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(cls_idx)])  # returns (L,H,W)
        # sum cams over layers then resize
        merged = np.mean(layer_cams, axis=0)
        cam_up = cv2.resize(merged, (224, 224), interpolation=cv2.INTER_CUBIC)
        cam_up = cv2.GaussianBlur(cam_up, (5,5), sigmaX=0)
        cam_up = (cam_up - cam_up.min()) / (cam_up.max() + 1e-8)
        cams_weighted += cam_up * prob
        
        ####### 250612 #######
        # grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(cls_idx)])[0]
        # cam_tensor = torch.tensor(grayscale_cam)[None,None]  # 1×1×H×W
        # cam_up = F.interpolate(cam_tensor, size=(224,224), mode="bilinear", align_corners=False).squeeze().numpy()
        # cams_weighted += cam_up * prob
    cams_weighted = (cams_weighted - cams_weighted.min()) / (cams_weighted.max() + 1e-8)

    overlay = show_cam_on_image(vis_img, cams_weighted, use_rgb=True)
    return img_pil, cams_weighted, overlay, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Swinv2 checkpoint")
    parser.add_argument("--model_id", type=str, default="microsoft/swinv2-base-patch4-window12-192-22k")
    parser.add_argument("--img", type=str, default=None, help="Single image path; if omitted random from DogEmotion")
    parser.add_argument("--img_dir", type=str, default="../Pet", help="Data directory")
    parser.add_argument("--multi", action="store_true", help="Use multiple deep layers for CAM aggregation")
    parser.add_argument("--random_state", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Print current random state
    if args.random_state is not None:
        print(f"Using random seed: {args.random_state}")
    else:
        print("No random seed specified, using system default")
        print(f"Current random state: {random.getstate()[1][0]}")  # First number in state tuple is the seed
        args.random_state = random.getstate()[1][0]

    infer_tf, vis_tf = build_transforms()
    model = load_model(args.ckpt, args.model_id)

    img_paths = pick_images(args.img_dir, args.img, args.random_state)

    n = len(img_paths)
    plt.figure(figsize=(12, 4 * n))
    overlays_collected = []  # store overlay images
    for row, p in enumerate(img_paths):
        if isinstance(p, tuple):
            img_path, label = p
        else:
            img_path, label = p, "?"
        img_pil, cam_map, overlay, probs = compute_cam(model, img_path, infer_tf, vis_tf, multi_layers=args.multi)
        overlays_collected.append(overlay)
        pred_cls = probs.argmax()

        # Original
        plt.subplot(n, 3, row * 3 + 1)
        plt.imshow(img_pil)
        plt.title(f"Original\n{img_path.split('/')[-1]} | label: {label}")
        plt.axis("off")

        # Heatmap only
        plt.subplot(n, 3, row * 3 + 2)
        plt.imshow(cam_map, cmap="jet")
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")

        # Overlay
        plt.subplot(n, 3, row * 3 + 3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPred: {CLASSES[pred_cls]} ({probs[pred_cls]:.2%})")
        plt.axis("off")
    plt.tight_layout()

    os.makedirs("results_gradcam", exist_ok=True)
    out_path = os.path.join("results_gradcam", f"gradcam_swinv2_{get_timestamp()}_random_state_{args.random_state}.png")

    try:
        plt.savefig(out_path, dpi=300, format="png")
        print(f"Saved visualization → {out_path}")
    except Exception as e:
        print(f"[WARN] savefig failed ({e}); fallback to cv2.imwrite")
        if overlays_collected:
            grid = np.concatenate(overlays_collected, axis=0)
            cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"Saved fallback visualization → {out_path}")
        else:
            print("No overlay images collected; nothing to save.")
    plt.show()
    


if __name__ == "__main__":
    main() 