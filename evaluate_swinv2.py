import argparse
import glob
import os
from collections import Counter
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoImageProcessor, Swinv2ForImageClassification
import evaluate
from tqdm import tqdm

# ---------------------------------------------------------------------------
class PetExpressionDataset(Dataset):
    """Dataset wrapper for Pet folder.

    Folder mapping:
      Angry  -> angry
      happy  -> happy
      Other  -> relaxed (assumed)
      Sad    -> sad
    """

    classes: List[str] = ["angry", "happy", "relaxed", "sad"]
    folder2cls = {
        "Angry": "angry",
        "happy": "happy",
        "Other": "relaxed",
        "Sad": "sad",
    }
    cls2id = {c: i for i, c in enumerate(classes)}

    def __init__(self, root_dir: str, processor, transform: T.Compose | None = None, include_other: bool = False):
        self.samples: List[Tuple[str, int]] = []
        self.processor = processor
        self.transform = transform
        self.include_other = include_other

        for folder, cls in self.folder2cls.items():
            if folder == "Other" and not include_other:
                # skip 'Other' samples when flag is False
                continue
            pattern = os.path.join(root_dir, folder, "*")
            for p in glob.glob(pattern):
                self.samples.append((p, self.cls2id[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        inputs = self.processor(img, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# ---------------------------------------------------------------------------

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/best_swin_250611-234218_f10.9549_epoch07.pt", help="Path to checkpoint .pt")
    parser.add_argument("--model_id", type=str, default="microsoft/swinv2-base-patch4-window12-192-22k")
    parser.add_argument("--root", type=str, default="../Pet", help="Pet dataset root directory")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--include_other", action="store_true", help="Include 'Other' folder as 'relaxed' class")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = Swinv2ForImageClassification.from_pretrained(
        args.model_id,
        num_labels=4,
        id2label={i: c for i, c in enumerate(PetExpressionDataset.classes)},
        label2id={c: i for i, c in enumerate(PetExpressionDataset.classes)},
        ignore_mismatched_sizes=True,
    )

    # load checkpoint weights
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device).eval()

    # dataset & loader
    test_tf = T.Compose([])  # no augmentation; processor handles resize/norm
    ds = PetExpressionDataset(args.root, processor, test_tf, include_other=args.include_other)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # metrics
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    metric_prec = evaluate.load("precision")
    metric_rec = evaluate.load("recall")
    metric_conf = evaluate.load("confusion_matrix")

    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())

    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)

    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    prec = metric_prec.compute(predictions=preds, references=labels, average="macro")["precision"]
    rec = metric_rec.compute(predictions=preds, references=labels, average="macro")["recall"]
    conf = metric_conf.compute(predictions=preds, references=labels)["confusion_matrix"]

    # balanced accuracy
    import numpy as np
    conf_arr = np.array(conf)
    tp = np.diag(conf_arr)
    fn = conf_arr.sum(axis=1) - tp
    bal_acc = (tp / np.clip(tp + fn, 1, None)).mean()

    print("\nEvaluation on Pet dataset ({} samples)".format(len(ds)))
    print("Accuracy         : {:.4f}".format(acc))
    print("Macro Precision  : {:.4f}".format(prec))
    print("Macro Recall     : {:.4f}".format(rec))
    print("Macro F1         : {:.4f}".format(f1))
    print("Balanced Accuracy: {:.4f}".format(bal_acc))
    print("Confusion Matrix :\n", conf)


if __name__ == "__main__":
    main() 