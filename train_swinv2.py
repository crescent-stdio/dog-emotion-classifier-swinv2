import os
import glob
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    Swinv2ForImageClassification,
)
import evaluate
from tqdm import tqdm

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def get_timestamp() -> str:
    """Return current timestamp YYMMDD-HHMMSS as str."""
    return datetime.now().strftime("%y%m%d-%H%M%S")


# --------- Dataset ---------
class DogExpressionDataset(Dataset):
    """Four-class dataset with automatic 8/1/1 split.

    Args
    ----
    root_dir : path to the dataset root (contains angry/, happy/, ...)
    split    : train | val | test
    ratio    : tuple of three floats summing to 1.0 → (train, val, test)
    transform: optional torchvision Transform applied *before* processor.
    processor: Hugging Face image processor (resizing / norm handled there).
    """

    classes: List[str] = ["angry", "happy", "relaxed", "sad"]
    cls2id = {c: i for i, c in enumerate(classes)}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        transform: T.Compose | None = None,
        processor=None,
    ) -> None:
        assert split in {"train", "val", "test"}
        self.processor = processor
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        train_r, val_r, test_r = ratio
        assert abs(sum(ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

        for cls in self.classes:
            paths = glob.glob(os.path.join(root_dir, cls, "*"))
            paths.sort()
            random.shuffle(paths)
            n = len(paths)
            n_train = int(n * train_r)
            n_val = int(n * val_r)
            ranges = {
                "train": paths[:n_train],
                "val": paths[n_train : n_train + n_val],
                "test": paths[n_train + n_val :],
            }
            self.samples.extend([(p, self.cls2id[cls]) for p in ranges[split]])

    # ------------------------------------------------------------ dunder
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        proc = self.processor(img, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in proc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# --------- Data augmentation (PIL-based) ---------
train_tf = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]
)
val_test_tf = T.Compose([])  # identity → processor handles resize / norm


# --------- Hyper-parameters (CLI-able) ---------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="DogEmotion", help="Dataset root")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
parser.add_argument("--lr", type=float, default=3e-5, help="Base learning rate")
parser.add_argument("--model", type=str, default="microsoft/swinv2-base-patch4-window12-192-22k")
parser.add_argument("--out", type=str, default="models", help="Output dir")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)


# --------- Model / Processor / Dataloaders ---------
processor = AutoImageProcessor.from_pretrained(args.model)
model = Swinv2ForImageClassification.from_pretrained(
    args.model,
    num_labels=len(DogExpressionDataset.classes),
    id2label={i: c for i, c in enumerate(DogExpressionDataset.classes)},
    label2id={c: i for i, c in enumerate(DogExpressionDataset.classes)},
    ignore_mismatched_sizes=True,
)


# ---- Freeze entire backbone, then unfreeze last stage + head ----
for p in model.parameters():
    p.requires_grad = False

backbone = model.swinv2  # underlying Swinv2Model

# 1) unfreeze last stage of the encoder
encoder = backbone.encoder if hasattr(backbone, "encoder") else backbone
if hasattr(encoder, "layers"):
    stages = encoder.layers
elif hasattr(encoder, "stages"):
    stages = encoder.stages
else:
    raise AttributeError("Cannot find stage container in Swinv2 encoder")

last_stage = stages[-1]
for p in last_stage.parameters():
    p.requires_grad = True

# 2) unfreeze final normalization layer if present
for attr_name in ["norm", "layernorm", "layer_norm", "ln", "final_norm"]:
    if hasattr(backbone, attr_name):
        for p in getattr(backbone, attr_name).parameters():
            p.requires_grad = True
        break  # found

# 3) unfreeze classifier head
for p in model.classifier.parameters():
    p.requires_grad = True

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"→ Using device: {device}. Param count: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

train_ds = DogExpressionDataset(args.root, "train", transform=train_tf, processor=processor)
val_ds = DogExpressionDataset(args.root, "val", transform=val_test_tf, processor=processor)
test_ds = DogExpressionDataset(args.root, "test", transform=val_test_tf, processor=processor)

## size of train/val/test
print(f"→ Train size: {len(train_ds)}")
print(f"→ Val size: {len(val_ds)}")
print(f"→ Test size: {len(test_ds)}")

def collate_fn(batch: List[dict]):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)


# --------- Optimiser & LR ---------
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss()
# Set up mixed precision training if using CUDA
if device.type == "cuda":
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# --------- Metrics ---------
metric_accuracy = evaluate.load("accuracy")
metric_f1_macro = evaluate.load("f1")           # pass average="macro" at compute()
metric_precision = evaluate.load("precision")   # for completeness
metric_recall = evaluate.load("recall")
metric_confmat = evaluate.load("confusion_matrix")

# helper for top-k accuracy (k=2)
def topk_acc(preds: torch.Tensor, labels: torch.Tensor, k: int = 2) -> float:
    topk = preds.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

def balanced_acc_from_conf(conf):
    conf = torch.tensor(conf)
    TP = conf.diag()
    FN = conf.sum(dim=1) - TP
    recall = TP.float() / (TP + FN).clamp(min=1)
    return recall.mean().item()

best_val_f1 = 0.0

# --------- Train loop ---------
for epoch in range(1, args.epochs + 1):
    # -------------------------- Training -------------------------
    model.train()
    running_loss = 0.0
    seen = 0
    train_bar = tqdm(train_loader, desc=f"[{epoch:02d}/{args.epochs}] train", leave=False)
    for batch in train_bar:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(device.type, enabled=(scaler is not None)):
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        train_bar.set_postfix(loss=f"{running_loss/seen:.4f}")
    scheduler.step()

    print(f"[Epoch {epoch:02d}] train_loss={running_loss/seen:.4f}")

    # ------------------------- Validation ------------------------
    model.eval()
    preds_list, labels_list, logits_list = [], [], []  # collect logits for top-k
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"[{epoch:02d}/{args.epochs}] val", leave=False)
        for batch in val_bar:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            logits_list.append(logits.cpu())
            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())

    preds = torch.cat(preds_list)
    labels = torch.cat(labels_list)
    all_logits = torch.cat(logits_list)

    # Evaluate metrics
    acc = metric_accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = metric_f1_macro.compute(predictions=preds, references=labels, average="macro")["f1"]
    precision_macro = metric_precision.compute(predictions=preds, references=labels, average="macro")["precision"]
    recall_macro = metric_recall.compute(predictions=preds, references=labels, average="macro")["recall"]
    conf_mat = metric_confmat.compute(predictions=preds, references=labels)["confusion_matrix"]
    bal_acc = balanced_acc_from_conf(conf_mat)
    top2 = topk_acc(all_logits, labels, k=2)

    print(
        "  val | Acc={:.4f}  Macro-F1={:.4f}  Prec={:.4f}  Rec={:.4f}  BalAcc={:.4f}  Top-2Acc={:.4f}".format(
            acc, f1_macro, precision_macro, recall_macro, bal_acc, top2
        )
    )

    # ------------------------ Checkpointing ----------------------
    if f1_macro > best_val_f1:
        best_val_f1 = f1_macro
        ckpt_name = f"best_swin_{get_timestamp()}_epoch{epoch}_f1{f1_macro:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "f1_macro": best_val_f1,
            },
            os.path.join(args.out, ckpt_name),  # save directly under output dir
        )
        print("-> New best checkpoint saved!\n")

    # --------- Test set ---------
    print("Evaluating best model on TEST set ...")
    # (Re)load best checkpoint
    ckpts = sorted(glob.glob(os.path.join(args.out, "best_swin_*")))
    assert ckpts, "No checkpoint was saved during training."
    print(f"-> Loading {ckpts[-1]}")
    model.load_state_dict(torch.load(ckpts[-1], map_location=device)["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)

    preds_list, labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())

    preds = torch.cat(preds_list)
    labels = torch.cat(labels_list)

    test_acc = metric_accuracy.compute(predictions=preds, references=labels)["accuracy"]
    test_f1 = metric_f1_macro.compute(predictions=preds, references=labels, average="macro")["f1"]
    test_precision = metric_precision.compute(predictions=preds, references=labels, average="macro")["precision"]   
    test_recall = metric_recall.compute(predictions=preds, references=labels, average="macro")["recall"]
    test_conf_mat = metric_confmat.compute(predictions=preds, references=labels)["confusion_matrix"]
    test_bal_acc = balanced_acc_from_conf(test_conf_mat)
    test_top2 = topk_acc(all_logits, labels, k=2)

    print("TEST | Acc={:.4f}  Macro-F1={:.4f}  Prec={:.4f}  Rec={:.4f}  BalAcc={:.4f}  Top-2Acc={:.4f}".format(test_acc, test_f1, test_precision, test_recall, test_bal_acc, test_top2) + "\n")
