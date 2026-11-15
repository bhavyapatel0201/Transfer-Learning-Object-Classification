#!/usr/bin/env python3
"""
Transfer-learning object classifier (assignment steps 1-6)
==========================================================

1.  Pretrained backbone: VGG19 or DenseNet121 (selectable via --arch)
2.  Dataset: phone-captured images, ≥6 classes, ≥25 images/class
    Expected layout (created by phone_to_dataset.py or manually):
        dataset/
            train/  class_a/ *.jpg  class_b/ *.jpg  …
            val/    class_a/ *.jpg  …
            test/   class_a/ *.jpg  …
3.  Augmentation applied to the train split.
4.  70/15/15 split handled by phone_to_dataset.py (or provided externally).
5.  Convolutional layers frozen; ADAM hyperparameter search over
    learning-rate and weight-decay; best weights saved.
6.  Test evaluation: accuracy, per-class precision/recall/F1,
    confusion matrix, and a grid of sample predictions.

Usage:
  # Step 1: organise your raw photos
  python transfer_classify.py --prepare --raw_dir raw_photos/ --dataset_dir dataset/

  # Step 2: train (VGG19, default; or --arch densenet121)
  python transfer_classify.py --train --dataset_dir dataset/ --arch vgg19

  # Step 3: evaluate on test set
  python transfer_classify.py --test  --dataset_dir dataset/ --arch vgg19

Dependencies: torch torchvision pillow scikit-learn matplotlib numpy
"""

import argparse, json, os, random, shutil, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)


# ──────────────────────────────────────────────────────────────────────
#  0. Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEVICE}")


# ──────────────────────────────────────────────────────────────────────
#  PREPARE  – 70 / 15 / 15 split from a flat raw_dir
# ──────────────────────────────────────────────────────────────────────
def prepare_dataset(raw_dir: str, dataset_dir: str,
                    split=(0.70, 0.15, 0.15)):
    """
    raw_dir layout:
        raw_photos/
            shoes/      img1.jpg  img2.jpg …
            fork/       img1.jpg  …
            …

    Copies (not moves) into dataset/{train,val,test}/<class>/.
    """
    raw = Path(raw_dir)
    ds  = Path(dataset_dir)
    splits = ["train", "val", "test"]
    for sp in splits:
        (ds / sp).mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in raw.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No sub-directories found in {raw_dir}")

    summary = {}
    for cls in class_dirs:
        imgs = list(cls.glob("*.[jJpP][pPnN][gG]")) + \
               list(cls.glob("*.JPEG")) + list(cls.glob("*.jpeg"))
        random.shuffle(imgs)
        n = len(imgs)
        if n < 10:
            print(f"[warn] {cls.name}: only {n} images — consider adding more")
        n_tr = max(1, int(n * split[0]))
        n_va = max(1, int(n * split[1]))
        buckets = {
            "train": imgs[:n_tr],
            "val":   imgs[n_tr:n_tr+n_va],
            "test":  imgs[n_tr+n_va:],
        }
        summary[cls.name] = {k: len(v) for k, v in buckets.items()}
        for sp, files in buckets.items():
            dest = ds / sp / cls.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)

    print("\n── Dataset split ──────────────────────────────")
    print(f"{'class':<18} {'train':>7} {'val':>7} {'test':>7}")
    for cls, counts in summary.items():
        print(f"{cls:<18} {counts['train']:>7} {counts['val']:>7} {counts['test']:>7}")
    print(f"Saved to {dataset_dir}")


# ──────────────────────────────────────────────────────────────────────
#  DATA LOADERS
# ──────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.08),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2),      # CutOut / random erasing
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_loaders(dataset_dir, batch_size=32):
    ds_dir = Path(dataset_dir)
    train_ds = datasets.ImageFolder(ds_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(ds_dir / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(ds_dir / "test",  transform=val_tf)

    num_w = min(4, os.cpu_count() or 1)
    kw = dict(batch_size=batch_size, num_workers=num_w, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw),
            train_ds.classes)


# ──────────────────────────────────────────────────────────────────────
#  MODEL BUILDING – freeze conv layers, replace classifier head
# ──────────────────────────────────────────────────────────────────────
def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # freeze ALL feature layers (conv + pooling)
        for p in model.features.parameters():
            p.requires_grad = False
        in_feat = model.classifier[6].in_features
        # replace final linear with a new head tuned for our num_classes
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for p in model.features.parameters():
            p.requires_grad = False
        in_feat = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose vgg19 or densenet121.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n[{arch}] trainable params: {trainable:,} / {total:,} total")
    return model.to(DEVICE)


# ──────────────────────────────────────────────────────────────────────
#  TRAINING LOOP  (one pass)
# ──────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = correct = n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


# ──────────────────────────────────────────────────────────────────────
#  HYPERPARAMETER SEARCH  (grid over lr × wd)
# ──────────────────────────────────────────────────────────────────────
def hp_search(arch, dataset_dir, epochs_hp=2, batch_size=32):
    """
    Grid search over the two most impactful Adam hyperparameters:
      - learning rate:   [1e-3, 1e-4]   (reduced for CPU speed)
      - weight decay:    [0, 1e-4]      (reduced for CPU speed)
    Trains each config for `epochs_hp` epochs (quick signal).
    Returns best (lr, wd) by val accuracy.
    """
    lr_grid = [1e-3, 1e-4]
    wd_grid  = [0.0, 1e-4]
    results  = []
    train_loader, val_loader, _, classes = get_loaders(dataset_dir, batch_size)
    criterion = nn.CrossEntropyLoss()

    print("\n── Hyperparameter search ───────────────────────────")
    print(f"{'lr':>10} {'wd':>10} {'val_acc':>10}")
    for lr in lr_grid:
        for wd in wd_grid:
            model     = build_model(arch, len(classes))
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=wd)
            for _ in range(epochs_hp):
                run_epoch(model, train_loader, criterion, optimizer)
            _, val_acc = run_epoch(model, val_loader, criterion)
            results.append((lr, wd, val_acc))
            print(f"{lr:>10.0e} {wd:>10.0e} {val_acc:>9.3f}")
            del model

    results.sort(key=lambda x: -x[2])
    best_lr, best_wd, best_acc = results[0]
    print(f"\nBest  lr={best_lr:.0e}  wd={best_wd:.0e}  val_acc={best_acc:.3f}")
    return best_lr, best_wd


# ──────────────────────────────────────────────────────────────────────
#  FULL TRAINING
# ──────────────────────────────────────────────────────────────────────
def train(arch, dataset_dir, epochs=30, batch_size=32, out_dir="."):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- hyperparameter search ----------------------------------------
    best_lr, best_wd = hp_search(arch, dataset_dir, batch_size=batch_size)

    # --- build final model and loaders --------------------------------
    train_loader, val_loader, _, classes = get_loaders(dataset_dir, batch_size)
    num_classes = len(classes)
    model = build_model(arch, num_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_lr, weight_decay=best_wd)
    # cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
    best_val = 0.0
    ckpt_path = out / f"{arch}_best.pt"

    print("\n── Full training ───────────────────────────────────")
    print(f"{'ep':>4} {'tr_loss':>9} {'tr_acc':>8} {'va_loss':>9} {'va_acc':>8}  LR")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_l, tr_a = run_epoch(model, train_loader, criterion, optimizer)
        va_l, va_a = run_epoch(model, val_loader,   criterion)
        scheduler.step()
        for k, v in zip(history, [tr_l, tr_a, va_l, va_a]):
            history[k].append(v)
        lr_now = scheduler.get_last_lr()[0]
        print(f"{ep:>4}  {tr_l:>8.4f}  {tr_a:>7.3f}  {va_l:>8.4f}  {va_a:>7.3f}"
              f"  {lr_now:.2e}  [{time.time()-t0:.0f}s]")
        if va_a > best_val:
            best_val = va_a
            torch.save({"model": model.state_dict(),
                        "classes": classes,
                        "arch": arch,
                        "best_lr": best_lr,
                        "best_wd": best_wd}, ckpt_path)

    print(f"\nBest val acc: {best_val:.4f} — checkpoint at {ckpt_path}")

    # ---- training curves ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    print(f"Saved training curves -> {out/'training_curves.png'}")

    # save class list for later
    with open(out / "classes.json", "w") as f:
        json.dump(classes, f)
    return str(ckpt_path)


# ──────────────────────────────────────────────────────────────────────
#  TESTING
# ──────────────────────────────────────────────────────────────────────
def test(arch, dataset_dir, out_dir=".", batch_size=32):
    out = Path(out_dir)
    ckpt_path = out / f"{arch}_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path} — run --train first")

    ckpt    = torch.load(ckpt_path, map_location=DEVICE)
    classes = ckpt["classes"]
    model   = build_model(arch, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, _, test_loader, _ = get_loaders(dataset_dir, batch_size)
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels, all_imgs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            out_logits = model(imgs.to(DEVICE))
            preds = out_logits.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_imgs.extend(imgs)

    # ---- metrics ------------------------------------------------
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest accuracy: {acc:.4f}  ({acc*100:.1f} %)")
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))

    # ---- confusion matrix ----------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes))))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"{arch} — Confusion matrix  (test acc {acc*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130)
    print(f"Saved confusion matrix -> {out/'confusion_matrix.png'}")

    # ---- sample prediction grid ----------------------------------
    unnorm = transforms.Compose([
        transforms.Normalize(
            mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
            std =[1/s for s in IMAGENET_STD])])

    n_show = min(36, len(all_imgs))
    idx    = random.sample(range(len(all_imgs)), n_show)
    cols   = 6; rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= n_show:
            ax.axis("off"); continue
        j       = idx[i]
        img_t   = unnorm(all_imgs[j]).permute(1, 2, 0).numpy().clip(0, 1)
        correct = all_preds[j] == all_labels[j]
        color   = "green" if correct else "red"
        ax.imshow(img_t)
        ax.set_title(f"P: {classes[all_preds[j]]}\nT: {classes[all_labels[j]]}",
                     fontsize=7, color=color)
        ax.axis("off")
    fig.suptitle(f"{arch} — Sample predictions  (green=✓, red=✗)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sample_predictions.png", dpi=130)
    print(f"Saved sample grid -> {out/'sample_predictions.png'}")


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Transfer-learning phone-object classifier")
    ap.add_argument("--prepare",     action="store_true",
                    help="Split raw photos into train/val/test")
    ap.add_argument("--train",       action="store_true")
    ap.add_argument("--test",        action="store_true")
    ap.add_argument("--raw_dir",     default="raw_photos")
    ap.add_argument("--dataset_dir", default="dataset")
    ap.add_argument("--arch",        default="vgg19",
                    choices=["vgg19", "densenet121"])
    ap.add_argument("--epochs",      type=int, default=30)
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--out_dir",     default="output")
    args = ap.parse_args()

    if args.prepare:
        prepare_dataset(args.raw_dir, args.dataset_dir)
    if args.train:
        train(args.arch, args.dataset_dir,
              epochs=args.epochs, batch_size=args.batch_size,
              out_dir=args.out_dir)
    if args.test:
        test(args.arch, args.dataset_dir,
             out_dir=args.out_dir, batch_size=args.batch_size)
    if not (args.prepare or args.train or args.test):
        ap.print_help()


if __name__ == "__main__":
    main()