"""
reorganize_kitchenware.py
Converts the Kaggle Kitchenware Competition dataset into the
raw_photos/class_name/ structure that v-3.py expects.

After downloading from:
  https://www.kaggle.com/competitions/kitchenware-classification/data

You will have:
  kitchenware-classification/
      images/          (all images)
      train.csv        (image_id, label)

Run:
  python reorganize_kitchenware.py
"""

import csv
import shutil
from pathlib import Path

# ── CONFIG ── change these if your folder names differ ──────────────
KITCHENWARE_DIR = Path("kitchenware-classification")   # unzipped folder
IMAGES_DIR      = KITCHENWARE_DIR / "images"
TRAIN_CSV       = KITCHENWARE_DIR / "train.csv"
OUT_DIR         = Path("raw_photos")
# ────────────────────────────────────────────────────────────────────

def main():
    if not IMAGES_DIR.exists():
        raise SystemExit(
            f"Could not find {IMAGES_DIR}\n"
            "Make sure you unzipped the Kaggle download and the folder is named "
            "'kitchenware-classification' in the same directory as this script."
        )

    counts = {}
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["Id"]          # e.g. "0000"
            label  = row["label"]       # e.g. "cup"

            # find the image file (could be .jpg or .JPG)
            src = None
            for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                candidate = IMAGES_DIR / f"{img_id}{ext}"
                if candidate.exists():
                    src = candidate
                    break

            if src is None:
                print(f"[warn] image not found for id={img_id}")
                continue

            dest_dir = OUT_DIR / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest_dir / src.name)
            counts[label] = counts.get(label, 0) + 1

    print("\n── Reorganized dataset ─────────────────────────────────")
    print(f"{'class':<15} {'images':>8}")
    for label, count in sorted(counts.items()):
        status = "OK" if count >= 25 else f"only {count} - add more"
        print(f"  {label:<15} {count:>6}   {status}")
    print(f"\nSaved to: {OUT_DIR.resolve()}")
    print("\nNext step:")
    print("  python v-3.py --prepare --raw_dir raw_photos --dataset_dir dataset")
    print("  python v-3.py --train   --arch vgg19 --dataset_dir dataset --out_dir output")
    print("  python v-3.py --test    --arch vgg19 --dataset_dir dataset --out_dir output")

if __name__ == "__main__":
    main()