"""
shrink_dataset.py
Copies exactly 50 images per class from raw_photos/
at FULL 224x224 quality (VGG19 native size).

Run this BEFORE v-3.py --prepare

Usage:
    python shrink_dataset.py
"""

import shutil
import random
from pathlib import Path
from PIL import Image

RAW_DIR     = Path("raw_photos")        # original full dataset
OUT_DIR     = Path("raw_photos_final")  # 50 imgs, 224x224
IMG_SIZE    = (224, 224)                # VGG19 native size - full quality
N_PER_CLASS = 50                        # 50 images per class

SEED = 42
random.seed(SEED)

def shrink():
    if not RAW_DIR.exists():
        raise SystemExit(f"Cannot find {RAW_DIR} — run reorganize_kitchenware.py first")

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes in {RAW_DIR}\n")
    print(f"  {'class':<15} {'available':>10} {'copied':>8} {'size':>10}")
    print("  " + "-" * 46)

    for cls in sorted(class_dirs):
        imgs = list(cls.glob("*.jpg")) + list(cls.glob("*.jpeg")) + \
               list(cls.glob("*.png")) + list(cls.glob("*.JPG"))

        chosen = random.sample(imgs, min(N_PER_CLASS, len(imgs)))

        dest_dir = OUT_DIR / cls.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in chosen:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                img.save(dest_dir / img_path.name, quality=95)
            except Exception as e:
                print(f"  [skip] {img_path.name}: {e}")

        print(f"  {cls.name:<15} {len(imgs):>10} {len(chosen):>8}   {IMG_SIZE[0]}x{IMG_SIZE[1]}")

    print(f"\nDataset saved to: {OUT_DIR.resolve()}")
    print(f"Total: {len(class_dirs)} classes x {N_PER_CLASS} images = {len(class_dirs)*N_PER_CLASS} images")
    print("\nNext steps:")
    print("  python v-3.py --prepare --raw_dir raw_photos_final --dataset_dir dataset_final")
    print("  python v-3.py --train   --arch vgg19 --dataset_dir dataset_final --out_dir output_final --epochs 30")
    print("  python v-3.py --test    --arch vgg19 --dataset_dir dataset_final --out_dir output_final")

if __name__ == "__main__":
    shrink()