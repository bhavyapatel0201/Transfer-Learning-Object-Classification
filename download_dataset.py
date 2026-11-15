"""
download_dataset.py  (v2 - uses Bing instead of Google)
Downloads 35 images per class using Bing Image Search via icrawler.
Bing works reliably unlike Google which blocks scrapers.

Usage:
    python download_dataset.py
"""

from icrawler.builtin import BingImageCrawler
from pathlib import Path

CLASSES = {
    "shoes":       "shoes footwear different types",
    "fork":        "fork kitchen utensil close up",
    "spoon":       "spoon kitchen utensil close up",
    "cup":         "cup mug coffee tea",
    "smartphone":  "smartphone mobile phone",
    "pan":         "frying pan kitchen cookware",
}

NUM_IMAGES = 35   # download 35, so after any failures you still have >=25
OUT_DIR    = Path("raw_photos")


def download():
    print("Downloading images using Bing (more reliable than Google)...\n")
    for class_name, search_term in CLASSES.items():
        save_dir = OUT_DIR / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading '{class_name}' -> {save_dir}")
        crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
            storage={"root_dir": str(save_dir)}
        )
        crawler.crawl(
            keyword=search_term,
            filters={"type": "photo"},
            max_num=NUM_IMAGES,
            file_idx_offset=0
        )

    # Summary
    print("\n── Download summary ────────────────────────────────────")
    total = 0
    all_ok = True
    for class_name in CLASSES:
        imgs = list((OUT_DIR / class_name).glob("*.jpg")) + \
               list((OUT_DIR / class_name).glob("*.jpeg")) + \
               list((OUT_DIR / class_name).glob("*.png"))
        count = len(imgs)
        status = "OK" if count >= 25 else f"need {25 - count} more"
        print(f"  {class_name:<15} {count:>3} images   {status}")
        total += count
        if count < 25:
            all_ok = False

    print(f"  {'TOTAL':<15} {total:>3} images")

    if all_ok:
        print("\nAll classes have >=25 images. Ready to train!")
        print("\nNext steps:")
        print("  python v-3.py --prepare --raw_dir raw_photos --dataset_dir dataset")
        print("  python v-3.py --train   --arch vgg19 --dataset_dir dataset --out_dir output")
        print("  python v-3.py --test    --arch vgg19 --dataset_dir dataset --out_dir output")
    else:
        print("\nSome classes have fewer than 25 images.")
        print("  Run the script again or add photos manually.")


if __name__ == "__main__":
    download()