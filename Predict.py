"""
predict.py
Run a single image through the trained VGG19 model and show the prediction.

Usage:
    python predict.py --image path/to/your/image.jpg
    python predict.py --image cup.jpg
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Same transforms as test time ─────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def build_model(num_classes):
    model = models.vgg19(weights=None)
    in_feat = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feat, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def predict(image_path, checkpoint="output_final/vgg19_best.pt"):
    # ── Load checkpoint ───────────────────────────────────────────────
    ckpt    = torch.load(checkpoint, map_location=DEVICE)
    classes = ckpt["classes"]
    model   = build_model(len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Load and preprocess image ─────────────────────────────────────
    img     = Image.open(image_path).convert("RGB")
    tensor  = transform(img).unsqueeze(0).to(DEVICE)

    # ── Inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = torch.topk(probs, min(5, len(classes)))

    # ── Print results ─────────────────────────────────────────────────
    predicted_class = classes[top5_idx[0].item()]
    confidence      = top5_probs[0].item() * 100

    print(f"\n{'─'*40}")
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {predicted_class.upper()}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"{'─'*40}")
    print(f"\n  Top {min(5,len(classes))} predictions:")
    for prob, idx in zip(top5_probs, top5_idx):
        bar = "█" * int(prob.item() * 30)
        print(f"    {classes[idx.item()]:<12} {prob.item()*100:5.1f}%  {bar}")

    # ── Save visual output ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: image with prediction title
    axes[0].imshow(img)
    axes[0].set_title(
        f"Predicted: {predicted_class.upper()}  ({confidence:.1f}%)",
        fontsize=13,
        color="green" if confidence > 60 else "red"
    )
    axes[0].axis("off")

    # Right: bar chart of all class probabilities
    all_probs = probs.cpu().numpy()
    colors    = ["#2196F3" if c != predicted_class else "#4CAF50" for c in classes]
    bars = axes[1].barh(classes, all_probs * 100, color=colors)
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Class probabilities")
    axes[1].set_xlim(0, 100)
    for bar, p in zip(bars, all_probs):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f"{p*100:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    out_path = "prediction_result.png"
    fig.savefig(out_path, dpi=130)
    print(f"\n  Saved visual -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",      required=True,
                    help="Path to image file (jpg/png)")
    ap.add_argument("--checkpoint", default="output_final/vgg19_best.pt",
                    help="Path to model checkpoint")
    args = ap.parse_args()

    predict(args.image, args.checkpoint)