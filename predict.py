import argparse

# torch before pandas on Windows (see dataset.py — avoids WinError 1114 with native DLLs).
import torch
import pandas as pd

from dataset import get_test_dataloader, load_idx_to_label
from model import CIFAR10CNN


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 test inference and Kaggle submission.csv")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cifar10_model.pth",
        help="Path to trained state_dict (.pth)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="test",
        help="Folder of unlabeled test PNGs ({id}.png)",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="trainLabels.csv",
        help="Training labels CSV (builds same class index order as training)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.csv",
        help="Output submission path",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_label = load_idx_to_label(args.labels_csv)

    model = CIFAR10CNN(num_classes=len(idx_to_label)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = get_test_dataloader(
        args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    rows = []
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            for image_id, pred_idx in zip(ids, preds):
                rows.append(
                    {"id": int(image_id) if str(image_id).isdigit() else image_id, "label": idx_to_label[pred_idx]}
                )

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
