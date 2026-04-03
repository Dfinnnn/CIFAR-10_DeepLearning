# Streamlit UI for CIFAR-10 CNN inference
from pathlib import Path

# torch before pandas on Windows (see dataset.py — avoids WinError 1114 with native DLLs).
import torch
import pandas as pd
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

from dataset import load_idx_to_label
from model import CIFAR10CNN

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "cifar10_model.pth"
DEFAULT_LABELS = ROOT / "trainLabels.csv"


@st.cache_resource
def load_model(checkpoint_path: str, labels_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_label = load_idx_to_label(labels_csv)
    model = CIFAR10CNN(num_classes=len(idx_to_label)).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, idx_to_label, device


def inference_transform():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ]
    )


def main():
    st.set_page_config(page_title="CIFAR-10 CNN", layout="centered")
    st.title("CIFAR-10 image classifier")
    st.caption(
        "Upload any image; it is resized to 32×32 with the same normalization as test inference."
    )

    checkpoint = DEFAULT_CHECKPOINT
    labels_csv = DEFAULT_LABELS

    with st.sidebar:
        st.header("Paths")
        checkpoint = Path(
            st.text_input("Checkpoint (.pth)", value=str(DEFAULT_CHECKPOINT))
        )
        labels_csv = Path(st.text_input("Labels CSV", value=str(DEFAULT_LABELS)))

    if not labels_csv.is_file():
        st.error(f"Labels file not found: `{labels_csv}`.")
        st.stop()
    if not checkpoint.is_file():
        st.error(
            f"Checkpoint not found: `{checkpoint}`. Train with `train.py` or adjust the path in the sidebar."
        )
        st.stop()

    try:
        model, idx_to_label, device = load_model(str(checkpoint), str(labels_csv))
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("Upload a PNG or JPEG to run inference.")
        return

    image = Image.open(uploaded).convert("RGB")
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with c2:
        small = image.resize((32, 32), Image.Resampling.LANCZOS)
        st.image(small, caption="Model input size (32×32)", use_container_width=True)

    x = inference_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    k = int(probs.argmax())
    st.success(
        f"Predicted: **{idx_to_label[k]}** — confidence **{100.0 * probs[k]:.1f}%**"
    )

    names = [idx_to_label[i] for i in range(len(idx_to_label))]
    chart_df = (
        pd.DataFrame({"probability": probs}, index=names)
        .sort_values("probability", ascending=False)
    )
    st.subheader("Class probabilities")
    st.bar_chart(chart_df)


if __name__ == "__main__":
    main()
