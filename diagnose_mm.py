import torch
from pathlib import Path
from torch.utils.data import DataLoader
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier
from config import TrainConfig
import numpy as np

def diag():
    device = torch.device("cpu")
    ckpt_path = Path("checkpoints/layoutlmv3_margin_star_full.pt")
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]
    num_classes = ckpt.get("num_classes", 16)

    model = LayoutLMv3DocClassifier(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=Path("rvl-cdip"),
        ocr_box_root=Path("rvl-cdip-box"),
        split_file=Path("test.txt"),
        processor_name=model_name,
        max_length=max_length,
        debug_samples=10,
    )

    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"{'Sample':<10} {'True':<5} {'Pred':<5} {'Logits'}")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            label = batch["labels"].item()

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            pred = logits.argmax(dim=1).item()
            logits_np = logits.numpy()[0]
            
            # Print top 3 logits
            top3_idx = logits_np.argsort()[-3:][::-1]
            top3_val = logits_np[top3_idx]
            top3_str = ", ".join([f"{idx}:{val:.2f}" for idx, val in zip(top3_idx, top3_val)])

            print(f"{i:<10} {label:<5} {pred:<5} {top3_str}")

if __name__ == "__main__":
    diag()
