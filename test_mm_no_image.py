import torch
from pathlib import Path
from model_multimodal import LayoutLMv3DocClassifier
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

def test_mm_no_image():
    device = torch.device("cpu")
    mm_ckpt = torch.load("checkpoints/layoutlmv3_margin_star_full.pt", map_location="cpu")
    model = LayoutLMv3DocClassifier(mm_ckpt["model_name"], 16)
    model.load_state_dict(mm_ckpt["model_state"])
    model.to(device)
    model.eval()
    
    ds = RVLCDIPLayoutLMv3Dataset(Path("rvl-cdip"), Path("rvl-cdip-box"), Path("test.txt"), mm_ckpt["model_name"], 128, 100)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            labels = batch["labels"].numpy()
            
            # Pass None for pixel_values
            logits, _ = model(input_ids, attention_mask, bbox, None)
            pred = logits.argmax(dim=1).cpu().numpy()
            
            y_true.append(labels)
            y_pred.append(pred)
            
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"LayoutLMv3 Accuracy WITHOUT image: {acc*100:.2f}%")

if __name__ == "__main__":
    test_mm_no_image()
