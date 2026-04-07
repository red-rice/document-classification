import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from model import BertDocClassifier
from config import TrainConfig
import numpy as np
from sklearn.metrics import accuracy_score

class RVLCDIPBoxTextDataset(Dataset):
    def __init__(self, ocr_box_root, split_file, tokenizer_name, max_length=128, debug_samples=None):
        self.ocr_box_root = Path(ocr_box_root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.items = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            if debug_samples:
                lines = lines[:debug_samples]
            for ln in lines:
                parts = ln.strip().split()
                rel_path = parts[0]
                label = int(parts[1])
                tsv_path = self.ocr_box_root / Path(rel_path).with_suffix(".tsv")
                self.items.append((tsv_path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tsv_path, label = self.items[idx]
        words = []
        if tsv_path.exists():
            with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for ln in lines[1:]:
                    words.append(ln.split("\t")[0])
        text = " ".join(words) if words else "[EMPTY]"
        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def test_bert_on_box():
    device = torch.device("cpu")
    ckpt_path = Path("checkpoints/bert_margin_star_full.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    model = BertDocClassifier(ckpt["model_name"], num_classes=16)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    ds = RVLCDIPBoxTextDataset(
        ocr_box_root="rvl-cdip-box",
        split_file="test.txt",
        tokenizer_name=ckpt["model_name"],
        max_length=ckpt["max_length"],
        debug_samples=100
    )
    
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            
            logits, _ = model(input_ids, attention_mask)
            pred = logits.argmax(dim=1).cpu().numpy()
            
            y_true.append(labels)
            y_pred.append(pred)
            
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"BERT Accuracy on rvl-cdip-box text: {acc*100:.2f}%")

if __name__ == "__main__":
    test_bert_on_box()
