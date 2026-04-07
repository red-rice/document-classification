import torch
from pathlib import Path
from model import BertDocClassifier
from model_multimodal import LayoutLMv3DocClassifier
from data import RVLCDIPOCRTextDataset
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from torch.utils.data import DataLoader

def check_norms():
    device = torch.device("cpu")
    
    # BERT
    bert_ckpt = torch.load("checkpoints/bert_margin_star_full.pt", map_location="cpu")
    bert_model = BertDocClassifier(bert_ckpt["model_name"], 16)
    bert_model.load_state_dict(bert_ckpt["model_state"])
    bert_model.eval()
    
    # LayoutLMv3
    mm_ckpt = torch.load("checkpoints/layoutlmv3_margin_star_full.pt", map_location="cpu")
    mm_model = LayoutLMv3DocClassifier(mm_ckpt["model_name"], 16)
    mm_model.load_state_dict(mm_ckpt["model_state"])
    mm_model.eval()
    
    # Get one sample
    bert_ds = RVLCDIPOCRTextDataset(Path("rvl-cdip-text"), Path("test.txt"), bert_ckpt["model_name"], 128, 1)
    mm_ds = RVLCDIPLayoutLMv3Dataset(Path("rvl-cdip"), Path("rvl-cdip-box"), Path("test.txt"), mm_ckpt["model_name"], 128, 1)
    
    bert_batch = next(iter(DataLoader(bert_ds, batch_size=1)))
    mm_batch = next(iter(DataLoader(mm_ds, batch_size=1)))
    
    with torch.no_grad():
        _, h_bert = bert_model(bert_batch["input_ids"], bert_batch["attention_mask"])
        _, h_mm = mm_model(mm_batch["input_ids"], mm_batch["attention_mask"], mm_batch["bbox"], mm_batch["pixel_values"])
        
        print(f"BERT h norm: {torch.norm(h_bert).item():.4f}")
        print(f"MM h norm: {torch.norm(h_mm).item():.4f}")
        
        # Check logits
        logits_bert, _ = bert_model(bert_batch["input_ids"], bert_batch["attention_mask"])
        logits_mm, _ = mm_model(mm_batch["input_ids"], mm_batch["attention_mask"], mm_batch["bbox"], mm_batch["pixel_values"])
        
        print(f"BERT logits: {logits_bert[0][:5]}")
        print(f"MM logits: {logits_mm[0][:5]}")

if __name__ == "__main__":
    check_norms()
