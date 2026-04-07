import torch
from pathlib import Path
from data_multimodal import RVLCDIPLayoutLMv3Dataset
import numpy as np

def inspect_data():
    model_name = "microsoft/layoutlmv3-base"
    max_length = 128
    
    test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=Path("rvl-cdip"),
        ocr_box_root=Path("rvl-cdip-box"),
        split_file=Path("test.txt"),
        processor_name=model_name,
        max_length=max_length,
        debug_samples=10,
    )

    for i in range(10):
        img_path, tsv_path, label = test_ds.items[i]
        print(f"\nSample {i}: Label {label}")
        print(f"Image: {img_path}")
        print(f"TSV: {tsv_path}")
        
        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words = [ln.split("\t")[0] for ln in lines[1:6]] # First 5 words
            print(f"Words: {' '.join(words)}")

if __name__ == "__main__":
    inspect_data()
