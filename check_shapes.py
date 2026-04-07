import torch
from pathlib import Path
from data_multimodal import RVLCDIPLayoutLMv3Dataset

def check_shapes():
    model_name = "microsoft/layoutlmv3-base"
    max_length = 128
    
    test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=Path("rvl-cdip"),
        ocr_box_root=Path("rvl-cdip-box"),
        split_file=Path("test.txt"),
        processor_name=model_name,
        max_length=max_length,
        debug_samples=1,
    )

    batch = test_ds[0]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")

if __name__ == "__main__":
    check_shapes()
