import torch

def inspect_ckpt(path):
    print(f"\nInspecting {path}")
    try:
        ckpt = torch.load(path, map_location="cpu")
        print(f"Keys: {ckpt.keys()}")
        for k in ["model_name", "num_classes", "max_length", "best_val_acc", "loss_name"]:
            if k in ckpt:
                print(f"{k}: {ckpt[k]}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

inspect_ckpt("checkpoints/bert_margin_star_full.pt")
inspect_ckpt("checkpoints/layoutlmv3_margin_star_full.pt")
