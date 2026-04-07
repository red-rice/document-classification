# Text-only classification results
#
# display format
# -----------------------------------------
# loss    |   ACC     |    wPRE   |    mREC
# -----------------------------------------
# margin* |    98.52  |    98.55  |    94.02

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import Paths, TrainConfig
from data import RVLCDIPOCRTextDataset, set_seed, resolve_split_limit
from model import BertDocClassifier


def _parse_samples(val: str):
    if val is None or val.lower() == "full":
        return None
    return int(val)


def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"]

            logits, _ = model(input_ids, attention_mask, token_type_ids)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--use",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Device: 'cpu' or 'gpu'.",
    )
    ap.add_argument("--test_samples", type=str, default=None, help="Number of test samples, or 'full'.")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    paths = Paths(
        project_root=project_root,
        qs_ocr_large_dir=project_root / "rvl-cdip-text",
        rvl_cdip_dir=project_root / "rvl-cdip",
        rvl_cdip_ood_text_dir=project_root / "rvl-cdip-o-text",
        train_list=project_root / "train.txt",
        val_list=project_root / "val.txt",
        test_list=project_root / "test.txt",
    )

    cfg = TrainConfig()
    if args.use is not None:
        cfg.device = "cuda" if args.use == "gpu" else "cpu"
    if args.test_samples is not None:
        cfg.test_samples = _parse_samples(args.test_samples)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]
    num_classes = int(ckpt.get("num_classes", 16))

    model = BertDocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_limit = resolve_split_limit(cfg, "test")
    test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.test_list,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=test_limit,
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=cfg.num_workers)

    labels, preds = run_inference(model, test_loader, device)

    acc  = accuracy_score(labels, preds) * 100
    wpre = precision_score(labels, preds, average="weighted", zero_division=0) * 100
    mrec = recall_score(labels, preds, average="macro", zero_division=0) * 100

    # derive label from checkpoint filename: bert_margin_star.pt -> margin*
    loss_label = Path(args.ckpt).stem.removeprefix("bert_").replace("_star", "*")

    header = f"{'loss':<10} | {'ACC':>8} | {'wPRE':>8} | {'mREC':>8}"
    sep = "-" * len(header)
    row = f"{loss_label:<10} | {acc:>8.2f} | {wpre:>8.2f} | {mrec:>8.2f}"

    print(sep)
    print(header)
    print(sep)
    print(row)
    print(sep)


if __name__ == "__main__":
    main()
