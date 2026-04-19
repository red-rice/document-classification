import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import Paths, TrainConfig
from data import RVLCDIPOCRTextDataset, set_seed, resolve_split_limit
from model import BertDocClassifier


def _parse_samples(val: str):
    if val is None or str(val).lower() == "full":
        return None
    return int(val)


def run_inference_with_loss(model, loader, device):
    model.eval()
    ce = torch.nn.CrossEntropyLoss(reduction="sum")

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask, token_type_ids)

            loss = ce(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item())
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, total)
    return all_labels, all_preds, avg_loss


def compute_metrics(labels, preds):
    labels_np = np.array(labels)
    preds_np = np.array(preds)

    acc = accuracy_score(labels_np, preds_np) * 100
    wpre = precision_score(labels_np, preds_np, average="weighted", zero_division=0) * 100
    mrec = recall_score(labels_np, preds_np, average="macro", zero_division=0) * 100
    correct = int((labels_np == preds_np).sum())
    error = 100 - acc

    return {
        "ACC": acc,
        "wPRE": wpre,
        "mREC": mrec,
        "correct_predictions": correct,
        "error": error,
    }


def build_dataset(paths, split_file, model_name, max_length, limit):
    return RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=split_file,
        tokenizer_name=model_name,
        max_length=max_length,
        debug_samples=limit,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use", choices=["cpu", "gpu"], default=None)

    ap.add_argument("--train_samples", type=str, default=None)
    ap.add_argument("--val_samples", type=str, default=None)
    ap.add_argument("--test_samples", type=str, default=None)

    ap.add_argument("--save_dir", default="eval_outputs_text")
    args = ap.parse_args()

    cfg = TrainConfig()

    if args.use:
        cfg.device = "cuda" if args.use == "gpu" else "cpu"

    if args.train_samples or args.val_samples or args.test_samples:
        cfg.debug_samples = None

    if args.train_samples:
        cfg.train_samples = _parse_samples(args.train_samples)
    if args.val_samples:
        cfg.val_samples = _parse_samples(args.val_samples)
    if args.test_samples:
        cfg.test_samples = _parse_samples(args.test_samples)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

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

    ckpt = torch.load(args.ckpt, map_location="cpu")

    model = BertDocClassifier(
        ckpt["model_name"],
        num_classes=int(ckpt.get("num_classes", 16)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    max_length = ckpt["max_length"]

    train_limit = resolve_split_limit(cfg, "train")
    val_limit = resolve_split_limit(cfg, "val")
    test_limit = resolve_split_limit(cfg, "test")

    val_ds = build_dataset(paths, paths.val_list, ckpt["model_name"], max_length, val_limit)
    test_ds = build_dataset(paths, paths.test_list, ckpt["model_name"], max_length, test_limit)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    val_y, val_pred, val_loss = run_inference_with_loss(model, val_loader, device)
    test_y, test_pred, test_loss = run_inference_with_loss(model, test_loader, device)

    val_metrics = compute_metrics(val_y, val_pred)
    val_metrics["val_loss"] = val_loss
    test_metrics = compute_metrics(test_y, test_pred)
    test_metrics["test_loss"] = test_loss

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.ckpt).stem

    np.save(save_dir / f"{stem}_y_true.npy", np.array(test_y))
    np.save(save_dir / f"{stem}_y_pred_text.npy", np.array(test_pred))

    print("\nSPLIT:")
    print(f"{train_limit} / {val_limit} / {test_limit}")

    print("\nText-only Test:")
    print(f"ACC = {test_metrics['ACC']:.2f}")
    print(f"wPRE = {test_metrics['wPRE']:.2f}")
    print(f"mREC = {test_metrics['mREC']:.2f}")
    print(f"Correct predictions = {test_metrics['correct_predictions']}")
    # print(f"Test loss = {test_loss:.6f}")
    print(f"Test loss = {test_metrics['test_loss']:.6f}")
    print(f"Test error = {test_metrics['error']:.2f}")

    print("\nValidation:")
    print(f"ACC = {val_metrics['ACC']:.2f}")
    print(f"wPRE = {val_metrics['wPRE']:.2f}")
    print(f"mREC = {val_metrics['mREC']:.2f}")
    print(f"Correct predictions = {val_metrics['correct_predictions']}")
    # print(f"Validation loss = {val_loss:.6f}")
    print(f"Validation loss = {val_metrics['val_loss']:.6f}")
    print(f"Validation error = {val_metrics['error']:.2f}")


if __name__ == "__main__":
    main()