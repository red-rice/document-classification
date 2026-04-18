import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import Paths, TrainConfig
from data import set_seed, resolve_split_limit
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier


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
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            bbox = batch["bbox"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )

            loss = ce(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item())
            total += int(labels.size(0))

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, total)
    return all_labels, all_preds, avg_loss


def compute_classification_metrics(labels, preds):
    labels_np = np.array(labels, dtype=np.int64)
    preds_np = np.array(preds, dtype=np.int64)

    acc = accuracy_score(labels_np, preds_np) * 100.0
    wpre = precision_score(labels_np, preds_np, average="weighted", zero_division=0) * 100.0
    mrec = recall_score(labels_np, preds_np, average="macro", zero_division=0) * 100.0
    correct = int((labels_np == preds_np).sum())
    total = int(labels_np.shape[0])
    error_rate = 100.0 - acc

    return {
        "ACC": acc,
        "wPRE": wpre,
        "mREC": mrec,
        "correct_predictions": correct,
        "total_samples": total,
        "error_rate": error_rate,
    }


def build_dataset(paths, ocr_box_root, split_file, model_name, max_length, sample_limit):
    return RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=split_file,
        processor_name=model_name,
        max_length=max_length,
        debug_samples=sample_limit,
    )


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
    ap.add_argument("--train_samples", type=str, default=None, help="Train samples or 'full'.")
    ap.add_argument("--val_samples", type=str, default=None, help="Val samples or 'full'.")
    ap.add_argument("--test_samples", type=str, default=None, help="Test samples or 'full'.")
    ap.add_argument("--batch_size", type=int, default=None, help="Override evaluation batch size.")
    ap.add_argument("--save_dir", type=str, default="eval_outputs_multimodal")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.use is not None:
        cfg.device = "cuda" if args.use == "gpu" else "cpu"

    if args.train_samples is not None or args.val_samples is not None or args.test_samples is not None:
        cfg.debug_samples = None
    if args.train_samples is not None:
        cfg.train_samples = _parse_samples(args.train_samples)
    if args.val_samples is not None:
        cfg.val_samples = _parse_samples(args.val_samples)
    if args.test_samples is not None:
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
    ocr_box_root = project_root / "rvl-cdip-box"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt.get("model_name", "microsoft/layoutlmv3-base")
    max_length = ckpt.get("max_length", cfg.max_length)
    num_classes = int(ckpt.get("num_classes", 16))

    model = LayoutLMv3DocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_limit = resolve_split_limit(cfg, "train")
    val_limit = resolve_split_limit(cfg, "val")
    test_limit = resolve_split_limit(cfg, "test")

    val_ds = build_dataset(paths, ocr_box_root, paths.val_list, model_name, max_length, val_limit)
    test_ds = build_dataset(paths, ocr_box_root, paths.test_list, model_name, max_length, test_limit)

    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(val_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)

    val_labels, val_preds, val_loss = run_inference_with_loss(model, val_loader, device)
    val_metrics = compute_classification_metrics(val_labels, val_preds)
    val_metrics["val_loss"] = val_loss
    val_metrics["validation_error"] = 100.0 - val_metrics["ACC"]

    test_labels, test_preds, test_loss = run_inference_with_loss(model, test_loader, device)
    test_metrics = compute_classification_metrics(test_labels, test_preds)
    test_metrics["test_loss"] = test_loss
    test_metrics["test_error"] = 100.0 - test_metrics["ACC"]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.ckpt).stem

    np.save(save_dir / f"{stem}_y_true.npy", np.array(test_labels, dtype=np.int64))
    np.save(save_dir / f"{stem}_y_pred_multi.npy", np.array(test_preds, dtype=np.int64))
    np.save(save_dir / f"{stem}_val_y_true.npy", np.array(val_labels, dtype=np.int64))
    np.save(save_dir / f"{stem}_val_y_pred_multi.npy", np.array(val_preds, dtype=np.int64))

    with open(save_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": {
                    "train_samples": train_limit if train_limit is not None else "full",
                    "val_samples": val_limit if val_limit is not None else "full",
                    "test_samples": test_limit if test_limit is not None else "full",
                },
                "checkpoint": str(args.ckpt),
                "validation": val_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    print("\nSPLIT:")
    print(
        f"{train_limit if train_limit is not None else 'full'} / "
        f"{val_limit if val_limit is not None else 'full'} / "
        f"{test_limit if test_limit is not None else 'full'}"
    )

    print("\nMultimodal Test:")
    print(f"ACC = {test_metrics['ACC']:.2f}")
    print(f"wPRE = {test_metrics['wPRE']:.2f}")
    print(f"mREC = {test_metrics['mREC']:.2f}")
    print(f"Correct predictions = {test_metrics['correct_predictions']}")
    print(f"Error rate = {test_metrics['error_rate']:.2f}")
    print(f"Test loss = {test_metrics['test_loss']:.6f}")
    print(f"Test error = {test_metrics['test_error']:.2f}")

    print("\nValidation:")
    print(f"ACC = {val_metrics['ACC']:.2f}")
    print(f"wPRE = {val_metrics['wPRE']:.2f}")
    print(f"mREC = {val_metrics['mREC']:.2f}")
    print(f"Correct predictions = {val_metrics['correct_predictions']}")
    print(f"Error rate = {val_metrics['error_rate']:.2f}")
    print(f"Validation loss = {val_metrics['val_loss']:.6f}")
    print(f"Validation error = {val_metrics['validation_error']:.2f}")

    print("\nPrediction files saved:")
    print(save_dir / f"{stem}_y_true.npy")
    print(save_dir / f"{stem}_y_pred_multi.npy")
    print(save_dir / f"{stem}_val_y_true.npy")
    print(save_dir / f"{stem}_val_y_pred_multi.npy")
    print(save_dir / f"{stem}_metrics.json")


if __name__ == "__main__":
    main()