import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import TrainConfig, LossConfig, Paths
from data import set_seed, resolve_split_limit
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier
from sampler import MinPerClassBatchSampler
from loss import CustomMarginContrastiveLoss


def build_loader_for_loss(train_ds, cfg: TrainConfig, loss_name: str, device: torch.device):
    """
    CE uses normal shuffled batches.
    Margin* uses MinPerClassBatchSampler to ensure positives per class exist in batch.
    """
    if loss_name == "ce":
        return DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    if loss_name == "margin_star":
        labels = [y for _, _, y in train_ds.items]
        batch_sampler = MinPerClassBatchSampler(
            labels,
            cfg.batch_size,
            cfg.min_per_class,
            seed=cfg.seed,
        )
        return DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    raise ValueError(f"Unsupported loss: {loss_name}")


def build_val_loader(val_ds, cfg: TrainConfig, device: torch.device):
    return DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )


def evaluate_closed_set(model, loader, device: torch.device) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            bbox = batch["bbox"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels_t = batch["labels"].to(device, non_blocking=True)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )

            pred = logits.argmax(dim=1)
            correct += int((pred == labels_t).sum().item())
            total += int(labels_t.size(0))

    return correct / max(1, total)


def _parse_samples(val: str):
    """'full' -> None (no limit), else int."""
    if val is None or val.lower() == "full":
        return None
    return int(val)


def main():
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Memory allocated:", torch.cuda.memory_allocated(0))
        print("Memory reserved:", torch.cuda.memory_reserved(0))
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument(
        "--loss",
        type=str,
        default="margin_star",
        choices=["ce", "margin_star"],
    )
    ap.add_argument(
        "--use",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Device to use: 'cpu' or 'gpu' (maps to cuda).",
    )
    ap.add_argument(
        "--train_samples",
        type=str,
        default=None,
        help="Number of training samples, or 'full' for all.",
    )
    ap.add_argument(
        "--val_samples",
        type=str,
        default=None,
        help="Number of validation samples, or 'full' for all.",
    )
    ap.add_argument(
        "--test_samples",
        type=str,
        default=None,
        help="Number of test samples, or 'full' for all.",
    )
    args = ap.parse_args()

    cfg = TrainConfig()

    if args.use is not None:
        cfg.device = "cuda" if args.use == "gpu" else "cpu"

    if args.train_samples is not None:
        cfg.train_samples = _parse_samples(args.train_samples)
    if args.val_samples is not None:
        cfg.val_samples = _parse_samples(args.val_samples)
    if args.test_samples is not None:
        cfg.test_samples = _parse_samples(args.test_samples)

    loss_cfg = LossConfig()

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

    model_name = "microsoft/layoutlmv3-base"

    train_debug_samples = resolve_split_limit(cfg, "train")
    val_debug_samples = resolve_split_limit(cfg, "val")

    # When using a sample limit with margin_star, filter to classes with enough samples.
    allowed = None
    if cfg.train_samples is not None and args.loss == "margin_star":
        tmp_ds = RVLCDIPLayoutLMv3Dataset(
            rvl_root=paths.rvl_cdip_dir,
            ocr_box_root=ocr_box_root,
            split_file=paths.train_list,
            processor_name=model_name,
            max_length=cfg.max_length,
            debug_samples=train_debug_samples,
        )
        labels_all = [y for _, _, y in tmp_ds.items]
        cnt = Counter(labels_all)
        allowed = {lab for lab, c in cnt.items() if c >= cfg.min_per_class}
        if len(allowed) == 0:
            raise RuntimeError(
                "No label has enough samples for Margin*. "
                "Increase train_samples or reduce min_per_class."
            )

    train_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.train_list,
        processor_name=model_name,
        max_length=cfg.max_length,
        debug_samples=train_debug_samples,
        allowed_labels=allowed,
    )
    val_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.val_list,
        processor_name=model_name,
        max_length=cfg.max_length,
        debug_samples=val_debug_samples,
        allowed_labels=allowed,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Check OCR boxes and split paths.")
    if len(val_ds) == 0:
        raise RuntimeError("Validation dataset is empty. Check OCR boxes and split paths.")

    train_loader = build_loader_for_loss(train_ds, cfg, args.loss, device=device)
    val_loader = build_val_loader(val_ds, cfg, device=device)

    print("Loss:", args.loss)
    print("Train dataset size:", len(train_ds))
    print("Val dataset size:", len(val_ds))
    print("Resolved train_debug_samples:", train_debug_samples)
    print("Resolved val_debug_samples:", val_debug_samples)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Device:", device)
    print("Model:", model_name)
    print("Max length:", cfg.max_length)
    print("Epochs:", cfg.epochs)

    # Initial GPU memory info
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("\n--- GPU Memory Baseline ---")
        allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
        print(f"Initial GPU Memory: allocated={allocated_gb:.2f}GB, reserved={reserved_gb:.2f}GB\n")

    model = LayoutLMv3DocClassifier(model_name, num_classes=16).to(device)

    ce_criterion = torch.nn.CrossEntropyLoss()

    metric_criterion = None
    if args.loss == "margin_star":
        metric_criterion = CustomMarginContrastiveLoss(
            alpha=loss_cfg.alpha,
            beta=loss_cfg.beta,
            lam=loss_cfg.lam,
            eps=loss_cfg.eps,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = -1.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / f"layoutlmv3_{args.loss}.pt"

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_metric = 0.0

        # GPU memory monitoring (especially first epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{cfg.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            bbox = batch["bbox"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels_t = batch["labels"].to(device, non_blocking=True)

            if labels_t.min().item() < 0 or labels_t.max().item() > 15:
                raise ValueError(
                    f"Found label outside 0..15: min={labels_t.min().item()} max={labels_t.max().item()}"
                )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, h = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                )

                if args.loss == "ce":
                    ce_loss = ce_criterion(logits, labels_t)
                    metric_loss = torch.tensor(0.0, device=device)
                    loss = ce_loss
                elif args.loss == "margin_star":
                    # metric_criterion already computes CE + lam * L_margin
                    loss = metric_criterion(logits, h, labels_t)
                    # For logging, we can still compute CE separately if we want,
                    # but it's redundant. Let's just use the components from metric_criterion
                    # if possible, or just recompute CE for logging.
                    ce_loss = ce_criterion(logits, labels_t)
                    metric_loss = loss - ce_loss
                else:
                    raise ValueError(f"Unsupported loss: {args.loss}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_v = float(loss.detach().cpu())
            ce_v = float(ce_loss.detach().cpu())
            metric_v = float(metric_loss.detach().cpu()) if args.loss == "margin_star" else 0.0

            running_loss += loss_v
            running_ce += ce_v
            running_metric += metric_v

            pbar.set_postfix(
                {
                    "loss": f"{running_loss / (step + 1):.4f}",
                    "ce": f"{running_ce / (step + 1):.4f}",
                    "metric": f"{running_metric / (step + 1):.4f}",
                }
            )

        val_acc = evaluate_closed_set(model, val_loader, device=device)
        print(f"Epoch {epoch + 1}: val_acc = {val_acc * 100:.2f}")

        # GPU memory stats after epoch
        if device.type == "cuda":
            allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  GPU Memory: allocated={allocated_gb:.2f}GB, reserved={reserved_gb:.2f}GB, peak={peak_gb:.2f}GB")
            if epoch == 0 and peak_gb > 10.0:
                print(f"  ⚠️  WARNING: High GPU memory usage in first epoch! Consider reducing batch_size or max_length.")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "num_classes": 16,
                    "max_length": cfg.max_length,
                    "best_val_acc": best_acc,
                    "loss_name": args.loss,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")

    print(f"Training complete. Best val_acc = {best_acc * 100:.2f}")
    print(f"Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()