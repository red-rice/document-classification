import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from config import Paths, TrainConfig, LossConfig
from data import RVLCDIPOCRTextDataset, set_seed, resolve_split_limit
from sampler import MinPerClassBatchSampler
from model import BertDocClassifier

# Keep your Margin* loss exactly
from loss import CustomMarginContrastiveLoss

# Extra losses (CE, fixed-margin, SCL, weighted CE)
from losses_extra import CELoss, FixedMarginLoss, SCLLoss, WeightedCELoss


def build_criterion(
    loss_name: str,
    loss_cfg: LossConfig,
    train_ds: RVLCDIPOCRTextDataset,
    num_classes: int,
    device,
):
    """
    Returns a callable criterion with signature (logits, h, labels) -> loss.
    """
    if loss_name == "margin_star":
        return CustomMarginContrastiveLoss(
            alpha=loss_cfg.alpha,
            beta=loss_cfg.beta,
            lam=loss_cfg.lam,
            eps=loss_cfg.eps,
        )

    if loss_name == "margin":
        return FixedMarginLoss(
            alpha=loss_cfg.alpha,
            beta=loss_cfg.beta,
            lam=loss_cfg.lam,
            eps=loss_cfg.eps,
        )

    if loss_name == "scl":
        return SCLLoss(temperature=0.1, lam=1.0)

    if loss_name == "weight":
        counts = Counter([y for _, y in train_ds.items])
        weights = np.zeros((num_classes,), dtype=np.float32)
        for c in range(num_classes):
            weights[c] = 1.0 / max(1, counts.get(c, 1))
        weights = weights / weights.mean()
        class_w = torch.tensor(weights, dtype=torch.float32, device=device)
        return WeightedCELoss(class_weights=class_w)

    if loss_name == "ce":
        return CELoss()

    raise ValueError(f"Unknown loss: {loss_name}")


def _parse_samples(val: str):
    """'full' -> None (no limit), else int."""
    if val is None or val.lower() == "full":
        return None
    return int(val)


def evaluate_closed_set(model, loader, device: torch.device) -> float:
    """Evaluate model accuracy on validation/test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels_t = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask, token_type_ids)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels_t).sum().item())
            total += int(labels_t.size(0))

    return correct / max(1, total)


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
        choices=["margin_star", "margin", "scl", "weight", "ce"],
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
        if args.use == "gpu":
            if torch.cuda.is_available():
                cfg.device = "cuda"
            else:
                print("Warning: CUDA is requested but not available. Falling back to CPU.")
                cfg.device = "cpu"
        else:
            cfg.device = "cpu"

    if args.train_samples is not None:
        cfg.train_samples = _parse_samples(args.train_samples)
    if args.val_samples is not None:
        cfg.val_samples = _parse_samples(args.val_samples)
    if args.test_samples is not None:
        cfg.test_samples = _parse_samples(args.test_samples)

    loss_cfg = LossConfig()

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_debug_samples = resolve_split_limit(cfg, "train")
    val_debug_samples = resolve_split_limit(cfg, "val")

    # When using a sample limit with margin_star, filter to classes with enough samples.
    allowed = None
    if cfg.train_samples is not None and args.loss == "margin_star":
        tmp_ds = RVLCDIPOCRTextDataset(
            qs_root=paths.qs_ocr_large_dir,
            split_file=paths.train_list,
            tokenizer_name=cfg.model_name,
            max_length=cfg.max_length,
            debug_samples=train_debug_samples,
        )
        labels_all = [int(tmp_ds.items[i][1]) for i in range(len(tmp_ds))]
        cnt = Counter(labels_all)
        allowed = {lab for lab, c in cnt.items() if c >= cfg.min_per_class}
        if len(allowed) == 0:
            raise RuntimeError(
                "Subset has no label with >= min_per_class samples. "
                "Increase train_samples or reduce min_per_class."
            )

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.train_list,
        tokenizer_name=cfg.model_name,
        max_length=cfg.max_length,
        debug_samples=train_debug_samples,
        allowed_labels=allowed,
    )

    val_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir,
        split_file=paths.val_list,
        tokenizer_name=cfg.model_name,
        max_length=cfg.max_length,
        debug_samples=val_debug_samples,
        allowed_labels=allowed,
    )

    labels = [int(y) for (_, y) in train_ds.items]

    # RVL-CDIP is always 16 classes (0..15)
    num_classes = 16

    batch_sampler = MinPerClassBatchSampler(
        labels,
        cfg.batch_size,
        cfg.min_per_class,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    print("Loss:", args.loss)
    print("Train dataset size:", len(train_ds))
    print("Val dataset size:", len(val_ds))
    print("Num unique labels:", len(set([y for _, y in train_ds.items])))
    print("Min label:", min([y for _, y in train_ds.items]), "Max label:", max([y for _, y in train_ds.items]))
    print("Train loader len (num batches):", len(train_loader))
    print("Val loader len (num batches):", len(val_loader))
    print("Resolved train_debug_samples:", train_debug_samples)
    print("Resolved val_debug_samples:", val_debug_samples)

    model = BertDocClassifier(cfg.model_name, num_classes=num_classes).to(device)
    criterion = build_criterion(args.loss, loss_cfg, train_ds, num_classes, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"bert_{args.loss}.pt"

    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{cfg.epochs}")
        for step, batch in enumerate(pbar):
            if step == 0:
                print(f'>>> entered training loop, got {step+1} batch')

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels_t = batch["labels"].to(device)

            if labels_t.min().item() < 0 or labels_t.max().item() > 15:
                raise ValueError(
                    f"Found label outside 0..15: min={labels_t.min().item()} max={labels_t.max().item()}"
                )

            optim.zero_grad(set_to_none=True)
            logits, h = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, h, labels_t)
            loss.backward()
            optim.step()

            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        # Validation after each epoch
        val_acc = evaluate_closed_set(model, val_loader, device=device)
        print(f"Epoch {epoch + 1}: val_acc = {val_acc * 100:.2f}%")

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "model_name": cfg.model_name,
                    "max_length": cfg.max_length,
                    "best_val_acc": best_acc,
                    "loss_name": args.loss,
                },
                ckpt_path,
            )
            print(f"✓ Saved best checkpoint (val_acc={best_acc * 100:.2f}%) to {ckpt_path}")

        model.train()  # Resume training mode

    print(f"\nTraining complete. Best val_acc = {best_acc * 100:.2f}%")
    print(f"Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()