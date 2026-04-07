import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig
from data import (
    RVLCDIPOCRTextDataset,
    RVLCDIPOODTextDataset,
    set_seed,
    resolve_split_limit,
)
from model import BertDocClassifier
from knn_ood import (
    extract_embeddings_and_logits,
    build_faiss_l2_index,
    knn1_score_and_neighbor,
)
from metrics import compute_end_to_end_metrics


LOSS_ORDER = ["margin", "margin_star", "scl", "weight", "ce"]

LOSS_DISPLAY = {
    "margin":      "Margin",
    "margin_star": "Margin*",
    "scl":         "SCL",
    "weight":      "Weight",
    "ce":          "CE",
}


def _parse_samples(val: str):
    if val is None or val.lower() == "full":
        return None
    return int(val)


def find_ckpts(ckpt_dir: Path) -> List[Tuple[str, Path]]:
    found = []
    for loss in LOSS_ORDER:
        p = ckpt_dir / f"bert_{loss}.pt"
        if p.exists():
            found.append((loss, p))
    return found


def compute_scores_preds_nns(train_index, emb: np.ndarray, logits: np.ndarray):
    n         = emb.shape[0]
    scores    = np.zeros(n, dtype=np.float32)
    pred      = logits.argmax(axis=1).astype(np.int64)
    nn_labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        s, _, nn    = knn1_score_and_neighbor(train_index, emb[i])
        scores[i]    = s
        nn_labels[i] = nn
    return scores, pred, nn_labels


def compute_metrics(theta, y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn):
    accept_id  = (scores     >= theta) & (pred     == nn)
    accept_ood = (ood_scores >= theta) & (ood_pred == ood_nn)

    y_pred_id = np.full_like(y_true, -1)
    y_pred_id[accept_id] = pred[accept_id]

    return compute_end_to_end_metrics(
        y_true_id=y_true,
        is_id_pred_idset=accept_id,
        y_pred_idset=y_pred_id,
        is_id_pred_oodset=accept_ood,
    )


def find_theta(target_precision, y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn):
    thresholds = np.unique(np.concatenate([scores, ood_scores]))
    best = None

    for t in thresholds:
        m = compute_metrics(t, y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn)
        if m.pre >= target_precision:
            if best is None or m.rec > best[1].rec:
                best = (float(t), m)

    if best is None:
        t    = float(thresholds.max())
        best = (t, compute_metrics(t, y_true, scores, pred, nn, ood_scores, ood_pred, ood_nn))

    return best


def extract_ckpt_data(project_root: Path, ckpt_path: Path, cfg: TrainConfig):
    """Extract embeddings/scores for a checkpoint once; reuse across all thresholds."""
    paths = Paths(
        project_root=project_root,
        qs_ocr_large_dir=project_root / "rvl-cdip-text",
        rvl_cdip_dir=project_root / "rvl-cdip",
        rvl_cdip_ood_text_dir=project_root / "rvl-cdip-o-text",
        train_list=project_root / "train.txt",
        val_list=project_root / "val.txt",
        test_list=project_root / "test.txt",
    )

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt       = torch.load(str(ckpt_path), map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]

    model = BertDocClassifier(model_name, num_classes=ckpt.get("num_classes", 16)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_limit = resolve_split_limit(cfg, "train")
    val_limit   = resolve_split_limit(cfg, "val")
    test_limit  = resolve_split_limit(cfg, "test")

    train_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir, split_file=paths.train_list,
        tokenizer_name=model_name, max_length=max_length, debug_samples=train_limit,
    )
    val_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir, split_file=paths.val_list,
        tokenizer_name=model_name, max_length=max_length, debug_samples=val_limit,
    )
    test_ds = RVLCDIPOCRTextDataset(
        qs_root=paths.qs_ocr_large_dir, split_file=paths.test_list,
        tokenizer_name=model_name, max_length=max_length, debug_samples=test_limit,
    )
    ood_ds = RVLCDIPOODTextDataset(
        ood_text_dir=paths.rvl_cdip_ood_text_dir,
        tokenizer_name=model_name, max_length=max_length, debug_samples=test_limit,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=cfg.num_workers)
    ood_loader   = DataLoader(ood_ds,   batch_size=16, shuffle=False, num_workers=cfg.num_workers)

    train_emb, _, train_y          = extract_embeddings_and_logits(model, train_loader, device=str(device))
    val_emb,   val_logits,  val_y  = extract_embeddings_and_logits(model, val_loader,   device=str(device))
    test_emb,  test_logits, test_y = extract_embeddings_and_logits(model, test_loader,  device=str(device))
    ood_emb,   ood_logits,  _      = extract_embeddings_and_logits(model, ood_loader,   device=str(device))

    index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    val_scores,  val_pred,  val_nn  = compute_scores_preds_nns(index, val_emb.astype(np.float32),  val_logits)
    ood_scores,  ood_pred,  ood_nn  = compute_scores_preds_nns(index, ood_emb.astype(np.float32),  ood_logits)
    test_scores, test_pred, test_nn = compute_scores_preds_nns(index, test_emb.astype(np.float32), test_logits)

    return dict(
        val_y=val_y, val_scores=val_scores, val_pred=val_pred, val_nn=val_nn,
        ood_scores=ood_scores, ood_pred=ood_pred, ood_nn=ood_nn,
        test_y=test_y, test_scores=test_scores, test_pred=test_pred, test_nn=test_nn,
    )


def evaluate_loss(data: dict, target_pre: float):
    theta, _ = find_theta(
        target_pre,
        data["val_y"], data["val_scores"], data["val_pred"], data["val_nn"],
        data["ood_scores"], data["ood_pred"], data["ood_nn"],
    )
    return compute_metrics(
        theta,
        data["test_y"], data["test_scores"], data["test_pred"], data["test_nn"],
        data["ood_scores"], data["ood_pred"], data["ood_nn"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Path to a single checkpoint .pt file.")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Scan dir for all bert_*.pt checkpoints.")
    ap.add_argument(
        "--use",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Device: 'cpu' or 'gpu'.",
    )
    ap.add_argument(
        "--target_pre",
        type=float,
        default=99.0,
        help="Target precision in percent (default 99.0). Can pass multiple: --target_pre 99.0 98.8 98.0",
        nargs="+",
    )
    ap.add_argument("--train_samples", type=str, default=None, help="Training samples or 'full'.")
    ap.add_argument("--val_samples",   type=str, default=None, help="Validation samples or 'full'.")
    ap.add_argument("--test_samples",  type=str, default=None, help="Test/OOD samples or 'full'.")
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

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        loss = ckpt_path.stem.removeprefix("bert_")
        ckpts = [(loss, ckpt_path)]
    else:
        ckpt_dir = Path(args.ckpt_dir)
        ckpts = find_ckpts(ckpt_dir)
        if not ckpts:
            print(f"No checkpoints found in {ckpt_dir}.")
            print("Expected names: bert_margin_star.pt, bert_ce.pt, etc.")
            return

    project_root = Path(args.project_root)
    thresholds   = [t / 100.0 for t in args.target_pre]

    print(f"\nSampling: train={cfg.train_samples or 'full'}, val={cfg.val_samples or 'full'}, test={cfg.test_samples or 'full'}")

    # Extract embeddings once per checkpoint
    ckpt_data = []
    for loss, ckpt_path in ckpts:
        print(f"\n--- Extracting embeddings: {loss} ({ckpt_path.name}) ---")
        data = extract_ckpt_data(project_root, ckpt_path, cfg)
        ckpt_data.append((loss, data))

    # Print results for each threshold (fast — no re-extraction)
    header = f"{'#':<3} {'loss':<10} {'PRE↑':>8} {'REC↑':>8} {'F1↑':>8} {'COV↑':>8}"
    sep    = "-" * len(header)

    for thr in thresholds:
        print(f"\nTarget PRE ≥ {thr*100:.1f}%")
        print(sep)
        print(header)
        print(sep)

        for i, (loss, data) in enumerate(ckpt_data, 1):
            m = evaluate_loss(data, thr)
            print(
                f"{i:<3} {LOSS_DISPLAY.get(loss, loss):<10} "
                f"{m.pre*100:>8.2f} {m.rec*100:>8.2f} {m.f1*100:>8.2f} {m.cov*100:>8.2f}"
            )

        print(sep)


if __name__ == "__main__":
    main()
