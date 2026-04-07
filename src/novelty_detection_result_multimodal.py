# Multimodal novelty detection results (LayoutLMv3)
#
# display format
# --------------------------------------------------
# loss    |   FPR↓   |   AUC↑   |   F1↑   |   COV↑
# --------------------------------------------------
# margin* |    2.10  |   98.55  |   94.02  |   95.05

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig
from data import set_seed, resolve_split_limit
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from data_multimodal_ood import RVLCDIPOODLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier
from knn_ood_multimodal import (
    extract_embeddings_and_logits_multimodal,
    build_faiss_l2_index,
    estimate_threshold_theta,
    knn_star_predict,
    knn_predict_no_agreement,
    knn1_score_and_neighbor,
)
from metrics import compute_auc, compute_fpr_at_tpr95, compute_end_to_end_metrics


LOSS_ORDER = ["margin_star", "ce"]

LOSS_DISPLAY = {
    "margin_star": "Margin*",
    "ce": "CE",
}


def _parse_samples(val: str):
    if val is None or val.lower() == "full":
        return None
    return int(val)


def find_ckpts(ckpt_dir: Path) -> List[Tuple[str, Path]]:
    found = []
    for loss in LOSS_ORDER:
        p = ckpt_dir / f"layoutlmv3_{loss}.pt"
        if p.exists():
            found.append((loss, p))
    return found


def compute_scores(train_index, embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for i in range(embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
    return scores


def eval_one_checkpoint(project_root: Path, ckpt_path: Path, cfg: TrainConfig, tpr_target: float) -> dict:
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
    ood_box_root = project_root / "rvl-cdip-o-box"

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_name  = ckpt.get("model_name", "microsoft/layoutlmv3-base")
    max_length  = ckpt.get("max_length", cfg.max_length)
    num_classes = int(ckpt.get("num_classes", 16))

    model = LayoutLMv3DocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_limit = resolve_split_limit(cfg, "train")
    val_limit   = resolve_split_limit(cfg, "val")
    test_limit  = resolve_split_limit(cfg, "test")

    train_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir, ocr_box_root=ocr_box_root,
        split_file=paths.train_list, processor_name=model_name,
        max_length=max_length, debug_samples=train_limit,
    )
    val_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir, ocr_box_root=ocr_box_root,
        split_file=paths.val_list, processor_name=model_name,
        max_length=max_length, debug_samples=val_limit,
    )
    test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir, ocr_box_root=ocr_box_root,
        split_file=paths.test_list, processor_name=model_name,
        max_length=max_length, debug_samples=test_limit,
    )
    ood_ds = RVLCDIPOODLayoutLMv3Dataset(
        rvl_ood_root=project_root / "rvl-cdip-o",
        ocr_box_root=ood_box_root,
        split_dir=paths.rvl_cdip_ood_text_dir,
        processor_name=model_name,
        max_length=max_length,
        debug_samples=test_limit,
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    ood_loader   = DataLoader(ood_ds,   batch_size=8, shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    train_emb, _, train_y          = extract_embeddings_and_logits_multimodal(model, train_loader, device=str(device))
    val_emb,   _, _                = extract_embeddings_and_logits_multimodal(model, val_loader,   device=str(device))
    test_emb,  test_logits, test_y = extract_embeddings_and_logits_multimodal(model, test_loader,  device=str(device))
    ood_emb,   ood_logits,  _      = extract_embeddings_and_logits_multimodal(model, ood_loader,   device=str(device))

    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    theta = estimate_threshold_theta(
        train_index=train_index,
        val_embeddings=val_emb.astype(np.float32),
        tpr_target=tpr_target,
    )

    id_scores  = compute_scores(train_index, test_emb.astype(np.float32))
    ood_scores = compute_scores(train_index, ood_emb.astype(np.float32))

    auc = compute_auc(id_scores, ood_scores) * 100.0
    fpr = compute_fpr_at_tpr95(id_scores, ood_scores, tpr_target=tpr_target) * 100.0

    is_id_test_knn,      pred_knn      = knn_predict_no_agreement(train_index, test_emb, test_logits, theta)
    is_id_ood_knn,       _             = knn_predict_no_agreement(train_index, ood_emb,  ood_logits,  theta)
    is_id_test_knn_star, pred_knn_star = knn_star_predict(train_index, test_emb, test_logits, theta)
    is_id_ood_knn_star,  _             = knn_star_predict(train_index, ood_emb,  ood_logits,  theta)

    knn_m      = compute_end_to_end_metrics(test_y, is_id_test_knn,      pred_knn,      is_id_ood_knn)
    knn_star_m = compute_end_to_end_metrics(test_y, is_id_test_knn_star, pred_knn_star, is_id_ood_knn_star)

    return {
        "fpr":          fpr,
        "auc":          auc,
        "knn_f1":       knn_m.f1   * 100.0,
        "knn_cov":      knn_m.cov  * 100.0,
        "knn_star_f1":  knn_star_m.f1  * 100.0,
        "knn_star_cov": knn_star_m.cov * 100.0,
    }


def _print_table(title: str, results: list, f1_key: str, cov_key: str):
    header = f"{'#':<3} {'loss':<10} {'FPR↓':>8} {'AUC↑':>8} {'F1↑':>8} {'COV↑':>8}"
    sep    = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for i, (loss, m) in enumerate(results, start=1):
        print(
            f"{i:<3} {LOSS_DISPLAY.get(loss, loss):<10} "
            f"{m['fpr']:>8.2f} {m['auc']:>8.2f} {m[f1_key]:>8.2f} {m[cov_key]:>8.2f}"
        )
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Path to a single checkpoint .pt file.")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Scan dir for all layoutlmv3_*.pt checkpoints.")
    ap.add_argument(
        "--use",
        type=str,
        default=None,
        choices=["cpu", "gpu"],
        help="Device: 'cpu' or 'gpu'.",
    )
    ap.add_argument(
        "--tpr",
        type=float,
        default=0.95,
        help="TPR target for threshold estimation, e.g. 0.95 (default), 0.90, 0.99.",
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
        loss = ckpt_path.stem.removeprefix("layoutlmv3_")
        ckpts = [(loss, ckpt_path)]
    else:
        ckpt_dir = Path(args.ckpt_dir)
        ckpts = find_ckpts(ckpt_dir)
        if not ckpts:
            print(f"No checkpoints found in {ckpt_dir}.")
            print("Expected names: layoutlmv3_margin_star.pt, layoutlmv3_ce.pt, etc.")
            return

    results = []
    for loss, ckpt_path in ckpts:
        print(f"\n--- Evaluating {loss} ({ckpt_path.name}) ---")
        m = eval_one_checkpoint(Path(args.project_root), ckpt_path, cfg, tpr_target=args.tpr)
        results.append((loss, m))

    tpr_pct = int(args.tpr * 100)
    _print_table(f"KNN     (FPR@TPR{tpr_pct})", results, "knn_f1",      "knn_cov")
    _print_table(f"KNN*    (FPR@TPR{tpr_pct})", results, "knn_star_f1", "knn_star_cov")

    print("\nNotes:")
    print(f"  FPR = FPR@TPR{tpr_pct}  |  AUC from ID-test vs OOD scores")
    print(f"  KNN* adds consensus agreement (pred == 1NN label) for rejection")


if __name__ == "__main__":
    main()
