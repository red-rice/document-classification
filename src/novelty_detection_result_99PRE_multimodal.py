# Multimodal novelty detection results at target precision (LayoutLMv3)
#
# display format
# ----------------------------------------------------------
# #   loss       PRE↑     REC↑      F1↑     COV↑   Met?
# ----------------------------------------------------------
# 1   Margin*   99.10    62.30    76.50    45.20      Y

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

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
    knn1_score_and_neighbor,
)
from metrics import compute_end_to_end_metrics


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


def compute_scores_preds_nns(train_index, emb: np.ndarray, logits: np.ndarray):
    n = emb.shape[0]
    scores    = np.zeros(n, dtype=np.float32)
    pred      = logits.argmax(axis=1).astype(np.int64)
    nn_labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        s, _, nn = knn1_score_and_neighbor(train_index, emb[i])
        scores[i]    = s
        nn_labels[i] = nn
    return scores, pred, nn_labels


def compute_pipeline_metrics(
    theta: float,
    y_true: np.ndarray,
    id_scores: np.ndarray, id_pred: np.ndarray, id_nn: np.ndarray,
    ood_scores: np.ndarray, ood_pred: np.ndarray, ood_nn: np.ndarray,
    use_agreement: bool,
) -> Dict:
    if use_agreement:
        accept_id  = (id_scores  >= theta) & (id_pred  == id_nn)
        accept_ood = (ood_scores >= theta) & (ood_pred == ood_nn)
    else:
        accept_id  = (id_scores  >= theta)
        accept_ood = (ood_scores >= theta)

    tp         = int(accept_id.sum())
    fn         = int((~accept_id).sum())
    fp         = int(accept_ood.sum())
    tn         = int((~accept_ood).sum())
    tp_correct = int(((id_pred == y_true) & accept_id).sum())

    pre = tp_correct / max(1, tp + fp)
    rec = tp_correct / max(1, tp + fn)
    f1  = 0.0 if (pre + rec) == 0 else 2.0 * pre * rec / (pre + rec)
    cov = tp_correct / max(1, tp + fp + tn + fn)

    return {"pre": pre, "rec": rec, "f1": f1, "cov": cov,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "tp_correct": tp_correct}


def select_theta_for_target_precision(
    target_precision: float,
    y_true: np.ndarray,
    id_scores: np.ndarray, id_pred: np.ndarray, id_nn: np.ndarray,
    ood_scores: np.ndarray, ood_pred: np.ndarray, ood_nn: np.ndarray,
    use_agreement: bool,
):
    all_scores = np.concatenate([id_scores, ood_scores])
    candidates = np.sort(np.unique(all_scores))
    candidates = np.concatenate([candidates, np.array([all_scores.max() + 1e-6], dtype=np.float32)])

    satisfying = []
    fallback   = []

    for theta in candidates:
        m = compute_pipeline_metrics(
            float(theta), y_true,
            id_scores, id_pred, id_nn,
            ood_scores, ood_pred, ood_nn,
            use_agreement,
        )
        record = (float(theta), m)
        if m["pre"] >= target_precision:
            satisfying.append(record)
        fallback.append(record)

    if satisfying:
        satisfying.sort(key=lambda x: (x[1]["rec"], x[1]["cov"], -x[0]), reverse=True)
        best_theta, best_metrics = satisfying[0]
        return best_theta, best_metrics, True

    fallback.sort(key=lambda x: (x[1]["pre"], x[1]["rec"], x[1]["cov"], -x[0]), reverse=True)
    best_theta, best_metrics = fallback[0]
    return best_theta, best_metrics, False


def apply_threshold_on_eval(
    theta: float,
    y_true: np.ndarray,
    id_scores: np.ndarray, id_pred: np.ndarray, id_nn: np.ndarray,
    ood_scores: np.ndarray, ood_pred: np.ndarray, ood_nn: np.ndarray,
    use_agreement: bool,
):
    if use_agreement:
        is_id_pred_idset  = (id_scores  >= theta) & (id_pred  == id_nn)
        is_id_pred_oodset = (ood_scores >= theta) & (ood_pred == ood_nn)
    else:
        is_id_pred_idset  = (id_scores  >= theta)
        is_id_pred_oodset = (ood_scores >= theta)

    y_pred_idset = np.full_like(y_true, -1)
    y_pred_idset[is_id_pred_idset] = id_pred[is_id_pred_idset]

    return compute_end_to_end_metrics(
        y_true_id=y_true,
        is_id_pred_idset=is_id_pred_idset,
        y_pred_idset=y_pred_idset,
        is_id_pred_oodset=is_id_pred_oodset,
    )


def split_ood_calib_eval(emb: np.ndarray, logits: np.ndarray, calib_frac: float, seed: int):
    n   = emb.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_calib  = max(1, int(round(n * calib_frac)))
    n_calib  = min(n_calib, n - 1) if n > 1 else 1
    calib_idx = idx[:n_calib]
    eval_idx  = idx[n_calib:] if n > 1 else idx[:1]
    return emb[calib_idx], logits[calib_idx], emb[eval_idx], logits[eval_idx]


def eval_one_checkpoint(
    project_root: Path,
    ckpt_path: Path,
    target_precision: float,
    ood_calib_frac: float,
    cfg: TrainConfig,
) -> dict:
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

    ckpt        = torch.load(str(ckpt_path), map_location="cpu")
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

    pm = dict(shuffle=False, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=8, **pm)
    val_loader   = DataLoader(val_ds,   batch_size=8, **pm)
    test_loader  = DataLoader(test_ds,  batch_size=8, **pm)
    ood_loader   = DataLoader(ood_ds,   batch_size=8, **pm)

    train_emb, _, train_y          = extract_embeddings_and_logits_multimodal(model, train_loader, device=str(device))
    val_emb,   val_logits,  val_y  = extract_embeddings_and_logits_multimodal(model, val_loader,   device=str(device))
    test_emb,  test_logits, test_y = extract_embeddings_and_logits_multimodal(model, test_loader,  device=str(device))
    ood_emb,   ood_logits,  _      = extract_embeddings_and_logits_multimodal(model, ood_loader,   device=str(device))

    train_index = build_faiss_l2_index(train_emb.astype(np.float32), train_y.astype(np.int64))

    ood_calib_emb, ood_calib_logits, ood_eval_emb, ood_eval_logits = split_ood_calib_eval(
        ood_emb, ood_logits, calib_frac=ood_calib_frac, seed=cfg.seed,
    )

    val_scores,       val_pred,       val_nn       = compute_scores_preds_nns(train_index, val_emb.astype(np.float32),       val_logits)
    ood_calib_scores, ood_calib_pred, ood_calib_nn = compute_scores_preds_nns(train_index, ood_calib_emb.astype(np.float32), ood_calib_logits)
    test_scores,      test_pred,      test_nn      = compute_scores_preds_nns(train_index, test_emb.astype(np.float32),      test_logits)
    ood_eval_scores,  ood_eval_pred,  ood_eval_nn  = compute_scores_preds_nns(train_index, ood_eval_emb.astype(np.float32),  ood_eval_logits)

    results = {}
    for variant, use_agreement in [("knn", False), ("knn_star", True)]:
        theta, calib_m, met = select_theta_for_target_precision(
            target_precision=target_precision,
            y_true=val_y,
            id_scores=val_scores,       id_pred=val_pred,       id_nn=val_nn,
            ood_scores=ood_calib_scores, ood_pred=ood_calib_pred, ood_nn=ood_calib_nn,
            use_agreement=use_agreement,
        )
        eval_m = apply_threshold_on_eval(
            theta=theta, y_true=test_y,
            id_scores=test_scores,      id_pred=test_pred,      id_nn=test_nn,
            ood_scores=ood_eval_scores, ood_pred=ood_eval_pred, ood_nn=ood_eval_nn,
            use_agreement=use_agreement,
        )
        results[variant] = {
            "theta":       theta,
            "target_met":  met,
            "calib_pre":   calib_m["pre"] * 100.0,
            "pre":         eval_m.pre  * 100.0,
            "rec":         eval_m.rec  * 100.0,
            "f1":          eval_m.f1   * 100.0,
            "cov":         eval_m.cov  * 100.0,
        }

    return results


def _print_table(title: str, results: list):
    header = f"{'#':<3} {'loss':<10} {'PRE↑':>8} {'REC↑':>8} {'F1↑':>8} {'COV↑':>8} {'Met?':>6}"
    sep    = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for i, (loss, m) in enumerate(results, start=1):
        met = "Y" if m["target_met"] else "N"
        print(
            f"{i:<3} {LOSS_DISPLAY.get(loss, loss):<10} "
            f"{m['pre']:>8.2f} {m['rec']:>8.2f} {m['f1']:>8.2f} {m['cov']:>8.2f} {met:>6}"
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
    ap.add_argument("--target_pre",    type=float, default=99.0, help="Target precision in percent, default 99.")
    ap.add_argument("--ood_calib_frac", type=float, default=0.5,  help="Fraction of OOD used for calibration.")
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

    target_precision = args.target_pre / 100.0

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

    all_results = []
    for loss, ckpt_path in ckpts:
        print(f"\n--- Evaluating {loss} ({ckpt_path.name}) ---")
        m = eval_one_checkpoint(
            project_root=Path(args.project_root),
            ckpt_path=ckpt_path,
            target_precision=target_precision,
            ood_calib_frac=args.ood_calib_frac,
            cfg=cfg,
        )
        all_results.append((loss, m))

    knn_results      = [(l, m["knn"])      for l, m in all_results]
    knn_star_results = [(l, m["knn_star"]) for l, m in all_results]

    _print_table(f"KNN     at target PRE ≥ {args.target_pre:.2f}%", knn_results)
    _print_table(f"KNN*    at target PRE ≥ {args.target_pre:.2f}%", knn_star_results)

    print("\nSampling mode:")
    print(f"  train={cfg.train_samples or 'full'}, val={cfg.val_samples or 'full'}, test={cfg.test_samples or 'full'}")

    print("\nNotes:")
    print("- Threshold θ selected on calibration data to achieve target precision if possible.")
    print("- ID calibration set = RVL-CDIP validation split.")
    print("- OOD calibration set = random fraction of RVL-CDIP-O (--ood_calib_frac).")
    print("- PRE/REC/F1/COV measured on ID test + held-out OOD evaluation subset.")
    print("- 'Met?' = whether calibration threshold reached the requested precision.")


if __name__ == "__main__":
    main()
