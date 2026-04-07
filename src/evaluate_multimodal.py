import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Paths, TrainConfig, OODConfig
from data import set_seed, resolve_split_limit
from data_multimodal import RVLCDIPLayoutLMv3Dataset
from data_multimodal_ood import RVLCDIPOODLayoutLMv3Dataset
from model_multimodal import LayoutLMv3DocClassifier
from knn_ood_multimodal import (
    build_faiss_l2_index,
    extract_embeddings_and_logits_multimodal,
    estimate_threshold_theta,
    knn_predict_no_agreement,
    knn_star_predict,
    knn1_score_and_neighbor,
)
from metrics import compute_auc, compute_fpr_at_tpr95, compute_end_to_end_metrics


def compute_scores_from_index(train_index, embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros((embeddings.shape[0],), dtype=np.float32)
    for i in range(embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        scores[i] = s
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--use_knn_star",
        action="store_true",
        help="Use KNN* (agreement). Default is KNN (no agreement).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override eval batch size. Defaults to cfg.batch_size.",
    )
    ap.add_argument(
        "--debug_samples",
        type=int,
        default=None,
        help="If set, use this sample count for all splits. Pass -1 to disable debug mode.",
    )
    ap.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Used only when debug_samples is disabled.",
    )
    ap.add_argument(
        "--val_samples",
        type=int,
        default=None,
        help="Used only when debug_samples is disabled.",
    )
    ap.add_argument(
        "--test_samples",
        type=int,
        default=None,
        help="Used only when debug_samples is disabled.",
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
    if args.debug_samples is not None:
        cfg.debug_samples = None if args.debug_samples < 0 else args.debug_samples
    if args.train_samples is not None:
        cfg.train_samples = args.train_samples
    if args.val_samples is not None:
        cfg.val_samples = args.val_samples
    if args.test_samples is not None:
        cfg.test_samples = args.test_samples

    ood_cfg = OODConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]
    num_classes = int(ckpt.get("num_classes", 16))

    model = LayoutLMv3DocClassifier(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ocr_box_root = project_root / "rvl-cdip-box"
    ood_box_root = project_root / "rvl-cdip-o-box"
    rvl_ood_root = project_root / "rvl-cdip-o"
    ood_text_root = project_root / "rvl-cdip-o-text"

    train_debug_samples = resolve_split_limit(cfg, "train")
    val_debug_samples = resolve_split_limit(cfg, "val")
    test_debug_samples = resolve_split_limit(cfg, "test")

    train_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.train_list,
        processor_name=model_name,
        max_length=max_length,
        debug_samples=train_debug_samples,
    )
    val_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.val_list,
        processor_name=model_name,
        max_length=max_length,
        debug_samples=val_debug_samples,
    )
    test_ds = RVLCDIPLayoutLMv3Dataset(
        rvl_root=paths.rvl_cdip_dir,
        ocr_box_root=ocr_box_root,
        split_file=paths.test_list,
        processor_name=model_name,
        max_length=max_length,
        debug_samples=test_debug_samples,
    )
    ood_ds = RVLCDIPOODLayoutLMv3Dataset(
        rvl_ood_root=rvl_ood_root,
        ocr_box_root=ood_box_root,
        split_dir=ood_text_root,
        processor_name=model_name,
        max_length=max_length,
        debug_samples=test_debug_samples,
    )

    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    ood_loader = DataLoader(
        ood_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_emb, _, train_y = extract_embeddings_and_logits_multimodal(
        model, train_loader, device=str(device)
    )
    train_index = build_faiss_l2_index(
        train_emb.astype(np.float32),
        train_y.astype(np.int64),
    )

    val_emb, _, _ = extract_embeddings_and_logits_multimodal(
        model, val_loader, device=str(device)
    )
    test_emb, test_logits, test_y = extract_embeddings_and_logits_multimodal(
        model, test_loader, device=str(device)
    )
    ood_emb, ood_logits, _ = extract_embeddings_and_logits_multimodal(
        model, ood_loader, device=str(device)
    )

    theta = estimate_threshold_theta(
        train_index,
        val_emb.astype(np.float32),
        tpr_target=ood_cfg.tpr_target,
    )
    print(f"Theta (TPR>={ood_cfg.tpr_target} on VAL): {theta:.6f}")

    id_scores = compute_scores_from_index(train_index, test_emb.astype(np.float32))
    ood_scores = compute_scores_from_index(train_index, ood_emb.astype(np.float32))

    auc = compute_auc(id_scores, ood_scores)
    fpr95 = compute_fpr_at_tpr95(
        id_scores,
        ood_scores,
        tpr_target=ood_cfg.tpr_target,
    )
    print(f"AUC: {auc:.4f}")
    print(f"FPR@TPR{int(ood_cfg.tpr_target * 100)}: {fpr95:.4f}")

    if args.use_knn_star:
        is_id_test, pred_test = knn_star_predict(train_index, test_emb, test_logits, theta)
        is_id_ood, pred_ood = knn_star_predict(train_index, ood_emb, ood_logits, theta)
        print("Using KNN* (agreement) for ambiguity rejection.")
    else:
        is_id_test, pred_test = knn_predict_no_agreement(train_index, test_emb, test_logits, theta)
        is_id_ood, pred_ood = knn_predict_no_agreement(train_index, ood_emb, ood_logits, theta)
        print("Using KNN (no agreement).")

    e2e = compute_end_to_end_metrics(
        y_true_id=test_y,
        is_id_pred_idset=is_id_test,
        y_pred_idset=pred_test,
        is_id_pred_oodset=is_id_ood,
    )

    print("\nEnd-to-end metrics:")
    print(f"  PRE: {e2e.pre:.4f}")
    print(f"  REC: {e2e.rec:.4f}")
    print(f"  F1 : {e2e.f1:.4f}")
    print(f"  COV: {e2e.cov:.4f}")
    print(
        f"  Counts: TP={e2e.tp}, FN={e2e.fn}, FP={e2e.fp}, "
        f"TN={e2e.tn}, TP_correct={e2e.tp_correct}"
    )

    print("\nSampling:")
    print(f"  train_debug_samples = {train_debug_samples}")
    print(f"  val_debug_samples   = {val_debug_samples}")
    print(f"  test_debug_samples  = {test_debug_samples}")


if __name__ == "__main__":
    main()