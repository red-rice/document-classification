from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score

@dataclass
class OODMetrics:
    auc: float
    fpr_at_tpr: float

@dataclass
class EndToEndMetrics:
    pre: float
    rec: float
    f1: float
    cov: float
    tp: int
    fp: int
    tn: int
    fn: int
    tp_correct: int

def compute_auc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Positive = ID, Negative = OOD as described in paper section 4.2.
    Higher score => more ID-like (here score is -distance, closer to 0 is higher).
    """
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))

def compute_fpr_at_tpr95(id_scores: np.ndarray, ood_scores: np.ndarray, tpr_target: float = 0.95) -> float:
    """
    Using the paper’s thresholding approach:
    - choose theta so that at least tpr_target of ID are accepted
    - then compute FPR on OOD as fraction accepted as ID.
    """
    scores_sorted = np.sort(id_scores)[::-1]
    n = len(scores_sorted)
    ind = int(np.ceil(tpr_target * n))
    ind0 = max(0, min(n - 1, ind - 1))
    theta = scores_sorted[ind0]

    ood_accepted = (ood_scores >= theta).sum()
    fpr = float(ood_accepted / max(1, len(ood_scores)))
    return fpr

def compute_end_to_end_metrics(
    # ground truth
    y_true_id: np.ndarray,        # labels for ID test samples
    # predictions
    is_id_pred_idset: np.ndarray, # bool: accepted as ID for ID test samples
    y_pred_idset: np.ndarray,     # predicted label or -1 for ID test samples
    is_id_pred_oodset: np.ndarray # bool: accepted as ID for OOD samples
) -> EndToEndMetrics:
    """
    Following paper definition:
      Positive = predicted ID
      Negative = predicted OOD
      - For ID set:
          TP = predicted ID
          FN = predicted OOD
          TP_correct = predicted ID AND label correct
      - For OOD set:
          FP = predicted ID (mistakenly accepted)
          TN = predicted OOD (correctly rejected)
    """
    # ID test
    tp_mask = is_id_pred_idset
    fn_mask = ~is_id_pred_idset
    tp = int(tp_mask.sum())
    fn = int(fn_mask.sum())

    tp_correct = int(((y_pred_idset == y_true_id) & tp_mask).sum())

    # OOD set
    fp = int(is_id_pred_oodset.sum())
    tn = int((~is_id_pred_oodset).sum())

    # Equations (6)-(9)
    rec = tp_correct / max(1, (tp + fn))
    pre = tp_correct / max(1, (tp + fp))
    f1 = 0.0 if (pre + rec) == 0 else 2.0 * pre * rec / (pre + rec)
    cov = tp_correct / max(1, (tp + fp + tn + fn))

    return EndToEndMetrics(
        pre=float(pre),
        rec=float(rec),
        f1=float(f1),
        cov=float(cov),
        tp=tp, fp=fp, tn=tn, fn=fn, tp_correct=tp_correct
    )