from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class FaissKNNIndex:
    index: faiss.Index
    train_labels: np.ndarray  # [N]

def build_faiss_l2_index(train_embeddings: np.ndarray, train_labels: np.ndarray) -> FaissKNNIndex:
    """
    train_embeddings: [N, D] float32
    """
    assert train_embeddings.dtype == np.float32
    N, D = train_embeddings.shape
    index = faiss.IndexFlatL2(D)  # exact L2
    index.add(train_embeddings)
    return FaissKNNIndex(index=index, train_labels=train_labels.astype(np.int64))

def knn1_score_and_neighbor(index: FaissKNNIndex, emb: np.ndarray) -> Tuple[float, int, int]:
    """
    emb: [D] float32
    Returns:
      score = -euclidean_distance_to_1NN
      nn_idx, nn_label
    """
    emb2 = emb.reshape(1, -1)
    dist2, idx = index.index.search(emb2, k=1)  # dist2 shape [1,1]
    nn_idx = int(idx[0, 0])
    nn_label = int(index.train_labels[nn_idx])
    euclid = float(np.sqrt(dist2[0, 0]))
    score = -euclid
    return score, nn_idx, nn_label

def extract_embeddings_and_logits(model, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      embeddings: [N, D] float32
      logits: [N, C] float32
      labels: [N] int64
    """
    model.eval()
    embs = []
    logs = []
    ys = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels = batch["labels"].cpu().numpy().astype(np.int64)

            logits, h = model(input_ids, attention_mask, token_type_ids)
            embs.append(h.cpu().numpy().astype(np.float32))
            logs.append(logits.cpu().numpy().astype(np.float32))
            ys.append(labels)

    embeddings = np.concatenate(embs, axis=0)
    logits = np.concatenate(logs, axis=0)
    labels = np.concatenate(ys, axis=0)
    return embeddings, logits, labels

def estimate_threshold_theta(
    train_index: FaissKNNIndex,
    val_embeddings: np.ndarray,
    tpr_target: float
) -> float:
    """
    Algorithm 2 Function 1 (threshold_estimation):
      - compute score for each val doc using 1NN distance
      - sort scores descending
      - ind = ceil(tpr * n)
      - theta = scores[ind-1] (careful with 0-based)
    """
    scores = []
    for i in range(val_embeddings.shape[0]):
        s, _, _ = knn1_score_and_neighbor(train_index, val_embeddings[i])
        scores.append(s)

    scores = np.array(scores, dtype=np.float32)
    scores_sorted = np.sort(scores)[::-1]  # descending
    n = len(scores_sorted)
    ind = int(np.ceil(tpr_target * n))  # 1..n
    ind0 = max(0, min(n - 1, ind - 1))
    theta = float(scores_sorted[ind0])
    return theta

def knn_star_predict(
    train_index: FaissKNNIndex,
    embeddings: np.ndarray,
    logits: np.ndarray,
    theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements Section 3.4 consensus agreement (KNN*):
      - predicted label = argmax(logits)
      - find 1NN label
      - if pred != nn_label => reject as OOD immediately
      - else use score>=theta acceptance
    Returns:
      is_id_pred: [N] bool (True means accepted as ID)
      pred_label_or_minus1: [N] int (pred label if accepted else -1)
    """
    pred = logits.argmax(axis=1).astype(np.int64)

    is_id = np.zeros((embeddings.shape[0],), dtype=bool)
    out_label = np.full((embeddings.shape[0],), -1, dtype=np.int64)

    for i in range(embeddings.shape[0]):
        score, _, nn_label = knn1_score_and_neighbor(train_index, embeddings[i])

        # consensus agreement
        if pred[i] != nn_label:
            is_id[i] = False
            out_label[i] = -1
            continue

        # novelty threshold
        if score >= theta:
            is_id[i] = True
            out_label[i] = pred[i]
        else:
            is_id[i] = False
            out_label[i] = -1

    return is_id, out_label

def knn_predict_no_agreement(
    train_index: FaissKNNIndex,
    embeddings: np.ndarray,
    logits: np.ndarray,
    theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KNN without agreement (paper: KNN), still uses argmax logits for label when accepted.
    """
    pred = logits.argmax(axis=1).astype(np.int64)

    is_id = np.zeros((embeddings.shape[0],), dtype=bool)
    out_label = np.full((embeddings.shape[0],), -1, dtype=np.int64)

    for i in range(embeddings.shape[0]):
        score, _, _ = knn1_score_and_neighbor(train_index, embeddings[i])
        if score >= theta:
            is_id[i] = True
            out_label[i] = pred[i]
        else:
            is_id[i] = False
            out_label[i] = -1

    return is_id, out_label