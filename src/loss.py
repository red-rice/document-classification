import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMarginContrastiveLoss(nn.Module):
    """
    Implements Algorithm 1:
      - z = L2-normalized(h)
      - xi = alpha * max_{pos pairs} ||z_i - z_p||^2
      - L_pos = mean_{i, p in P(i)} ||z_i - z_p||^2
      - L_neg = mean_{i, n in N(i)} relu(xi - ||z_i - z_n||^2)
      - L_margin = (beta * L_pos + L_neg) / d
      - L = CE + lam * L_margin
    """
    def __init__(self, alpha: float, beta: float, lam: float, eps: float = 1e-12):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def pairwise_squared_l2(z: torch.Tensor) -> torch.Tensor:
        # z: [B, D], returns [B, B] squared L2 distances
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        z2 = (z * z).sum(dim=1, keepdim=True)             # [B,1]
        dist2 = z2 + z2.t() - 2.0 * (z @ z.t())
        return torch.clamp(dist2, min=0.0)

    def forward(self, logits: torch.Tensor, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # CE on logits
        ce_loss = self.ce(logits, labels)

        # Normalize embeddings
        z = F.normalize(h, p=2, dim=1, eps=self.eps)  # [B, D]
        B, D = z.shape

        dist2 = self.pairwise_squared_l2(z)  # [B,B]

        # Masks
        labels_col = labels.view(-1, 1)  # [B,1]
        same = (labels_col == labels_col.t())
        diff = ~same
        eye = torch.eye(B, dtype=torch.bool, device=labels.device)
        pos_mask = same & ~eye
        neg_mask = diff

        # Need at least one positive pair in batch, else margin undefined.
        # The sampler is designed to avoid this, matching the paper's custom minibatch logic.
        if pos_mask.sum() == 0:
            # Explicitly fail: this should not happen if sampler is correct.
            raise RuntimeError("No positive pairs in batch. Check sampler/min_per_class.")

        # xi: max positive squared distance then scaled by alpha
        xi = dist2[pos_mask].max() * self.alpha

        # L_pos: mean of positive squared distances
        L_pos = dist2[pos_mask].mean()

        # L_neg: mean hinge over negative pairs
        if neg_mask.sum() == 0:
            # extremely rare (all same class) — treat as zero
            L_neg = torch.zeros((), device=labels.device, dtype=dist2.dtype)
        else:
            L_neg = F.relu(xi - dist2[neg_mask]).mean()

        L_margin = (self.beta * L_pos + L_neg)
        total = ce_loss + self.lam * L_margin
        return total