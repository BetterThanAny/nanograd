"""L1 and smooth-L1 (Huber) losses."""
from __future__ import annotations

import numpy as np

from nanograd.function import Function
from nanograd.tensor import Tensor


class L1Loss(Function):
    def forward(self, pred, target):
        diff = pred - target
        self.save_for_backward(diff)
        self.n = diff.size
        return np.asarray(np.abs(diff).mean(), dtype=pred.dtype)

    def backward(self, g):
        (diff,) = self.saved
        grad = g * np.sign(diff) / self.n
        return grad, -grad


class HuberLoss(Function):
    def forward(self, pred, target, *, delta=1.0):
        self.delta = delta
        diff = pred - target
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        self.save_for_backward(diff, abs_diff)
        self.n = diff.size
        return np.asarray(loss.mean(), dtype=pred.dtype)

    def backward(self, g):
        diff, abs_diff = self.saved
        d = self.delta
        # grad: diff where |diff|<d, d*sign(diff) otherwise
        grad = np.where(abs_diff <= d, diff, d * np.sign(diff))
        grad = g * grad / self.n
        return grad, -grad


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return L1Loss.apply(pred, target)


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    return HuberLoss.apply(pred, target, delta=delta)


class FocalLoss(Function):
    """Focal loss for binary classification with logits.

    FL = -alpha * (1-p)^gamma * t * log(p) - (1-alpha) * p^gamma * (1-t) * log(1-p)

    Reduces the loss contribution of easy examples and focuses on hard ones.
    """

    def forward(self, logits, target, *, alpha=0.25, gamma=2.0):
        # stable sigmoid — both branches computed then masked; suppress benign overflows
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.where(
                logits >= 0,
                1.0 / (1.0 + np.exp(-logits)),
                np.exp(logits) / (1.0 + np.exp(logits)),
            ).astype(logits.dtype)
        eps = 1e-7
        p_c = np.clip(p, eps, 1 - eps)
        # focal weight
        pos_w = alpha * (1 - p_c) ** gamma
        neg_w = (1 - alpha) * p_c ** gamma
        loss = -(target * pos_w * np.log(p_c) + (1 - target) * neg_w * np.log(1 - p_c))
        self.save_for_backward(p, target)
        self.alpha = alpha
        self.gamma = gamma
        self.n = logits.size
        return np.asarray(loss.mean(), dtype=logits.dtype)

    def backward(self, g):
        p, t = self.saved
        a = self.alpha
        gm = self.gamma
        # dL/dlogit for focal: derived analytically
        # simpler: approximate as sigmoid(logit) - t scaled by focal weight + additional term
        # Full derivative:
        #  d/dlogit [ -a*(1-p)^g * t * log(p) ] = a*t*(1-p)^(g-1) * (g*log(p) + (1-p)/p * (-p*(1-p)))
        # That's complex. Let me recompute:
        # Let x = logit, p = sigmoid(x), dp/dx = p*(1-p).
        # L = -a*(1-p)^g * t * log(p) - (1-a)*p^g * (1-t) * log(1-p)
        # dL/dp (for t=1 term):
        #   d/dp [-a*(1-p)^g * log(p)] = -a*[-g*(1-p)^(g-1)*log(p) + (1-p)^g / p]
        #                              = a*g*(1-p)^(g-1)*log(p) - a*(1-p)^g / p
        # then dL/dx = dL/dp * dp/dx = dL/dp * p*(1-p)
        # too messy; use chain rule via numerical computation at forward for dL/dx
        eps = 1e-7
        p_c = np.clip(p, eps, 1 - eps)
        g_pow = (1 - p_c) ** gm
        p_pow = p_c ** gm

        # compute dL/dp explicitly
        dL_dp_pos = -a * (-gm * (1 - p_c) ** (gm - 1) * np.log(p_c) + g_pow / p_c)
        dL_dp_neg = -(1 - a) * (gm * p_c ** (gm - 1) * np.log(1 - p_c) - p_pow / (1 - p_c))
        dL_dp = t * dL_dp_pos + (1 - t) * dL_dp_neg
        dp_dx = p * (1 - p)
        grad_logits = g * dL_dp * dp_dx / self.n
        return grad_logits, None  # no grad wrt target


def focal_loss(logits: Tensor, target: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    return FocalLoss.apply(logits, target, alpha=alpha, gamma=gamma)


class TripletLoss(Function):
    """Triplet margin loss: max(0, d(anchor, positive) - d(anchor, negative) + margin).

    Uses squared L2 distance. Inputs are embeddings of shape (B, D).
    """

    def forward(self, anchor, positive, negative, *, margin=1.0):
        d_ap = np.sum((anchor - positive) ** 2, axis=-1)
        d_an = np.sum((anchor - negative) ** 2, axis=-1)
        loss_per = np.maximum(d_ap - d_an + margin, 0.0)
        active = (loss_per > 0).astype(anchor.dtype)
        self.save_for_backward(anchor, positive, negative, active)
        self.n = anchor.shape[0]
        return np.asarray(loss_per.mean(), dtype=anchor.dtype)

    def backward(self, g):
        anchor, positive, negative, active = self.saved
        active_b = active[:, None]  # (B, 1)
        # dL/danchor = 2*(anchor - positive) - 2*(anchor - negative) = 2*(negative - positive)
        d_anchor = g * 2.0 * (negative - positive) * active_b / self.n
        d_positive = g * -2.0 * (anchor - positive) * active_b / self.n
        d_negative = g * 2.0 * (anchor - negative) * active_b / self.n
        return d_anchor, d_positive, d_negative


def triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0) -> Tensor:
    return TripletLoss.apply(anchor, positive, negative, margin=margin)
