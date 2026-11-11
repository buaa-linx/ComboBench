"""Auxiliary losses used by the Dual-MoE fusion module."""

import torch

__all__ = ["moe_load_balance_loss", "moe_entropy_loss"]


def moe_load_balance_loss(p: torch.Tensor) -> torch.Tensor:
    """L_lb = sum_k (u_k - 1/N)^2, u_k = mean_b p_{b,k}."""
    n = p.size(1)
    u = p.mean(dim=0)
    target = p.new_tensor(1.0 / n)
    return ((u - target) ** 2).sum()


def moe_entropy_loss(p: torch.Tensor) -> torch.Tensor:
    """Mean batch entropy H(p) = -sum_k p log p."""
    plog = (p.clamp_min(1e-8)).log()
    h = -(p * plog).sum(dim=1)
    return h.mean()
