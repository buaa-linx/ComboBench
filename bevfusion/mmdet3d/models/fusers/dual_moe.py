from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from mmdet3d.models.builder import FUSERS
from mmdet3d.models.moe.branch_moe import moe_entropy_loss, moe_load_balance_loss

__all__ = ["DualMoEFuser", "MoEFuser"]


def _sparsify_topk(probs: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), probs.size(1))
    vals, idx = torch.topk(probs, k, dim=1)
    out = torch.zeros_like(probs)
    out.scatter_(1, idx, vals)
    out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return out


class _ConvBranchMoE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        top_k: int,
        noisy_gating: bool,
        noise_std: float,
        use_experts: bool = True,
        inter_viable: bool = False,
    ) -> None:
        super().__init__()
        self.inter_viable = bool(inter_viable)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.noisy_gating = bool(noisy_gating)
        self.noise_std = float(noise_std)
        self.use_experts = bool(use_experts)

        hidden_channels = max(
            64, min(128, self.in_channels // 2 if self.in_channels > 1 else 64)
        )

        if self.use_experts:
            self.gate_backbone = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.in_channels, hidden_channels),
                nn.ReLU(inplace=True),
            )
            self.routing_head = nn.Linear(hidden_channels, self.num_experts)
            self.quality_head = nn.Linear(hidden_channels, 1)
            self.noise_linear = nn.Linear(hidden_channels, self.num_experts)
            nn.init.zeros_(self.noise_linear.weight)
            nn.init.zeros_(self.noise_linear.bias)

            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            self.in_channels,
                            self.out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            self.out_channels,
                            self.out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(self.out_channels),
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )
            self.quality_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 1),
            )

        self._last_router_probs: Optional[torch.Tensor] = None
        self._last_raw_router_probs: Optional[torch.Tensor] = None
        self._last_expert_outputs: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor):
        if not self.use_experts:
            out = self.proj(x)
            quality = self.quality_head(x)
            self._last_raw_router_probs = None
            self._last_router_probs = None
            self._last_expert_outputs = None
            return out, None, quality

        gate_feat = self.gate_backbone(x)
        logits = self.routing_head(gate_feat)
        if self.noisy_gating and self.training:
            noise_scale = F.softplus(self.noise_linear(gate_feat))
            logits = logits + torch.randn_like(logits) * noise_scale * self.noise_std

        raw_probs = F.softmax(logits, dim=1)
        probs = _sparsify_topk(raw_probs, self.top_k) if self.top_k > 0 else raw_probs
        quality = self.quality_head(gate_feat)

        out = None
        expert_outputs_list = [] if self.inter_viable else None
        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(x)
            if expert_outputs_list is not None:
                expert_outputs_list.append(expert_out)
            out = (
                (expert_out * probs[:, expert_idx].view(-1, 1, 1, 1))
                if out is None
                else out + expert_out * probs[:, expert_idx].view(-1, 1, 1, 1)
            )

        if self.inter_viable:
            self._last_raw_router_probs = raw_probs.detach()
            self._last_router_probs = probs.detach()
            self._last_expert_outputs = torch.stack(expert_outputs_list, dim=1).detach()
        else:
            self._last_raw_router_probs = None
            self._last_router_probs = None
            self._last_expert_outputs = None
        return out, probs, quality


@FUSERS.register_module()
class DualMoEFuser(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_experts: int = 16,
        top_k: int = 2,
        noisy_gating: bool = True,
        noise_std: float = 1.0,
        lambda_lb: float = 0.05,
        lambda_ent: float = 0.001,
        fusion_mode: str = "weighted",
        use_experts: bool = True,
        inter_viable: bool = False,
    ) -> None:
        super().__init__()
        self.inter_viable = bool(inter_viable)
        if len(in_channels) != 2:
            raise ValueError(f"in_channels must have 2 entries, got {in_channels}")
        if fusion_mode not in ("weighted", "concat"):
            raise ValueError(
                f"fusion_mode must be 'weighted' or 'concat', got {fusion_mode!r}"
            )

        self.in_channels = list(map(int, in_channels))
        self.out_channels = int(out_channels)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.noisy_gating = bool(noisy_gating)
        self.noise_std = float(noise_std)
        self.lambda_lb = float(lambda_lb)
        self.lambda_ent = float(lambda_ent)
        self.fusion_mode = str(fusion_mode)
        self.use_experts = bool(use_experts)

        self.camera_moe = _ConvBranchMoE(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            noisy_gating=self.noisy_gating,
            noise_std=self.noise_std,
            use_experts=self.use_experts,
            inter_viable=self.inter_viable,
        )
        self.lidar_moe = _ConvBranchMoE(
            in_channels=self.in_channels[1],
            out_channels=self.out_channels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            noisy_gating=self.noisy_gating,
            noise_std=self.noise_std,
            use_experts=self.use_experts,
            inter_viable=self.inter_viable,
        )

        if self.fusion_mode == "concat":
            self.fusion = nn.Sequential(
                nn.Conv2d(
                    2 * self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.refine = nn.Sequential(
                nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        self._aux_loss: Optional[torch.Tensor] = None
        self._modality_weights: Optional[torch.Tensor] = None
        self._cam_expert_weights: Optional[torch.Tensor] = None
        self._lid_expert_weights: Optional[torch.Tensor] = None
        self._cam_feat_before_moe: Optional[torch.Tensor] = None
        self._lid_feat_before_moe: Optional[torch.Tensor] = None
        self._cam_after_moe: Optional[torch.Tensor] = None
        self._lid_after_moe: Optional[torch.Tensor] = None
        self._cam_weighted: Optional[torch.Tensor] = None
        self._lid_weighted: Optional[torch.Tensor] = None
        self._modality_scores: Optional[torch.Tensor] = None
        self._fused_feature: Optional[torch.Tensor] = None

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if not isinstance(features, (list, tuple)) or len(features) != 2:
            raise ValueError(
                f"DualMoEFuser expects 2 feature maps, got {type(features)} len={len(features)}"
            )

        cam_feat, lid_feat = features
        cam_out, p_cam, q_cam = self.camera_moe(cam_feat)
        lid_out, p_lid, q_lid = self.lidar_moe(lid_feat)

        modality_weights = None
        modality_scores = None
        weighted_cam = None
        weighted_lid = None
        if self.fusion_mode == "concat":
            fused = torch.cat([cam_out, lid_out], dim=1)
            out = self.fusion(fused)
        else:
            modality_scores = torch.cat([q_cam, q_lid], dim=1)
            modality_weights = F.softmax(modality_scores, dim=1)
            weighted_cam = cam_out * modality_weights[:, 0].view(-1, 1, 1, 1)
            weighted_lid = lid_out * modality_weights[:, 1].view(-1, 1, 1, 1)
            fused = weighted_cam + weighted_lid
            out = self.refine(fused)

        if self.inter_viable:
            self._modality_weights = (
                None if modality_weights is None else modality_weights.detach()
            )
            self._modality_scores = (
                None if modality_scores is None else modality_scores.detach()
            )
            self._cam_weighted = None if weighted_cam is None else weighted_cam.detach()
            self._lid_weighted = None if weighted_lid is None else weighted_lid.detach()
            self._cam_expert_weights = None if p_cam is None else p_cam.detach()
            self._lid_expert_weights = None if p_lid is None else p_lid.detach()
            self._cam_feat_before_moe = cam_feat.detach()
            self._lid_feat_before_moe = lid_feat.detach()
            self._cam_after_moe = cam_out.detach()
            self._lid_after_moe = lid_out.detach()
            self._fused_feature = fused.detach()
        else:
            self._modality_weights = None
            self._modality_scores = None
            self._cam_weighted = None
            self._lid_weighted = None
            self._cam_expert_weights = None
            self._lid_expert_weights = None
            self._cam_feat_before_moe = None
            self._lid_feat_before_moe = None
            self._cam_after_moe = None
            self._lid_after_moe = None
            self._fused_feature = None

        aux = None
        if self.use_experts and p_cam is not None and p_lid is not None:
            if self.lambda_lb > 0:
                aux = (
                    moe_load_balance_loss(p_cam) + moe_load_balance_loss(p_lid)
                ) * self.lambda_lb
            if self.lambda_ent > 0:
                ent = (
                    moe_entropy_loss(p_cam) + moe_entropy_loss(p_lid)
                ) * self.lambda_ent
                aux = ent if aux is None else aux + ent
        self._aux_loss = aux
        return out


MoEFuser = DualMoEFuser
