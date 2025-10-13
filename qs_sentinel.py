"""
Quantum-Sentinel/QAT training and evaluation utilities (clean rewrite, PyTorch 2.0+).

Exports
-------
- QuantumSentinelTrainer, QuantumAdversarialTrainer (QAT baseline)
- evaluate_model (clean + QC-FGSM + QC-PGD + QC-Poison metrics)
- configs: QSentinelConfig, QATConfig, PoisonEvalConfig
"""

from __future__ import annotations
import math, os, random, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:  # optional AMP
    from torch import amp as _amp_mod
except ImportError:
    from torch.cuda import amp as _amp_mod  # type: ignore
amp = _amp_mod
try:
    _GradScalerCtor = amp.GradScaler  # type: ignore[attr-defined]
except AttributeError:
    from torch.cuda.amp import GradScaler as _GradScalerCtor  # type: ignore

from defense_utils import QDBRLoss, AdvConfig, generate_adversarial


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class PoisonEvalConfig:
    enabled: bool = True
    steps: int = 32
    grad_mode: str = "normalized"   # "sign" | "normalized" | "raw"
    attack_mode: str = "hybrid"     # "targeted" | "untargeted" | "margin" | "hybrid"
    target_mode: str = "flip"
    clamp_params: bool = True
    restarts: int = 4
    random_start: bool = True
    momentum: float = 0.8
    hybrid_lambda: float = 0.65
    eval_batches: int = 4
    step_scale: float = 2.0
    max_batches_per_step: Optional[int] = None
    use_tqdm: bool = False


@dataclass
class QSentinelConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    seed: Optional[int] = 42
    grad_clip_norm: Optional[float] = 1.0
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.95
    log_every: int = 1

    qst_recon_w: float = 0.1
    qst_proto_w: float = 0.2
    qst_drift_w: float = 0.1
    qdbr_lambda: float = 0.1
    qdbr_center_momentum: float = 0.9

    quantum_noise_sigma: float = 0.01
    adv: AdvConfig = field(
        default_factory=lambda: AdvConfig(enabled=True, method="pgd", epsilon=0.02, steps=3, mix_ratio=0.5)
    )

    epsilons: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.10)
    poison_eval: PoisonEvalConfig = field(default_factory=PoisonEvalConfig)


@dataclass
class QATConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    seed: Optional[int] = 42
    grad_clip_norm: Optional[float] = 1.0
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.95
    log_every: int = 1
    adv: AdvConfig = field(
        default_factory=lambda: AdvConfig(enabled=True, method="pgd", epsilon=0.02, steps=3, mix_ratio=0.5)
    )


@dataclass
class SentinelSnapshot:
    anchor_params: Dict[str, torch.Tensor]
    mean_vector: Optional[torch.Tensor]
    drift_mean: float
    drift_std: float
    prototypes: Optional[torch.Tensor] = None
    proto_counts: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "SentinelSnapshot":
        return SentinelSnapshot(
            {k: v.to(device) for k, v in self.anchor_params.items()},
            self.mean_vector.to(device) if self.mean_vector is not None else None,
            self.drift_mean,
            self.drift_std,
            self.prototypes.to(device) if self.prototypes is not None else None,
            self.proto_counts.to(device) if self.proto_counts is not None else None,
        )


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _create_grad_scaler(device_type: str, enabled: bool):
    if not enabled or device_type != "cuda":
        class _NoopScaler:
            def scale(self, loss): return loss
            def unscale_(self, optimizer): return optimizer
            def step(self, optimizer): optimizer.step()
            def update(self): pass
        return _NoopScaler()
    try:
        return _GradScalerCtor(enabled=True)
    except TypeError:
        return _GradScalerCtor()


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _collect_quantum_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    quantum_params: List[Tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if any(tag in lowered for tag in ("quantum", "phase_params", "fusion_weights", "params")):
            quantum_params.append((name, param))
    return quantum_params


# --------------------------------------------------------------------------- #
# Quantum State Transformation module
# --------------------------------------------------------------------------- #
class QuantumStateTransformer(nn.Module):
    def __init__(self, num_classes: int, n_qubits: int, tau_init: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
        )
        self.tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        self.drift_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("prototypes", torch.zeros(num_classes, n_qubits))
        self.register_buffer("proto_counts", torch.zeros(num_classes))

    def forward(self, x: torch.Tensor, labels: torch.Tensor, adapter: "HybridAdapter"):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        with torch.no_grad():
            q1_orig, q2_orig = adapter.quantum_paths(x)
            combined_orig = adapter.combine_qfeat(q1_orig, q2_orig)
            fused_orig = adapter.fuse_quantum(combined_orig)

            q1_hat, q2_hat = adapter.quantum_paths(x_hat)
            combined_hat = adapter.combine_qfeat(q1_hat, q2_hat)
            fused_hat = adapter.fuse_quantum(combined_hat)

        fused_orig_norm = F.normalize(fused_orig, dim=1, eps=1e-6)
        fused_hat_norm = F.normalize(fused_hat, dim=1, eps=1e-6)

        proto = self.prototypes[labels]
        counts = self.proto_counts[labels].unsqueeze(1)
        proto_norm = F.normalize(proto, dim=1, eps=1e-6)
        use_proto = counts > 0
        ref_vec = torch.where(use_proto, proto_norm, fused_orig_norm)

        drift = 1.0 - (fused_hat_norm * ref_vec).sum(dim=1)
        tau_local = torch.sigmoid(self.tau + self.drift_scale * drift).view(-1, 1, 1, 1)
        x_repair = torch.clamp(tau_local * x_hat + (1 - tau_local) * x, 0.0, 1.0)

        info = {
            "x_hat": x_hat,
            "drift": drift.detach(),
            "tau": tau_local.detach().view(-1),
            "fq_orig": fused_orig.detach(),
            "fq_hat": fused_hat.detach(),
        }
        return x_repair, info

    @torch.no_grad()
    def update_prototypes(self, fq: torch.Tensor, labels: torch.Tensor, momentum: float) -> None:
        fq_norm = F.normalize(fq, dim=1, eps=1e-6)
        for c in torch.unique(labels):
            mask = labels == c
            if mask.any():
                mean_vec = fq_norm[mask].mean(dim=0)
                self.prototypes[c] = momentum * self.prototypes[c] + (1 - momentum) * mean_vec
                self.proto_counts[c] += mask.sum()


# --------------------------------------------------------------------------- #
# Hybrid adapter (staged passes without editing model source)
# --------------------------------------------------------------------------- #
class HybridAdapter:
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def classical_features(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        feat = self.model.resnet_backbone(x).view(bsz, -1)
        return self.model.feature_reducer(feat)

    def quantum_paths(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.quantum_conv1(x), self.model.quantum_conv2(x)

    def combine_qfeat(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.model.fusion_weights[0])
        beta = torch.sigmoid(self.model.fusion_weights[1])
        return (alpha * q1 + beta * q2) / (alpha + beta)

    def fuse_quantum(self, combined: torch.Tensor) -> torch.Tensor:
        return self.model.quantum_fusion(combined)

    def forward_with_combined(self, x: torch.Tensor, combined_override: torch.Tensor):
        classical = self.classical_features(x)
        fused = self.fuse_quantum(combined_override)
        logits, extras = self.attend_and_classify(fused, classical)
        extras.update({"classical_features": classical, "quantum_features": fused, "logits": logits})
        return extras

    def attend_and_classify(self, fused_qfeat: torch.Tensor, classical_feat: torch.Tensor):
        q_input = fused_qfeat.unsqueeze(1)
        c_input = classical_feat[:, : self.model.config.n_qubits].unsqueeze(1)
        attended_q, attn_weights = self.model.attention(q_input, q_input, q_input)
        final_feat = attended_q.squeeze(1) + 0.1 * c_input.squeeze(1)
        logits = self.model.classifier_head(final_feat)
        return logits, {"attention_weights": attn_weights.squeeze(1), "combined_features": final_feat}


# --------------------------------------------------------------------------- #
# Trainers: Quantum-Sentinel + QAT
# --------------------------------------------------------------------------- #
class QuantumSentinelTrainer:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        n_qubits: int,
        cfg: QSentinelConfig,
        device: torch.device,
        enable_qst: bool = True,
        enable_qdbr: bool = True,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.enable_qst = enable_qst
        self.enable_qdbr = enable_qdbr and cfg.qdbr_lambda > 0
        self.adapter = HybridAdapter(self.model)
        self.qst_module = QuantumStateTransformer(num_classes, n_qubits).to(device) if enable_qst else None
        self.history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec", "lr")}
        
    def _noise_context(self):
        sigma = self.cfg.quantum_noise_sigma
        if sigma <= 0:
            return torch.no_grad()

        class _NoiseCtx:
            def __init__(self, model, sigma):
                self.model = model
                self.sigma = sigma
                self.cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

            def __enter__(self):
                for module in self.model.modules():
                    for attr in ("params", "phase_params"):
                        if hasattr(module, attr):
                            param = getattr(module, attr)
                            self.cache.append((param, param.detach().clone()))
                            param.data.add_(torch.randn_like(param) * self.sigma)
                return self

            def __exit__(self, exc_type, exc, tb):
                for param, tensor in self.cache:
                    param.data.copy_(tensor)
                return False

        return _NoiseCtx(self.model, sigma)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[Dict[str, List[float]], SentinelSnapshot]:
        seed_everything(self.cfg.seed)
        device_type = self.device.type
        scaler = _create_grad_scaler(device_type, self.cfg.use_amp and device_type == "cuda")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, self.cfg.scheduler_step_size),
            gamma=self.cfg.scheduler_gamma,
        )
        ce_loss = nn.CrossEntropyLoss().to(self.device)
        qdbr_loss = None
        if self.enable_qdbr:
            assert self.qst_module is not None
            qdbr_loss = QDBRLoss(
                num_classes=self.qst_module.prototypes.size(0),
                feat_dim=self.qst_module.prototypes.size(1),
                center_momentum=self.cfg.qdbr_center_momentum,
                lambda_qdbr=self.cfg.qdbr_lambda,
            ).to(self.device)

        self.model.train()
        if self.qst_module is not None:
            self.qst_module.train()

        for epoch in range(self.cfg.epochs):
            start = time.perf_counter()
            running_loss = 0.0
            running_acc = 0.0
            batches = 0

            progress = tqdm(train_loader, desc=f"Q-Sentinel epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            for xb, yb in progress:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                x_used = xb
                qst_info = {"x_hat": xb, "drift": torch.zeros(xb.size(0), device=self.device)}
                if self.qst_module is not None:
                    x_used, qst_info = self.qst_module(xb, yb, self.adapter)
                    with torch.no_grad():
                        q_feat = self.adapter.infer(x_used)["quantum_features"]
                        self.qst_module.update_prototypes(q_feat, yb, self.cfg.qdbr_center_momentum)

                amp_ctx = amp.autocast(device_type=device_type, enabled=self.cfg.use_amp and device_type == "cuda")
                with amp_ctx, self._noise_context():
                    outputs = self.model(x_used)
                    logits_clean = outputs["logits"]
                    loss_clean = ce_loss(logits_clean, yb)

                    loss_adv = torch.tensor(0.0, device=self.device)
                    if self.cfg.adv.enabled and self.cfg.adv.mix_ratio > 0:
                        adv_batch = generate_adversarial(self.model, x_used, yb, ce_loss, self.cfg.adv)
                        logits_adv = self.model(adv_batch)["logits"]
                        loss_adv = ce_loss(logits_adv, yb)

                    loss_qst = torch.tensor(0.0, device=self.device)
                    if self.qst_module is not None:
                        recon = F.l1_loss(qst_info["x_hat"], xb)
                        proto_term = (
                            1
                            - F.normalize(outputs["quantum_features"], dim=1, eps=1e-6)
                            * F.normalize(self.qst_module.prototypes[yb], dim=1, eps=1e-6)
                        ).sum(dim=1).mean()
                        drift_term = qst_info["drift"].mean()
                        loss_qst = (
                            self.cfg.qst_recon_w * recon
                            + self.cfg.qst_proto_w * proto_term
                            + self.cfg.qst_drift_w * drift_term
                        )

                    loss_qdbr = torch.tensor(0.0, device=self.device)
                    if qdbr_loss is not None:
                        loss_qdbr = qdbr_loss(outputs["quantum_features"], yb)

                    mix = self.cfg.adv.mix_ratio if self.cfg.adv.enabled else 0.0
                    total_loss = (1 - mix) * loss_clean + mix * loss_adv + loss_qst + loss_qdbr

                scaler.scale(total_loss).backward()
                if self.cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    preds = logits_clean.argmax(dim=1)
                    running_acc += (preds == yb).float().mean().item()
                running_loss += float(total_loss.detach().item())
                batches += 1
                progress.set_postfix({"loss": f"{running_loss/batches:.4f}", "acc": f"{running_acc/batches:.3f}"})

            epoch_time = time.perf_counter() - start
            scheduler.step()
            val_loss, val_acc = self._validate(val_loader, ce_loss)
            self.history["train_loss"].append(running_loss / max(1, batches))
            self.history["train_acc"].append(running_acc / max(1, batches))
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_time_sec"].append(epoch_time)
            self.history["lr"].append(optimizer.param_groups[0]["lr"])

        snapshot = self._build_snapshot(val_loader)
        return self.history, snapshot

    def _validate(self, loader: DataLoader, loss_fn: nn.Module):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                logits = self.model(xb)["logits"]
                loss = loss_fn(logits, yb)
                total_loss += float(loss) * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        self.model.train()
        return total_loss / max(1, total), correct / max(1, total)

    def _build_snapshot(self, val_loader: DataLoader) -> SentinelSnapshot:
        self.model.eval()
        with torch.no_grad():
            feats: List[torch.Tensor] = []
            for xb, _ in val_loader:
                xb = xb.to(self.device, non_blocking=True)
                out = self.model(xb)
                q_feat = out.get("quantum_features")
                if q_feat is not None:
                    feats.append(F.normalize(q_feat, dim=1, eps=1e-6))
            if feats:
                feat_cat = torch.cat(feats, dim=0)
                mean_vec = F.normalize(feat_cat.mean(dim=0), dim=0, eps=1e-6)
                drift = 1 - feat_cat @ mean_vec
                drift_mean = float(drift.mean().item())
                drift_std = float(drift.std(unbiased=False).item()) if drift.numel() > 1 else 0.0
            else:
                mean_vec = None
                drift_mean = float("nan")
                drift_std = float("nan")

        anchor = {name: param.detach().cpu() for name, param in _collect_quantum_parameters(self.model)}
        prototypes = self.qst_module.prototypes.detach().cpu() if self.qst_module is not None else None
        proto_counts = self.qst_module.proto_counts.detach().cpu() if self.qst_module is not None else None
        return SentinelSnapshot(
            anchor,
            mean_vec.cpu() if mean_vec is not None else None,
            drift_mean,
            drift_std,
            prototypes,
            proto_counts,
        )


class QuantumAdversarialTrainer:
    def __init__(self, model: nn.Module, cfg: QATConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec", "lr")}

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        seed_everything(self.cfg.seed)
        device_type = self.device.type
        scaler = _create_grad_scaler(device_type, self.cfg.use_amp and device_type == "cuda")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, self.cfg.scheduler_step_size),
            gamma=self.cfg.scheduler_gamma,
        )
        ce_loss = nn.CrossEntropyLoss().to(self.device)
        self.model.train()

        for epoch in range(self.cfg.epochs):
            start = time.perf_counter()
            running_loss = 0.0
            running_acc = 0.0
            batches = 0
            progress = tqdm(train_loader, desc=f"QAT epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            for xb, yb in progress:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                amp_ctx = amp.autocast(device_type=device_type, enabled=self.cfg.use_amp and device_type == "cuda")
                with amp_ctx:
                    logits = self.model(xb)["logits"]
                    loss_clean = ce_loss(logits, yb)

                    loss_adv = torch.tensor(0.0, device=self.device)
                    if self.cfg.adv.enabled and self.cfg.adv.mix_ratio > 0:
                        adv_batch = generate_adversarial(self.model, xb, yb, ce_loss, self.cfg.adv)
                        logits_adv = self.model(adv_batch)["logits"]
                        loss_adv = ce_loss(logits_adv, yb)

                    total_loss = (1 - self.cfg.adv.mix_ratio) * loss_clean + self.cfg.adv.mix_ratio * loss_adv

                scaler.scale(total_loss).backward()
                if self.cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    running_acc += (logits.argmax(dim=1) == yb).float().mean().item()
                running_loss += float(total_loss.detach().item())
                batches += 1
                progress.set_postfix({"loss": f"{running_loss/batches:.4f}", "acc": f"{running_acc/batches:.3f}"})

            epoch_time = time.perf_counter() - start
            scheduler.step()
            val_loss, val_acc = self._validate(val_loader, ce_loss)
            self.history["train_loss"].append(running_loss / max(1, batches))
            self.history["train_acc"].append(running_acc / max(1, batches))
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_time_sec"].append(epoch_time)
            self.history["lr"].append(optimizer.param_groups[0]["lr"])

        return self.history

    def _validate(self, loader: DataLoader, loss_fn: nn.Module):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                logits = self.model(xb)["logits"]
                loss = loss_fn(logits, yb)
                total_loss += float(loss) * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        self.model.train()
        return total_loss / max(1, total), correct / max(1, total)


# --------------------------------------------------------------------------- #
# Evaluation (clean + QC-FGSM/PGD + QC-Poison)
# --------------------------------------------------------------------------- #
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    epsilons: Sequence[float],
    device: torch.device,
    *,
    poison_cfg: Optional[PoisonEvalConfig] = None,
    train_loader: Optional[DataLoader] = None,
    snapshot: Optional[SentinelSnapshot] = None,
) -> Dict[str, float]:
    model.to(device).eval()
    ce_loss = nn.CrossEntropyLoss().to(device)
    results: Dict[str, float] = {}

    # Clean pass
    clean_correct = 0
    total = 0
    losses = 0.0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)["logits"]
            loss = ce_loss(logits, yb)
            losses += float(loss) * xb.size(0)
            clean_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    results["clean_acc"] = clean_correct / max(1, total)
    results["clean_loss"] = losses / max(1, total)

    # FGSM & PGD
    def _attack(name: str, base_cfg: AdvConfig):
        worst_acc = 1.0
        avg_acc = []
        avg_asr = []
        for eps in epsilons:
            atk_cfg = AdvConfig(
                enabled=True,
                method=name,
                epsilon=eps,
                steps=base_cfg.steps,
                mix_ratio=1.0,
                random_start=True,
            )
            correct = 0
            total = 0
            flips = 0
            for xb, yb in data_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                clean_logits = model(xb)["logits"]
                adv_x = generate_adversarial(model, xb, yb, ce_loss, atk_cfg)
                adv_logits = model(adv_x)["logits"]
                preds = adv_logits.argmax(dim=1)
                clean_pred = clean_logits.argmax(dim=1)
                flips += (preds != clean_pred).sum().item()
                correct += (preds == yb).sum().item()
                total += yb.size(0)
            acc = correct / max(1, total)
            asr = flips / max(1, total)
            results[f"{name}_acc_eps{eps}"] = acc
            results[f"{name}_asr_eps{eps}"] = asr
            avg_acc.append(acc)
            avg_asr.append(asr)
            worst_acc = min(worst_acc, acc)
        results[f"{name}_avg_robust_acc"] = sum(avg_acc) / max(1, len(avg_acc))
        results[f"{name}_worst_case_acc"] = worst_acc
        results[f"{name}_avg_asr"] = sum(avg_asr) / max(1, len(avg_asr))

    _attack("fgsm", AdvConfig(enabled=True, method="fgsm", steps=1))
    _attack("pgd", AdvConfig(enabled=True, method="pgd", steps=5))

    # QC-Poison benchmark
    if poison_cfg and poison_cfg.enabled:
        qc_metrics = run_qc_poison_benchmark(
            model,
            train_loader=train_loader,
            eval_loader=data_loader,
            device=device,
            epsilons=epsilons,
            cfg=poison_cfg,
            snapshot=snapshot,
        )
        for eps, metrics in qc_metrics.items():
            for key, value in metrics.items():
                results[f"qc_poison_{key}_eps{eps}"] = value
        results["qc_poison_worst_case_acc"] = min(m["acc"] for m in qc_metrics.values())

    return results


def _restore_quantum_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    name_to_param = dict(model.named_parameters())
    for name, tensor in state.items():
        if name in name_to_param:
            name_to_param[name].data.copy_(tensor.to(name_to_param[name].device))


def _compute_poison_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_mode: str,
    attack_mode: str,
    hybrid_lambda: float,
    loss_fn: nn.Module,
):
    num_classes = logits.shape[-1]
    target_y = (1 - targets) if (target_mode == "flip" and num_classes == 2) else (targets + 1) % num_classes
    ce_true = loss_fn(logits, targets)
    ce_target = loss_fn(logits, target_y)
    true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    top_vals, top_idx = logits.topk(2, dim=1)
    alt_logits = torch.where(top_idx[:, 0] == targets, top_vals[:, 1], top_vals[:, 0])
    margin_loss = (alt_logits - true_logits).mean()

    if attack_mode == "targeted":
        loss = -ce_target
    elif attack_mode == "untargeted":
        loss = ce_true
    elif attack_mode == "margin":
        loss = margin_loss
    else:
        margin_w = hybrid_lambda
        untarget_w = (1 - margin_w) * 0.4
        target_w = (1 - margin_w) * 0.6
        loss = margin_w * margin_loss + untarget_w * ce_true - target_w * ce_target

    return loss, {"ce_true": ce_true.item(), "ce_target": ce_target.item(), "margin": margin_loss.item()}


def run_qc_poison_benchmark(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    epsilons: Sequence[float],
    cfg: PoisonEvalConfig,
    snapshot: Optional[SentinelSnapshot] = None,
) -> Dict[float, Dict[str, float]]:
    model = model.to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)
    results: Dict[float, Dict[str, float]] = {}

    for eps in epsilons:
        base_state = {name: param.detach().clone() for name, param in _collect_quantum_parameters(model)}
        best_state = None
        best_score = float("inf")

        for restart in range(max(1, cfg.restarts)):
            _restore_quantum_state(model, base_state)
            momentum_buf: Dict[str, torch.Tensor] = {}
            if cfg.random_start:
                for name, param in _collect_quantum_parameters(model):
                    noise = torch.empty_like(param).uniform_(-eps, eps)
                    param.data.copy_(base_state[name] + noise)
                    if cfg.clamp_params:
                        param.data.clamp_(-math.pi, math.pi)

            iterator = tqdm(
                train_loader,
                desc=f"QC-Poison ε={eps:.2f} (restart {restart+1}/{cfg.restarts})",
                leave=False,
            ) if cfg.use_tqdm else train_loader

            for step_idx, (xb, yb) in enumerate(iterator):
                if cfg.max_batches_per_step is not None and step_idx >= cfg.max_batches_per_step:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                model.zero_grad(set_to_none=True)

                with amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(xb)["logits"]
                    loss, _ = _compute_poison_objective(
                        logits,
                        yb,
                        target_mode=cfg.target_mode,
                        attack_mode=cfg.attack_mode,
                        hybrid_lambda=cfg.hybrid_lambda,
                        loss_fn=ce_loss,
                    )
                loss.backward()

                for name, param in _collect_quantum_parameters(model):
                    grad = param.grad
                    if grad is None:
                        continue
                    if cfg.grad_mode == "sign":
                        grad_update = grad.sign()
                    elif cfg.grad_mode == "normalized":
                        denom = grad.abs().amax() + 1e-12
                        grad_update = grad / denom
                    else:
                        grad_update = grad
                    if cfg.momentum > 0:
                        buf = momentum_buf.setdefault(name, torch.zeros_like(grad_update))
                        buf.mul_(cfg.momentum).add_(grad_update)
                        grad_update = buf
                    param.data.add_(cfg.step_scale * eps / max(1, cfg.steps) * grad_update)
                    delta = param.data - base_state[name]
                    delta.clamp_(-eps, eps)
                    param.data.copy_(base_state[name] + delta)
                    if cfg.clamp_params:
                        param.data.clamp_(-math.pi, math.pi)

            subset_metrics = _evaluate_subset(model, eval_loader, device, cfg.eval_batches, ce_loss, device.type == "cuda")
            score = subset_metrics["loss"]
            if score < best_score:
                best_score = score
                best_state = {name: param.detach().clone() for name, param in _collect_quantum_parameters(model)}

        if best_state is None:
            best_state = {name: param.detach().clone() for name, param in _collect_quantum_parameters(model)}
        _restore_quantum_state(model, best_state)

        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)["logits"]
                loss = ce_loss(logits, yb)
                total_loss += float(loss) * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)

        mean_shift, linf_shift = _quantum_param_shift(best_state, base_state)
        results[eps] = {
            "acc": correct / max(1, total),
            "loss": total_loss / max(1, total),
            "param_l1": mean_shift,
            "param_linf": linf_shift,
        }
        _restore_quantum_state(model, base_state)

    if snapshot is not None:
        _restore_quantum_state(model, snapshot.anchor_params)
    model.eval()
    return results


def _quantum_param_shift(poisoned: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    if not base:
        return float("nan"), float("nan")
    l1_vals = []
    linf_vals = []
    for name, base_tensor in base.items():
        if name not in poisoned:
            continue
        delta = poisoned[name] - base_tensor
        l1_vals.append(delta.abs().mean().item())
        linf_vals.append(delta.abs().max().item())
    if not l1_vals:
        return float("nan"), float("nan")
    return float(sum(l1_vals) / len(l1_vals)), float(max(linf_vals))


def _evaluate_subset(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: Optional[int],
    loss_fn: nn.Module,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (xb, yb) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
                logits = model(xb)["logits"]
                loss = loss_fn(logits, yb)
            total_loss += float(loss) * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}
