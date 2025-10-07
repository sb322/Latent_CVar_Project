# latent_cvar_model.py
import torch
import torch.nn as nn

from latent_cvar_modules import LatentCfg, GRUEncoder, CategoricalPolicy, CriticTailExcess, ExcessPredictor, QuantileHead
from latent_cvar_losses import (pinball_loss, info_nce, critic_excess_loss,
                                excess_predictor_loss, policy_cvar_pg_loss)

class LatentCVaRAgent(nn.Module):
    """
    Encoder + policy + critic + optional latent heads.
    compute_losses(...) returns (total_loss, scalar_stats, aux_tensors).
    """
    def __init__(self, cfg: LatentCfg, use_excess_head=True, use_quantile_head=True, use_contrastive=True):
        super().__init__()
        self.cfg = cfg
        self.encoder = GRUEncoder(cfg)                  # we may place this on CPU
        self.policy = CategoricalPolicy(cfg)            # placed on main device
        self.critic = CriticTailExcess(cfg)             # placed on main device
        self.excess_head = ExcessPredictor(cfg.z_dim, cfg.hidden) if use_excess_head else None
        self.quantile_head = QuantileHead(cfg.z_dim, cfg.hidden) if use_quantile_head else None
        self.use_contrastive = use_contrastive

    def encode_sequence(self, x_seq, h=None):
        return self.encoder(x_seq, h)

    def act(self, s_t, z_t):
        logits = self.policy(s_t, z_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, logits

    def compute_losses(self, batch, alpha: float,
                       lambda_excess_pred=1.0, lambda_quantile=0.5, lambda_contrast=0.1):
        """
        batch keys:
          s_seq [B,T,obs], a_seq [B,T,act], c_seq [B,T,1], s_t [B,obs],
          G [B], C [B], eta [B] or scalar, pos_idx [B] (optional).
        """
        s_seq, a_seq, c_seq = batch["s_seq"], batch["a_seq"], batch["c_seq"]
        s_t, G, C, eta = batch["s_t"], batch["G"], batch["C"], batch["eta"]

        # Build encoder input on encoder's device (CPU if moved there), then bring back to main device
        x_seq = torch.cat([s_seq, a_seq, c_seq], dim=-1)
        enc_device = next(self.encoder.parameters()).device
        x_seq_enc = x_seq.to(enc_device, non_blocking=True)
        z_seq, _ = self.encode_sequence(x_seq_enc)
        z_seq = z_seq.to(s_t.device, non_blocking=True)
        z_t = z_seq[:, -1, :]

        # Latent-conditioned critic
        bphi = self.critic(s_t, z_t)
        y_excess = torch.clamp(C - eta, min=0.0)

        loss_critic = critic_excess_loss(bphi, y_excess)

        # Aux: (C - eta)+ from Z
        loss_excess = torch.tensor(0.0, device=z_t.device)
        if self.excess_head is not None:
            gpsi = self.excess_head(z_t)
            loss_excess = excess_predictor_loss(gpsi, y_excess)

        # Quantile head for eta
        loss_quantile = torch.tensor(0.0, device=z_t.device)
        pred_eta = None
        if self.quantile_head is not None:
            pred_eta = self.quantile_head(z_t)
            loss_quantile = pinball_loss(pred_eta, C, alpha)

        # Contrastive hazard clustering (InfoNCE)
        loss_contrast = torch.tensor(0.0, device=z_t.device)
        if self.use_contrastive and "pos_idx" in batch:
            loss_contrast = info_nce(z_t, batch["pos_idx"], temperature=0.1)

        # Policy loss: advantage-like = G - b_phi
        adv_like = G - bphi
        eta_for_actor = pred_eta if pred_eta is not None else eta
        loss_actor = policy_cvar_pg_loss(batch.get("logp_a", torch.zeros_like(G)),
                                         adv_like, C, eta_for_actor.detach(), alpha)

        total = (loss_actor + loss_critic
                 + lambda_excess_pred * loss_excess
                 + lambda_quantile * loss_quantile
                 + lambda_contrast * loss_contrast)

        stats = {
            "loss/total": float(total.detach().cpu()),
            "loss/actor": float(loss_actor.detach().cpu()),
            "loss/critic": float(loss_critic.detach().cpu()),
            "loss/excess": float(loss_excess.detach().cpu()),
            "loss/quantile": float(loss_quantile.detach().cpu()),
            "loss/contrast": float(loss_contrast.detach().cpu()),
            "eta/used": float((eta_for_actor.detach().mean()).cpu()),
            "excess/mean": float(y_excess.detach().mean().cpu()),
        }

        aux = {"bphi": bphi.detach(), "z_t": z_t.detach(),
               "pred_eta": (pred_eta.detach() if pred_eta is not None else None)}
        return total, stats, aux