# latent_cvar_losses.py
import torch
import torch.nn.functional as F

def pinball_loss(pred_eta, C, alpha: float):
    """Quantile regression (pinball) loss for VaR."""
    u = C - pred_eta
    return torch.mean(torch.maximum(alpha * u, (alpha - 1.0) * u))

def info_nce(z, pos_idx, temperature: float = 0.1):
    """Contrastive InfoNCE with in-batch negatives; positives via pos_idx."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.t()                # [B,B]
    logits = sim / temperature
    targets = pos_idx              # [B]
    return F.cross_entropy(logits, targets)

def critic_excess_loss(b_phi_vals, y_excess):
    return F.mse_loss(b_phi_vals, y_excess)

def excess_predictor_loss(gpsi_vals, y_excess):
    return F.mse_loss(gpsi_vals, y_excess)

def policy_cvar_pg_loss(logp_actions, adv_like, C, eta, alpha: float):
    """
    CVaR-weighted policy gradient loss using a baseline-derived advantage.
    g_i ∝ w_i * (G_i - b_phi_i) with w_i = (C_i - eta)+ / (1 - alpha).
    """
    w = torch.clamp(C - eta, min=0.0) / max(1e-6, (1.0 - alpha))
    while w.dim() < logp_actions.dim():
        w = w.unsqueeze(-1)
    return -(w * logp_actions * adv_like.detach()).mean()