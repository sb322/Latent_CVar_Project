# variance_meter.py
import torch

@torch.no_grad()
def pg_signal_proxy(C: torch.Tensor, eta: torch.Tensor, G: torch.Tensor, bphi: torch.Tensor, alpha: float):
    """
    Proxy for per-sample PG magnitude before multiplying by score function.
    g_i ~ w_i * (G_i - bphi_i),  w_i = (C_i - eta)+/(1-alpha)
    """
    w = torch.clamp(C - eta, min=0.0) / max(1e-6, (1.0 - alpha))
    adv = (G - bphi)
    g = w * adv
    return {
        "proxy_g_mean": g.mean().item(),
        "proxy_g_var": g.var(unbiased=False).item(),
        "w_mean": w.mean().item(),
        "w_var": w.var(unbiased=False).item(),
        "adv_mean": adv.mean().item(),
        "adv_var": adv.var(unbiased=False).item(),
    }