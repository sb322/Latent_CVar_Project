# run_latent_bandit.py
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported MPS ops fall back to CPU

import numpy as np
import torch
import torch.optim as optim

from latent_cvar_modules import LatentCfg
from latent_cvar_model import LatentCVaRAgent
from utils_hazard import ensure_dir, append_log_csv, plot_curves_from_csv, make_pos_idx_by_bins
from variance_meter import pg_signal_proxy

# ---- device fallback: MPS -> CUDA -> CPU ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Speed hint (PyTorch 2.x can use this)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# Quick debug knobs for first run (so you see progress immediately)
PRINT_EVERY = 100
TOTAL_STEPS = 3000
BATCH_B = 128
BATCH_T = 8
alpha = 0.95

# ---- Risky Bandit generator (multi-step to form sequences) ----
def sample_step_cost(a, p_tail=0.08, safe_mu=0.2, safe_sigma=0.02, risky_mu=0.1):
    if a == 0:
        return np.clip(np.random.normal(safe_mu, safe_sigma), 0.0, 5.0)
    # risky arm: heavy-tail with prob p_tail
    if np.random.rand() < p_tail:
        xm, pa = 1.0, 1.5
        return float(np.random.pareto(pa) + xm)  # heavy tail
    return max(0.0, np.random.normal(risky_mu, 0.05))

def make_batch(B=BATCH_B, T=BATCH_T, alpha=alpha):
    obs_dim, act_dim = 2, 2
    s_seq = torch.zeros(B, T, obs_dim)
    a_seq = torch.zeros(B, T, act_dim)
    c_seq = torch.zeros(B, T, 1)
    s_t = torch.zeros(B, obs_dim)

    G = torch.zeros(B)
    C = torch.zeros(B)
    risky_frac = 0.0

    for b in range(B):
        actions, costs = [], []
        for t in range(T):
            a = np.random.choice([0, 1], p=[0.5, 0.5])
            actions.append(a)
            a_seq[b, t, :] = torch.tensor([1, 0]) if a == 0 else torch.tensor([0, 1])
            c = sample_step_cost(a)
            costs.append(c)
            c_seq[b, t, 0] = c
        C[b] = sum(costs)
        G[b] = C[b]                     # bandit: use total cost as proxy for cost-to-go
        risky_frac += sum(actions) / T

    # CPU quantile to avoid MPS stall
    eta_scalar = torch.quantile(C.detach().to("cpu"), q=alpha).item()
    eta = torch.full((B,), eta_scalar)

    # hazard bins for InfoNCE (works on CPU; device-neutral)
    excess = torch.clamp(C - eta, min=0.0)
    pos_idx = make_pos_idx_by_bins(excess, n_bins=5)

    # Move tensors to main device once
    s_seq = s_seq.to(device)
    a_seq = a_seq.to(device)
    c_seq = c_seq.to(device)
    s_t = s_t.to(device)
    G = G.to(device)
    C = C.to(device)
    eta = eta.to(device)
    pos_idx = pos_idx.to(device)

    batch = {"s_seq": s_seq, "a_seq": a_seq, "c_seq": c_seq,
             "s_t": s_t, "G": G, "C": C, "eta": eta, "pos_idx": pos_idx}
    return batch, (risky_frac / B), eta_scalar

# ---- config & agent ----
obs_dim, act_dim = 2, 2
cfg = LatentCfg(obs_dim=obs_dim, act_dim=act_dim, z_dim=32, hidden=128, gru_layers=1)

agent = LatentCVaRAgent(cfg, use_excess_head=True, use_quantile_head=True, use_contrastive=True).to(device)

# Run the encoder on CPU to avoid GRU-on-MPS stalls; keep policy/critic on main device
agent.encoder.to("cpu")

opt = optim.Adam(agent.parameters(), lr=3e-4)

# logging
run_dir = "runs/latent_bandit"
ensure_dir(run_dir)
csv_path = f"{run_dir}/metrics.csv"
field_order = ["step","meanC","eta","cvar","pi_risky",
               "loss_actor","loss_critic","loss_excess","loss_quantile","loss_contrast",
               "proxy_g_var","w_var","adv_var"]

for step in range(1, TOTAL_STEPS + 1):
    batch, risky_frac, eta_scalar = make_batch(B=BATCH_B, T=BATCH_T, alpha=alpha)

    opt.zero_grad(set_to_none=True)
    total, stats, aux = agent.compute_losses(
        batch, alpha=alpha,
        lambda_excess_pred=1.0,
        lambda_quantile=0.5,
        lambda_contrast=0.1
    )
    total.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    opt.step()

    # diagnostics (CPU quantile again to avoid MPS stall)
    C = batch["C"].detach()
    eta_scalar = torch.quantile(C.to("cpu"), q=alpha).item()
    excess = torch.clamp(C - eta_scalar, min=0.0)
    cvar = eta_scalar + (excess.mean().item() / (1 - alpha))

    proxy = pg_signal_proxy(C, batch["eta"], batch["G"], aux["bphi"], alpha)

    if step % PRINT_EVERY == 0:
        print(f"[{step}] meanC={C.mean().item():.3f}  eta={eta_scalar:.3f}  cvar={cvar:.3f}  "
              f"pi(risky)~{risky_frac:.2f}  lossA={stats['loss/actor']:.3f}  "
              f"lossCrit={stats['loss/critic']:.3f}  proxyVar={proxy['proxy_g_var']:.3f}")
        append_log_csv(csv_path, {
            "step": step,
            "meanC": C.mean().item(),
            "eta": eta_scalar,
            "cvar": cvar,
            "pi_risky": risky_frac,
            "loss_actor": stats["loss/actor"],
            "loss_critic": stats["loss/critic"],
            "loss_excess": stats["loss/excess"],
            "loss_quantile": stats["loss/quantile"],
            "loss_contrast": stats["loss/contrast"],
            "proxy_g_var": proxy["proxy_g_var"],
            "w_var": proxy["w_var"],
            "adv_var": proxy["adv_var"],
        }, field_order)

# finalize plot
plot_curves_from_csv(csv_path, f"{run_dir}/curves.png")
print(f"Saved logs to {csv_path} and plot to {run_dir}/curves.png")