# run_latent_bandit.py
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

# ---- Risky Bandit generator (multi-step to form sequences) ----
def sample_step_cost(a, p_tail=0.08, safe_mu=0.2, safe_sigma=0.02, risky_mu=0.1):
    if a == 0:
        return np.clip(np.random.normal(safe_mu, safe_sigma), 0.0, 5.0)
    # risky arm: heavy-tail with prob p_tail
    if np.random.rand() < p_tail:
        xm, alpha = 1.0, 1.5
        return float(np.random.pareto(alpha) + xm)  # heavy tail
    return max(0.0, np.random.normal(risky_mu, 0.05))

def make_batch(B=64, T=8, alpha=0.95):
    obs_dim, act_dim = 2, 2
    s_seq = torch.zeros(B, T, obs_dim, device=device)
    a_seq = torch.zeros(B, T, act_dim, device=device)
    c_seq = torch.zeros(B, T, 1, device=device)
    s_t = torch.zeros(B, obs_dim, device=device)

    G = torch.zeros(B, device=device)
    C = torch.zeros(B, device=device)
    risky_frac = 0.0

    for b in range(B):
        actions, costs = [], []
        for t in range(T):
            a = np.random.choice([0, 1], p=[0.5, 0.5])
            actions.append(a)
            a_seq[b, t, :] = torch.tensor([1, 0], device=device) if a == 0 else torch.tensor([0, 1], device=device)
            c = sample_step_cost(a)
            costs.append(c)
            c_seq[b, t, 0] = c
        C[b] = sum(costs)
        G[b] = C[b]                     # bandit: use total cost as proxy for cost-to-go
        risky_frac += sum(actions) / T

    # batch VaR and hazard bins for InfoNCE
    eta_scalar = torch.quantile(C.detach(), q=alpha).detach()
    eta = eta_scalar.expand(B)
    excess = torch.clamp(C - eta, min=0.0)
    pos_idx = make_pos_idx_by_bins(excess, n_bins=5)

    batch = {"s_seq": s_seq, "a_seq": a_seq, "c_seq": c_seq,
             "s_t": s_t, "G": G, "C": C, "eta": eta, "pos_idx": pos_idx}
    return batch, (risky_frac / B), eta_scalar.item()

# ---- config & agent ----
obs_dim, act_dim = 2, 2
cfg = LatentCfg(obs_dim=obs_dim, act_dim=act_dim, z_dim=32, hidden=128, gru_layers=1)
agent = LatentCVaRAgent(cfg, use_excess_head=True, use_quantile_head=True, use_contrastive=True).to(device)
opt = optim.Adam(agent.parameters(), lr=3e-4)

alpha = 0.95
print_every = 100

# logging
run_dir = "runs/latent_bandit"
ensure_dir(run_dir)
csv_path = f"{run_dir}/metrics.csv"
field_order = ["step","meanC","eta","cvar","pi_risky",
               "loss_actor","loss_critic","loss_excess","loss_quantile","loss_contrast",
               "proxy_g_var","w_var","adv_var"]

for step in range(1, 3001):
    batch, risky_frac, eta_scalar = make_batch(B=64, T=8, alpha=alpha)
    for k in batch:
        batch[k] = batch[k].to(device)

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

    # diagnostics
    C = batch["C"].detach()
    excess = torch.clamp(C - eta_scalar, min=0.0)
    cvar = eta_scalar + (excess.mean().item() / (1 - alpha))
    proxy = pg_signal_proxy(C, batch["eta"], batch["G"], aux["bphi"], alpha)

    if step % print_every == 0:
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