# run_latent_chain.py
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS safety

import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from env_risky_chain import RiskyChainEnv
from latent_cvar_modules import LatentCfg
from latent_cvar_model import LatentCVaRAgent
from utils_hazard import ensure_dir, append_log_csv, make_pos_idx_by_bins
from variance_meter import pg_signal_proxy

# ---- device pick: MPS -> CUDA -> CPU ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ====== knobs ======
alpha = 0.95
PRINT_EVERY = 50
TOTAL_STEPS = 3000
BATCH_B = 128
BATCH_T = 12
EPSILON = 0.10           # ε-greedy exploration
LR = 3e-4

# ====== rollout with POLICY actions (uses encoder; feeds log1p(cost) to encoder only) ======
@torch.no_grad()
def rollout_batch(agent, B=BATCH_B, T=BATCH_T, epsilon=EPSILON):
    obs_dim, act_dim = 2, 2
    s_seq = torch.zeros(B, T, obs_dim)
    a_seq = torch.zeros(B, T, act_dim)
    c_seq = torch.zeros(B, T, 1)          # RAW costs for metrics
    s_t_last = torch.zeros(B, obs_dim)

    G = torch.zeros(B)
    C = torch.zeros(B)

    envs = [RiskyChainEnv(seed=1337 + i) for i in range(B)]
    for b in range(B):
        obs = envs[b].reset()
        past_s, past_a, past_c_raw, past_c_enc = [], [], [], []

        for t in range(T):
            # ---- build encoder prefix (CPU encoder friendly) ----
            s_tensor = torch.tensor(np.array(past_s + [obs]), dtype=torch.float32).unsqueeze(0)  # [1,t+1,2]
            if len(past_a) == 0:
                a_tensor = torch.zeros(1, 1, 2)
                c_tensor = torch.zeros(1, 1, 1)
            else:
                a_tensor = torch.tensor(np.array(past_a), dtype=torch.float32).unsqueeze(0)
                # FEED ENCODED COSTS (log1p) to encoder
                c_tensor = torch.tensor(np.array(past_c_enc), dtype=torch.float32).unsqueeze(0)
            # pad a/c to match s length
            if a_tensor.shape[1] < s_tensor.shape[1]:
                pad = s_tensor.shape[1] - a_tensor.shape[1]
                a_tensor = torch.cat([a_tensor, torch.zeros(1, pad, 2)], dim=1)
                c_tensor = torch.cat([c_tensor, torch.zeros(1, pad, 1)], dim=1)

            # encode
            enc_device = next(agent.encoder.parameters()).device
            x_seq = torch.cat([s_tensor, a_tensor, c_tensor], dim=-1).to(enc_device)
            z_seq, _ = agent.encode_sequence(x_seq)
            z_t = z_seq[:, -1, :].to(device)
            if torch.isnan(z_t).any():
                z_t = torch.zeros_like(z_t)

            # policy on main device
            s_cur = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = agent.policy(s_cur, z_t)
            # sanitize logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp_(-20.0, 20.0)
            dist = torch.distributions.Categorical(logits=logits)
            if np.random.rand() < epsilon:
                a = np.random.choice([0, 1])
            else:
                a = int(dist.sample().item())

            # env step
            next_obs, cost_raw = envs[b].step(a)
            cost_enc = float(np.log1p(cost_raw))  # bounded growth for encoder

            # store
            a_one = np.array([1.0, 0.0]) if a == 0 else np.array([0.0, 1.0])
            past_s.append(obs)
            past_a.append(a_one)
            past_c_raw.append([cost_raw])
            past_c_enc.append([cost_enc])

            s_seq[b, t, :] = torch.tensor(obs, dtype=torch.float32)
            a_seq[b, t, :] = torch.tensor(a_one, dtype=torch.float32)
            c_seq[b, t, 0] = float(cost_raw)  # RAW for metrics

            obs = next_obs

        s_t_last[b, :] = torch.tensor(obs, dtype=torch.float32)
        C[b] = float(np.sum([c[0] for c in past_c_raw]))  # RAW costs sum
        G[b] = C[b]

    # VaR on CPU to avoid MPS stalls
    eta_scalar = torch.quantile(C.detach().to("cpu"), q=alpha).item()
    eta = torch.full((B,), eta_scalar)

    # hazard bins for InfoNCE
    excess = torch.clamp(C - eta, min=0.0)
    pos_idx = make_pos_idx_by_bins(excess, n_bins=5)

    # move to main device
    s_seq = s_seq.to(device); a_seq = a_seq.to(device); c_seq = c_seq.to(device)
    s_t_last = s_t_last.to(device); G = G.to(device); C = C.to(device)
    eta = eta.to(device); pos_idx = pos_idx.to(device)

    batch = {
        "s_seq": s_seq, "a_seq": a_seq, "c_seq": c_seq,
        "s_t": s_t_last, "G": G, "C": C, "eta": eta, "pos_idx": pos_idx
    }
    # roughly: fraction of risky actions taken
    pi_risky = float((a_seq[..., 1].sum() / (B * T)).item())
    return batch, pi_risky, eta_scalar

# ====== agent / opt ======
obs_dim, act_dim = 2, 2
cfg = LatentCfg(obs_dim=obs_dim, act_dim=act_dim, z_dim=32, hidden=128, gru_layers=1)
agent = LatentCVaRAgent(cfg, use_excess_head=True, use_quantile_head=True, use_contrastive=True).to(device)
# keep encoder on CPU for MPS stability
agent.encoder.to("cpu")
opt = optim.Adam(agent.parameters(), lr=LR)

# ====== logging ======
run_dir = "runs/risky_chain"
ensure_dir(run_dir)
csv_path = f"{run_dir}/metrics.csv"
field_order = ["step","meanC","eta","cvar","pi_risky",
               "loss_actor","loss_critic","loss_excess","loss_quantile","loss_contrast",
               "proxy_g_var","w_var","adv_var"]

# ====== train loop ======
for step in range(1, TOTAL_STEPS + 1):
    with torch.no_grad():
        batch, risky_frac, eta_scalar = rollout_batch(agent, B=BATCH_B, T=BATCH_T, epsilon=EPSILON)

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

    # diagnostics with CPU quantile
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

# ----- smoothed plot -----
df = pd.read_csv(csv_path)
df[["meanC", "eta", "cvar"]] = df[["meanC", "eta", "cvar"]].rolling(20).mean()
plt.figure(figsize=(8,5))
plt.plot(df["step"], df["meanC"], label="meanC")
plt.plot(df["step"], df["eta"], label="VaR η")
plt.plot(df["step"], df["cvar"], label="CVaR")
plt.xlabel("step"); plt.ylabel("value"); plt.legend(); plt.tight_layout()
plt.savefig(f"{run_dir}/curves_smoothed.png", dpi=200); plt.close()
print(f"Saved smoothed plot to {run_dir}/curves_smoothed.png")