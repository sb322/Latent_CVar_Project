# Latent-Conditioned CVaR Policy Gradient (OCE–CVaR Bandit)

This repository implements a **latent-variable reinforcement learning framework** for **risk-sensitive policy optimization** under the **Conditional Value-at-Risk (CVaR)** objective.  
It extends the baseline OCE–CVaR policy gradient code with a **latent representation learner** that captures *hazard structure* in the environment, improving both sample efficiency and gradient stability.

---

## 🎯 Motivation

Standard policy-gradient methods in risk-sensitive reinforcement learning suffer from **high variance** when optimizing CVaR objectives:

\[
\max_\pi \; \text{CVaR}_\alpha(-C(\pi)) \;=\; \eta - \frac{1}{1-\alpha}\, \mathbb{E}\big[(C-\eta)_+\big],
\]

where \( C = \sum_t \gamma^t c_t \) is cumulative cost and \(\eta\) is the Value-at-Risk threshold.

The tail-excess term \((C-\eta)_+\) is often **noisy** and **sparse**, especially when extreme events are rare.  
Our method introduces a **latent variable \(Z\)**, learned from trajectory prefixes, such that conditioning on \(Z\) makes the excess term nearly deterministic:

\[
\mathbb{V}\big[(C-\eta)_+ \mid Z\big] \;\ll\; \mathbb{V}\big[(C-\eta)_+\big].
\]

This allows us to use a **latent-conditioned critic \(b_\phi(s,Z)\)** as a baseline in the CVaR policy gradient — significantly reducing variance and improving convergence.

---

## 🧩 Key Components

| Module | Description |
|--------|--------------|
| `GRUEncoder` | Encodes trajectory prefixes \((s,a,c)\) into latent embedding \(Z_t\). |
| `CategoricalPolicy` | Two-armed policy for the Risky Bandit environment. |
| `CriticTailExcess` | Estimates expected tail-excess \(\mathbb{E}[(C-\eta)_+ | s,Z]\) for variance reduction. |
| `ExcessPredictor` | Auxiliary head predicting \((C-\eta)_+\) directly from \(Z\) to regularize representation learning. |
| `QuantileHead` | Predicts the Value-at-Risk \(\eta_\alpha\) via pinball (quantile) loss. |
| `InfoNCE Contrastive Loss` | Encourages \(Z\) embeddings of trajectories with similar risk levels to cluster together. |

---

## 🧪 Environment: Risky Bandit

We use a **synthetic two-armed Risky Bandit** to emulate tail-risk behavior:

- **Safe arm (0):** cost ∼ Normal(μ=0.2, σ=0.02)
- **Risky arm (1):** cost ∼ heavy-tailed Pareto(α=1.5) with probability 0.08; else small Gaussian(μ=0.1)

This setup captures the essence of rare but catastrophic outcomes.  
The policy’s goal is to minimize CVaR of the total episode cost.

---

## ⚙️ Implementation Details

**Architecture:**
- Latent encoder: GRU → Linear projection (→ Z)
- Policy & critic: 2-layer ELU MLPs
- Hidden size: 128
- Latent dimension: 32

**Loss terms:**
\[
L = L_{\text{actor}} + L_{\text{critic}} + 
\lambda_{\text{excess}} L_{\text{excess}} +
\lambda_{\text{quantile}} L_{\text{quantile}} +
\lambda_{\text{contrast}} L_{\text{contrast}}
\]

**Variance proxy metric:**
We track the variance of \( w_i (G_i - b_\phi(s_i,Z_i)) \)
as an indicator of gradient variance reduction over training.

---

## 🚀 Running Locally (PyCharm or Terminal)

1. **Clone the repo**
   ```bash
   git clone https://github.com/sb322/cvar-pg-mps.git
   cd cvar-pg-mps
