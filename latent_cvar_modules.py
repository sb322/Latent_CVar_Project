# latent_cvar_modules.py
import torch
import torch.nn as nn

def mlp(sizes, act=nn.ELU, last_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i != len(sizes) - 2:
            layers.append(act())
    if last_act is not None:
        layers.append(last_act())
    return nn.Sequential(*layers)

class LatentCfg:
    def __init__(self, obs_dim, act_dim, cost_dim=1, z_dim=32, hidden=128, gru_layers=1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cost_dim = cost_dim
        self.z_dim = z_dim
        self.hidden = hidden
        self.gru_layers = gru_layers

class GRUEncoder(nn.Module):
    """f_theta encodes (s,a,c) prefixes into Z_t."""
    def __init__(self, cfg: LatentCfg):
        super().__init__()
        self.input_dim = cfg.obs_dim + cfg.act_dim + cfg.cost_dim
        self.gru = nn.GRU(self.input_dim, cfg.hidden, num_layers=cfg.gru_layers, batch_first=True)
        self.proj = nn.Linear(cfg.hidden, cfg.z_dim)

    def forward(self, x_seq, h=None):
        out, h_T = self.gru(x_seq, h)         # [B,T,H], [L,B,H]
        z = self.proj(out)                    # [B,T,z_dim]
        return z, h_T

class CategoricalPolicy(nn.Module):
    """π(a|s,Z): 2-action categorical policy (bandit-style)."""
    def __init__(self, cfg: LatentCfg):
        super().__init__()
        in_dim = cfg.obs_dim + cfg.z_dim
        self.net = mlp([in_dim, cfg.hidden, cfg.hidden, 2])

    def forward(self, s_t, z_t):
        logits = self.net(torch.cat([s_t, z_t], dim=-1))
        return logits

class CriticTailExcess(nn.Module):
    """b_phi(s,Z) ≈ E[(C-eta)+ | s,Z] (variance-reduction baseline)."""
    def __init__(self, cfg: LatentCfg):
        super().__init__()
        in_dim = cfg.obs_dim + cfg.z_dim
        self.net = mlp([in_dim, cfg.hidden, cfg.hidden, 1])

    def forward(self, s_t, z_t):
        return self.net(torch.cat([s_t, z_t], dim=-1)).squeeze(-1)

class ExcessPredictor(nn.Module):
    """g_psi(Z) ≈ (C-eta)+ (aux head to shape Z)."""
    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.net = mlp([z_dim, hidden, hidden, 1])

    def forward(self, z_t):
        return self.net(z_t).squeeze(-1)

class QuantileHead(nn.Module):
    """q_xi(Z) predicts VaR eta_alpha via pinball loss."""
    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.net = mlp([z_dim, hidden, hidden, 1])

    def forward(self, z_t):
        return self.net(z_t).squeeze(-1)