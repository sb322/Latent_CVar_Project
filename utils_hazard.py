# utils_hazard.py
import os, csv, random, numpy as np
import torch
import matplotlib.pyplot as plt

def make_pos_idx_by_bins(excess: torch.Tensor, n_bins: int = 5, rng: random.Random | None = None):
    """
    InfoNCE positives: pick a random other sample from the SAME (C-eta)+ bin.
    """
    device = excess.device
    B = excess.numel()
    rng = rng or random.Random(123)

    qs = torch.linspace(0, 1, steps=n_bins + 1, device=device)
    edges = torch.quantile(excess, q=qs).tolist()
    x = excess.detach().cpu().numpy()
    bin_ids = np.digitize(x, edges[1:-1], right=True)  # 0..n_bins-1

    buckets = {b: [] for b in range(n_bins)}
    for i, b in enumerate(bin_ids): buckets[b].append(i)

    pos = []
    for i, b in enumerate(bin_ids):
        cand = buckets[b]
        if len(cand) <= 1:
            pos.append(i)
        else:
            j = i
            while j == i:
                j = rng.choice(cand)
            pos.append(j)
    return torch.tensor(pos, dtype=torch.long, device=device)

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def append_log_csv(path: str, row: dict, field_order: list[str]):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if new_file: w.writeheader()
        w.writerow({k: row.get(k, "") for k in field_order})

def plot_curves_from_csv(csv_path: str, out_png: str):
    xs, meanC, eta, cvar = [], [], [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["step"]))
            meanC.append(float(row["meanC"]))
            eta.append(float(row["eta"]))
            cvar.append(float(row["cvar"]))
    plt.figure()
    plt.plot(xs, meanC, label="meanC")
    plt.plot(xs, eta, label="VaR η")
    plt.plot(xs, cvar, label="CVaR")
    plt.xlabel("step"); plt.ylabel("value"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()