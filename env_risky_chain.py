# env_risky_chain.py
import numpy as np

class RiskyChainEnv:
    """
    Two-state Markov chain with heavy-tailed hazards in the risky state.
    States: 0 = SAFE, 1 = RISKY
    Actions: 0 = prefer_safe, 1 = prefer_risky
      - In SAFE:  a=0 keeps SAFE w.p. 0.95, a=1 moves to RISKY w.p. 0.7
      - In RISKY: a=1 keeps RISKY w.p. 0.95, a=0 moves to SAFE w.p. 0.7
    Costs:
      SAFE:  N(0.20, 0.02^2), clipped to [0,5]
      RISKY: with p_tail -> Pareto tail; else N(0.10, 0.05^2)+
    """
    def __init__(self, p_tail=0.06, seed=None):
        self.rng = np.random.RandomState(seed)
        self.p_tail = p_tail
        self.state = 0

    def reset(self, safe_start_prob=0.8):
        self.state = 0 if self.rng.rand() < safe_start_prob else 1
        return self._obs()

    def step(self, action):
        s = self.state
        if s == 0:  # SAFE
            if action == 0:
                self.state = 0 if self.rng.rand() < 0.95 else 1
            else:
                self.state = 1 if self.rng.rand() < 0.70 else 0
        else:       # RISKY
            if action == 1:
                self.state = 1 if self.rng.rand() < 0.95 else 0
            else:
                self.state = 0 if self.rng.rand() < 0.70 else 1

        if self.state == 0:
            cost = float(np.clip(self.rng.normal(0.20, 0.02), 0.0, 5.0))
        else:
            if self.rng.rand() < self.p_tail:
                xm, alpha = 1.0, 1.5
                cost = float(self.rng.pareto(alpha) + xm)
            else:
                cost = float(max(0.0, self.rng.normal(0.10, 0.05)))
        return self._obs(), cost

    def _obs(self):
        return np.array([1.0, 0.0]) if self.state == 0 else np.array([0.0, 1.0])