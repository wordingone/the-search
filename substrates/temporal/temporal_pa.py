"""
TemporalPerAction — one variable change from TemporalPrediction.

Replace 1 W matrix with n_actions W matrices.
W_a learns p(x_{t+1} | x_t, a=k): only updated when action k was taken.
Action selection: pick action k with highest last prediction error (most novel).

R3 audit delta: adds W.shape[0] per action (n_actions W matrices vs 1).
All other elements identical to TemporalPrediction.
"""

import torch


class TemporalPerAction:
    def __init__(self, d, n_actions, device='cpu'):
        self.d = d
        self.n_actions = n_actions
        self.device = device
        self.W = [torch.zeros(d, d) for _ in range(n_actions)]
        self.prev = None
        self.last_action = 0
        # Initialize high so all actions are equally novel at start
        self.pred_err = [1e6] * n_actions

    def step(self, x):
        x = x.float()

        if self.prev is None:
            self.prev = x.clone()
            self.last_action = 0
            return 0

        # UPDATE: only W for the last action taken
        a = self.last_action
        pred = self.W[a] @ self.prev
        err = x - pred
        self.pred_err[a] = err.norm().item()
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W[a] += torch.outer(err, self.prev) / denom

        # ACT: pick action with highest prediction error (most novel)
        action = max(range(self.n_actions), key=lambda i: self.pred_err[i])

        self.prev = x.clone()
        self.last_action = action
        return action
