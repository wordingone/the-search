"""
TemporalRowNorm — one change from TemporalPrediction.

Same W, same LMS update. Row-normalize W before the action read.
Suppresses the timer's magnitude advantage; amplifies low-SV directions.
"""
import torch
import torch.nn.functional as F


class TemporalRowNorm:
    def __init__(self, d, n_actions, device='cpu'):
        self.d = d
        self.n_actions = n_actions
        self.device = device
        self.W = torch.zeros(d, d, device=device)
        self.prev = None
        self.pred_err = 0.0

    def step(self, x):
        x = x.to(self.device).float()
        if self.prev is None:
            self.prev = x.clone()
            return 0

        pred = self.W @ self.prev
        err = x - pred
        self.pred_err = err.norm().item()
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W += torch.outer(err, self.prev) / denom

        # Row-normalize for action selection only (W unchanged)
        W_n = F.normalize(self.W, dim=1)
        action = (W_n @ x).argmax().item() % self.n_actions

        self.prev = x.clone()
        return action
