"""
PredictPersist — one variable change from TemporalPrediction.

Same W, same LMS update. Action persists until pred_err > median(recent),
then cycles to next action. No second system. No frozen similarity.
"""

import torch


class PredictPersist:
    def __init__(self, d, n_actions, device='cpu'):
        self.d = d
        self.n_actions = n_actions
        self.device = device
        self.W = torch.zeros(d, d, device=device)
        self.prev = None
        self.pred_err = 0.0
        self.action = 0
        self.err_history = []
        self.window = 100

    def step(self, x):
        x = x.to(self.device).float()
        if self.prev is None:
            self.prev = x.clone()
            return self.action

        pred = self.W @ self.prev
        err = x - pred
        self.pred_err = err.norm().item()
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W += torch.outer(err, self.prev) / denom

        self.err_history.append(self.pred_err)
        if len(self.err_history) > self.window:
            self.err_history = self.err_history[-self.window:]

        if len(self.err_history) >= 10:
            threshold = sorted(self.err_history)[len(self.err_history) // 2]
            if self.pred_err > threshold:
                self.action = (self.action + 1) % self.n_actions

        self.prev = x.clone()
        return self.action
