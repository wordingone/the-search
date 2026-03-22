"""
Constraint Map Validation — 6 MinimalLVQ variants.

Each run changes exactly one element of Run 0 (MinimalLVQ baseline).
Failure confirms the changed element is forced (I). Success overturns U claim.

Run 0: BASELINE     — Minimal Forced Substrate (the exact spec)
Run 1: L1_NORM      — L1 normalization instead of L2 (tests U: L2 norm)
Run 2: FIXED_LR     — lr=0.5 instead of 1-sim (tests U: adaptive lr)
Run 3: MEAN_THRESH  — mean instead of median threshold (tests U: gauge symmetry)
Run 4: CONTENT_ACT  — V[w][:n] argmax instead of w%n (tests U: index action)
Run 5: LR_SQ        — lr=1-sim^2 instead of 1-sim (tests U4 minimality)
"""

import torch
import torch.nn.functional as F


class MinimalLVQ:
    """Run 0: BASELINE. the exact MinimalLVQ (depth 1, forced elements only).
    Prediction: Level 1 by ~26K steps."""

    def __init__(self, d, device='cpu'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device
        self.last_thresh = 0.0

    def step(self, x, n_actions):
        x_n = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w = sims.argmax().item()
        action = w % n_actions

        lr = (1.0 - sims[w].clamp(0, 1)).item()
        self.V[w] = F.normalize(self.V[w] + lr * (x_n - self.V[w]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class MinimalLVQ_L1:
    """Run 1: L1 NORMALIZATION. Replace F.normalize (L2) with L1 norm.
    Tests: is L2 forced? Prediction: fails — timer dim dominates under L1."""

    def __init__(self, d, device='cpu'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device
        self.last_thresh = 0.0

    def _l1(self, x):
        return x.float() / x.float().abs().sum().clamp(min=1e-8)

    def step(self, x, n_actions):
        x_n = self._l1(x.to(self.device).float())

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w = sims.argmax().item()
        action = w % n_actions

        lr = (1.0 - sims[w].clamp(0, 1)).item()
        self.V[w] = self._l1(self.V[w] + lr * (x_n - self.V[w]))

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class MinimalLVQ_FixedLR:
    """Run 2: FIXED LR = 0.5. Replace 1-sim with constant 0.5.
    Tests: is adaptive lr forced? Prediction: might navigate (weak kill arg)."""

    def __init__(self, d, device='cpu'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device
        self.last_thresh = 0.0

    def step(self, x, n_actions):
        x_n = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w = sims.argmax().item()
        action = w % n_actions

        lr = 0.5
        self.V[w] = F.normalize(self.V[w] + lr * (x_n - self.V[w]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class MinimalLVQ_MeanThresh:
    """Run 3: MEAN THRESHOLD. Replace median with mean for spawn threshold.
    Tests: gauge symmetry claim. Prediction: navigates (mean ~= median at scale)."""

    def __init__(self, d, device='cpu'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device
        self.last_thresh = 0.0

    def step(self, x, n_actions):
        x_n = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w = sims.argmax().item()
        action = w % n_actions

        lr = (1.0 - sims[w].clamp(0, 1)).item()
        self.V[w] = F.normalize(self.V[w] + lr * (x_n - self.V[w]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.mean().item()  # MEAN not median
        self.last_thresh = thresh
        if sims[w].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class MinimalLVQ_ContentAction:
    """Run 4: CONTENT-BASED ACTION. action = V[w][:n_actions].argmax().
    Tests: is index-mod action forced? Prediction: fails — first dims have no meaning."""

    def __init__(self, d, device='cpu'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device
        self.last_thresh = 0.0

    def step(self, x, n_actions):
        x_n = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w = sims.argmax().item()
        action = self.V[w][:n_actions].argmax().item()  # content-based

        lr = (1.0 - sims[w].clamp(0, 1)).item()
        self.V[w] = F.normalize(self.V[w] + lr * (x_n - self.V[w]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class MinimalLVQ_LrSq:
    """Run 5: lr = 1 - sim^2. Valid convex weight, stronger than 1-sim near 0.
    Tests: U4 minimality of 1-sim. Prediction: navigates (both are valid)."""

    def __init__(self, d, device='cpu'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device
        self.last_thresh = 0.0

    def step(self, x, n_actions):
        x_n = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w = sims.argmax().item()
        action = w % n_actions

        s = sims[w].clamp(0, 1).item()
        lr = 1.0 - s * s  # 1 - sim^2
        self.V[w] = F.normalize(self.V[w] + lr * (x_n - self.V[w]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action
