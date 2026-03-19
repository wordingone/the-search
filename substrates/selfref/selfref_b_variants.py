"""
Constraint Map Validation (Round B) — SelfRef (depth 2) variants.

Run 0b: SelfRef exact (from selfref.py) — MUST navigate Level 1
Run 1b: L1 normalization at depth 2 (|x|/||x||_1)
Run 2b: fixed lr=0.5 at depth 2 (both lr0 and lr1)
Run 3b: SKIP (mean=median gauge symmetry confirmed at depth 1)
Run 4b: content-based action at depth 2 (V[w1][:n].argmax())
Run 5b: lr=1-sim^2 at depth 2 (both attractors) — KEY RUN
Run 6b: depth 1 + random non-winner attract (isolate chain vs attract)
"""

import random
import torch
import torch.nn.functional as F


class SelfRef_L1:
    """Run 1b: L1 normalization at depth 2.
    Replace both F.normalize calls with |x|/||x||_1.
    Prediction: still fails. L1 was catastrophic at depth 1 (cb=4)."""

    def __init__(self, d, device='cpu'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device

    def _l1(self, x):
        x = x.float()
        return x.abs() / x.abs().sum().clamp(min=1e-8)

    def step(self, x, n_actions):
        x_n = self._l1(x.to(self.device).float())

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w0 = sims.argmax().item()

        ref = self.V @ self.V[w0]
        ref[w0] = -float('inf')
        w1 = ref.argmax().item()

        action = w1 % n_actions

        lr0 = (1.0 - sims[w0].clamp(0, 1)).item()
        self.V[w0] = self._l1(self.V[w0] + lr0 * (x_n - self.V[w0]))

        lr1 = (1.0 - ref[w1].clamp(0, 1)).item()
        self.V[w1] = self._l1(self.V[w1] + lr1 * (self.V[w0] - self.V[w1]))

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w0].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class SelfRef_FixedLR:
    """Run 2b: fixed lr=0.5 at depth 2 (both lr0 and lr1).
    Prediction: fails. cb=4 at depth 1 — adaptive lr drives codebook growth."""

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
        w0 = sims.argmax().item()

        ref = self.V @ self.V[w0]
        ref[w0] = -float('inf')
        w1 = ref.argmax().item()

        action = w1 % n_actions

        lr0 = 0.5
        self.V[w0] = F.normalize(self.V[w0] + lr0 * (x_n - self.V[w0]), dim=0)

        lr1 = 0.5
        self.V[w1] = F.normalize(self.V[w1] + lr1 * (self.V[w0] - self.V[w1]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w0].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class SelfRef_ContentAction:
    """Run 4b: content-based action at depth 2.
    action = V[w1][:n_actions].argmax() instead of w1 % n_actions.
    Prediction: fails. Content-based had 98% dom at depth 1."""

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
        w0 = sims.argmax().item()

        ref = self.V @ self.V[w0]
        ref[w0] = -float('inf')
        w1 = ref.argmax().item()

        action = self.V[w1][:n_actions].argmax().item()  # content-based

        lr0 = (1.0 - sims[w0].clamp(0, 1)).item()
        self.V[w0] = F.normalize(self.V[w0] + lr0 * (x_n - self.V[w0]), dim=0)

        lr1 = (1.0 - ref[w1].clamp(0, 1)).item()
        self.V[w1] = F.normalize(self.V[w1] + lr1 * (self.V[w0] - self.V[w1]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w0].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class SelfRef_LrSq:
    """Run 5b: lr=1-sim^2 at depth 2 (both attractors). KEY RUN.
    If navigates: lr formula is genuine U (1-sim^2 at least as good as 1-sim).
    If fails: 1-sim is specifically forced."""

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
        w0 = sims.argmax().item()

        ref = self.V @ self.V[w0]
        ref[w0] = -float('inf')
        w1 = ref.argmax().item()

        action = w1 % n_actions

        s0 = sims[w0].clamp(0, 1).item()
        lr0 = 1.0 - s0 * s0
        self.V[w0] = F.normalize(self.V[w0] + lr0 * (x_n - self.V[w0]), dim=0)

        s1 = ref[w1].clamp(0, 1).item()
        lr1 = 1.0 - s1 * s1
        self.V[w1] = F.normalize(self.V[w1] + lr1 * (self.V[w0] - self.V[w1]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w0].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action


class SelfRef_Depth1RandAttract:
    """Run 6b: depth 1 + random non-winner attract.
    Isolates chain selection vs any inter-entry attraction.
    Prediction: fails. Random attract != chain-selected attract."""

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
        w0 = sims.argmax().item()

        action = w0 % n_actions  # depth 1 action

        lr0 = (1.0 - sims[w0].clamp(0, 1)).item()
        self.V[w0] = F.normalize(self.V[w0] + lr0 * (x_n - self.V[w0]), dim=0)

        # Random non-winner attract (not chain-selected)
        k = self.V.shape[0]
        candidates = [i for i in range(k) if i != w0]
        if candidates:
            w_rand = random.choice(candidates)
            sim_rw = (self.V[w_rand] @ self.V[w0]).clamp(0, 1).item()
            lr1 = 1.0 - sim_rw
            self.V[w_rand] = F.normalize(
                self.V[w_rand] + lr1 * (self.V[w0] - self.V[w_rand]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()
        self.last_thresh = thresh
        if sims[w0].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action
