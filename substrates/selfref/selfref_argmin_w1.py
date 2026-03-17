"""
SelfRef variant: argmax for w0 (recognition), argmin for w1 (exploration).
Hypothesis: recognition is exploitation, action selection is exploration.
"""
import torch
import torch.nn.functional as F


class SelfRefArgminW1:
    def __init__(self, d, device='cuda'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device

    def step(self, x, n_actions):
        x_n = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        sims = self.V @ x_n
        w0 = sims.argmax().item()          # recognition: most familiar (stable codebook)

        ref = self.V @ self.V[w0]
        ref[w0] = -float('inf')
        w1 = ref.argmin().item()           # ← CHANGED: argmin — most novel instruction

        action = w1 % n_actions

        lr0 = (1.0 - sims[w0].clamp(0, 1)).item()
        self.V[w0] = F.normalize(
            self.V[w0] + lr0 * (x_n - self.V[w0]), dim=0)

        lr1 = (1.0 - ref[w1].clamp(0, 1)).item()
        self.V[w1] = F.normalize(
            self.V[w1] + lr1 * (self.V[w0] - self.V[w1]), dim=0)

        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()

        if sims[w0].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action
