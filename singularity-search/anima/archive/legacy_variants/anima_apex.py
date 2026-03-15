"""
ANIMA-Apex: Optimized for Conditional Reasoning
================================================

The consistent weakness across all ANIMA variants is conditional logic (76-94%).
This task requires: "if x > threshold then f(x) else g(x)"

ANIMA-Apex adds:
1. Explicit comparison mechanism (learn thresholds)
2. Dual-branch processing (then/else pathways)
3. Soft selection between branches

This is targeted surgery on the weakest link.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ANIMAApex(nn.Module):
    """
    ANIMA-Apex: Zero + explicit conditional branching.
    """

    def __init__(
        self,
        sensory_dim: int = 8,
        d_model: int = 16,
        output_dim: int = 4,
    ):
        super().__init__()

        self.sensory_dim = sensory_dim
        self.d = d_model
        self.output_dim = output_dim

        # === SENSING ===
        self.W_enc = nn.Linear(sensory_dim, d_model)
        self.W_from_W = nn.Linear(d_model, d_model, bias=False)
        self.W_from_I = nn.Linear(d_model, d_model, bias=False)
        self.W_from_A = nn.Linear(d_model, d_model, bias=False)
        self.W_gate = nn.Linear(d_model * 2, d_model)

        # === MEMORY (GRU) ===
        self.I_z = nn.Linear(d_model * 3, d_model)
        self.I_r = nn.Linear(d_model * 3, d_model)
        self.I_h = nn.Linear(d_model * 3, d_model)

        # === CONDITIONAL BRANCHING (NEW) ===
        # Learn when to branch
        self.condition_detector = nn.Linear(d_model * 2, 1)  # Scalar gate

        # Two processing pathways
        self.branch_then = nn.Linear(d_model, d_model)
        self.branch_else = nn.Linear(d_model, d_model)

        # === ACTION ===
        self.A_from_W = nn.Linear(d_model, d_model, bias=False)
        self.A_from_I = nn.Linear(d_model, d_model, bias=False)
        self.A_from_A = nn.Linear(d_model, d_model, bias=False)
        self.A_gate = nn.Linear(d_model * 2, d_model)

        # === OUTPUT ===
        self.phi = nn.Linear(d_model, output_dim)

        # === STATE ===
        self.W = None
        self.I = None
        self.A = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.orthogonal_(param)
                with torch.no_grad():
                    param.mul_(0.95 / max(param.abs().max(), 1e-6))
            elif 'bias' in name:
                nn.init.zeros_(param)

    def reset(self, batch_size: int = 1, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        self.W = torch.zeros(batch_size, self.d, device=device)
        self.I = torch.zeros(batch_size, self.d, device=device)
        self.A = torch.zeros(batch_size, self.d, device=device)

    def step(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.W is None:
            self.reset(obs.shape[0], obs.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # === SENSE ===
        x = torch.tanh(self.W_enc(obs))
        W_input = x + self.W_from_W(self.W) + self.W_from_I(self.I) + self.W_from_A(self.A)
        W_gate = torch.sigmoid(self.W_gate(torch.cat([self.I, self.A], -1)))
        W_new = torch.tanh(W_input) * W_gate

        # === MEMORY ===
        combined = torch.cat([W_new, self.I, self.A], -1)
        z = torch.sigmoid(self.I_z(combined))
        r = torch.sigmoid(self.I_r(combined))
        h_input = torch.cat([W_new, r * self.I, self.A], -1)
        h = torch.tanh(self.I_h(h_input))
        I_new = (1 - z) * self.I + z * h

        # === CONDITIONAL BRANCHING ===
        # Detect condition based on current W and I
        condition_input = torch.cat([W_new, I_new], -1)
        condition_gate = torch.sigmoid(self.condition_detector(condition_input))

        # Process through both branches
        then_out = torch.tanh(self.branch_then(I_new))
        else_out = torch.tanh(self.branch_else(I_new))

        # Soft selection
        branched = condition_gate * then_out + (1 - condition_gate) * else_out

        # === ACTION ===
        A_input = self.A_from_W(W_new) + self.A_from_I(I_new) + self.A_from_A(self.A)
        # Add branched contribution
        A_input = A_input + branched
        A_gate = torch.sigmoid(self.A_gate(torch.cat([W_new, I_new], -1)))
        A_new = torch.tanh(A_input) * A_gate

        self.W, self.I, self.A = W_new, I_new, A_new

        return {'action': self.phi(A_new), 'condition_gate': condition_gate}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        self.reset(batch, x.device)

        outputs = []
        for t in range(seq_len):
            result = self.step(x[:, t])
            outputs.append(result['action'])

        return torch.stack(outputs, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ANIMAApexExact(ANIMAApex):
    """ANIMA-Apex scaled to exact parameter count."""

    def __init__(
        self,
        target_params: int,
        sensory_dim: int = 8,
        output_dim: int = 4,
    ):
        d = self._find_dimension(target_params, sensory_dim, output_dim)
        super().__init__(
            sensory_dim=sensory_dim,
            d_model=d,
            output_dim=output_dim,
        )

        current = self.count_parameters()
        diff = target_params - current
        if diff > 0:
            self.register_parameter('_pad', nn.Parameter(torch.zeros(diff)))

    def _find_dimension(self, target: int, s_dim: int, o_dim: int) -> int:
        best_d = 16
        best_diff = float('inf')

        for d in range(8, 64):
            class TempModel(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.W_enc = nn.Linear(s_dim, dim)
                    self.W_from_W = nn.Linear(dim, dim, bias=False)
                    self.W_from_I = nn.Linear(dim, dim, bias=False)
                    self.W_from_A = nn.Linear(dim, dim, bias=False)
                    self.W_gate = nn.Linear(dim * 2, dim)
                    self.I_z = nn.Linear(dim * 3, dim)
                    self.I_r = nn.Linear(dim * 3, dim)
                    self.I_h = nn.Linear(dim * 3, dim)
                    self.condition_detector = nn.Linear(dim * 2, 1)
                    self.branch_then = nn.Linear(dim, dim)
                    self.branch_else = nn.Linear(dim, dim)
                    self.A_from_W = nn.Linear(dim, dim, bias=False)
                    self.A_from_I = nn.Linear(dim, dim, bias=False)
                    self.A_from_A = nn.Linear(dim, dim, bias=False)
                    self.A_gate = nn.Linear(dim * 2, dim)
                    self.phi = nn.Linear(dim, o_dim)

            temp = TempModel(d)
            params = sum(p.numel() for p in temp.parameters())
            del temp

            diff = abs(params - target)
            if diff < best_diff:
                best_diff = diff
                best_d = d

        return best_d


if __name__ == "__main__":
    model = ANIMAApex()
    print(f"ANIMA-Apex parameters: {model.count_parameters():,}")

    target = 8964
    exact = ANIMAApexExact(target)
    print(f"Exact model ({target} target): {exact.count_parameters():,} params")
