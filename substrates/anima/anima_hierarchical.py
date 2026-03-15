"""
AnimaHierarchical: Hierarchical Temporal Horizons with Adaptive Mixing

Core Insight: Different tasks require different temporal horizons.
- Fast path: Immediate responses (decay=0.1) for gravity, implication
- Slow path: Long-term memory (SSM decay modes) for delay, sequence
- Meta path: Task context to adaptively mix fast/slow

Causal Design:
- Addresses EvolvedV2's failure on accumulation (projectile: 44%)
- Addresses ISSM's failure on threshold tasks (XOR: 0%)
- Combines best of both via learned horizon selection

Target Improvements:
- +20pp on sequence (fast path for simple extrapolation)
- +15pp on delay (slow path for retention)
- +10pp on projectile (adaptive accumulation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class HierarchicalTemporalCell(nn.Module):
    """
    Temporal cell with hierarchical state structure.

    State Components:
    - h_fast: [batch, d_model] - Rapid decay, immediate responses
    - h_slow: [batch, d_model, d_state] - Multi-scale SSM memory
    - h_meta: [batch, d_meta] - Task context for horizon selection
    """

    def __init__(self, d_model: int, d_state: int = 16, d_meta: int = 8, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_meta = d_meta

        # Fast path gates (GRU-like for immediate responses)
        self.fast_z = nn.Linear(d_model * 2, d_model, bias=bias)  # Update gate
        self.fast_r = nn.Linear(d_model * 2, d_model, bias=bias)  # Reset gate
        self.fast_h = nn.Linear(d_model * 2, d_model, bias=bias)  # Candidate
        self.fast_decay = 0.1  # Fixed rapid decay

        # Slow path gates (SSM-like for long-term memory)
        self.W_alpha = nn.Linear(d_model * 2, d_model, bias=bias)  # Preservation
        self.W_beta = nn.Linear(d_model * 2, d_model, bias=bias)   # Integration
        self.W_delta = nn.Linear(d_model * 2, d_model, bias=bias)  # Time scale

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_model, -1).clone())
        self.W_B = nn.Linear(d_model * 2, d_model * d_state, bias=bias)
        self.W_C = nn.Linear(d_model * 2, d_model * d_state, bias=bias)

        # Meta path (horizon selection)
        self.meta_update = nn.Linear(d_model + d_meta, d_meta, bias=bias)
        self.horizon_selector = nn.Linear(d_meta + d_model, d_model, bias=bias)

        # Output
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Linear(d_model * 2, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'A_log' in name or 'D' in name or 'norm' in name:
                continue
            elif param.dim() >= 2:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='linear')
                fan_in = param.shape[1]
                if 'W_B' in name or 'W_C' in name:
                    scale = 0.5 / (fan_in ** 0.5)
                else:
                    scale = min(1.0, 1.0 / (fan_in ** 0.25))
                with torch.no_grad():
                    param.mul_(scale)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        h_fast: torch.Tensor,
        h_slow: torch.Tensor,
        h_meta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x.shape[0]

        # Gate input combines fast state and input
        gate_input = torch.cat([h_fast, x], dim=-1)

        # === FAST PATH (GRU-style) ===
        z = torch.sigmoid(self.fast_z(gate_input))
        r = torch.sigmoid(self.fast_r(gate_input))
        h_candidate = torch.tanh(self.fast_h(torch.cat([r * h_fast, x], dim=-1)))
        h_fast_new = (1 - z) * h_fast * (1 - self.fast_decay) + z * h_candidate

        # === SLOW PATH (SSM-style) ===
        alpha = torch.sigmoid(self.W_alpha(gate_input))
        beta = torch.sigmoid(self.W_beta(gate_input))
        delta = F.softplus(self.W_delta(gate_input)) * 0.1

        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
        B = self.W_B(gate_input).view(batch, self.d_model, self.d_state)
        B_bar = delta.unsqueeze(-1) * B

        x_expanded = x.unsqueeze(-1)
        h_slow_preserved = alpha.unsqueeze(-1) * (A_bar * h_slow)
        h_slow_integrated = beta.unsqueeze(-1) * (B_bar * x_expanded)
        h_slow_new = h_slow_preserved + h_slow_integrated

        # === META PATH (Horizon Selection) ===
        # Update meta state with input context
        h_meta_new = torch.tanh(self.meta_update(torch.cat([x, h_meta], dim=-1)))

        # Compute horizon mixing coefficient (0=fast, 1=slow)
        tau = torch.sigmoid(self.horizon_selector(torch.cat([h_meta_new, x], dim=-1)))

        # === MIXED OUTPUT ===
        # Extract slow path output
        C = self.W_C(gate_input).view(batch, self.d_model, self.d_state)
        h_slow_out = (C * h_slow_new).sum(dim=-1)

        # Adaptive mixing based on task context
        h_mixed = (1 - tau) * h_fast_new + tau * h_slow_out

        # Gated output
        gamma = torch.sigmoid(self.gamma(gate_input))
        y = gamma * self.norm(h_mixed + self.D * x)

        return y, h_fast_new, h_slow_new, h_meta_new


class AnimaHierarchical(nn.Module):
    """
    AnimaHierarchical: Hierarchical temporal horizons with adaptive mixing.

    Key Innovation: Learns WHEN to use fast vs slow memory based on task context.
    - Fast path: O(d_model) immediate responses
    - Slow path: O(d_model * d_state) multi-scale memory
    - Meta path: O(d_meta) task context

    Addresses failures:
    - EvolvedV2's projectile (44%) via fast accumulation path
    - ISSM's XOR (0%) via clean threshold in fast path
    - Router's sequence (46%) via stable slow path
    """

    def __init__(
        self,
        sensory_dim: int = 8,
        d_model: int = 32,
        bottleneck_dim: int = 16,
        output_dim: int = 4,
        d_state: int = 16,
        d_meta: int = 8,
    ):
        super().__init__()

        self.sensory_dim = sensory_dim
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.d_state = d_state
        self.d_meta = d_meta

        # Type W (sensing)
        self.W_enc = nn.Linear(sensory_dim, d_model)
        self.W_from_W = nn.Linear(d_model, d_model, bias=False)
        self.W_from_I = nn.Linear(d_model, d_model, bias=False)
        self.W_from_A = nn.Linear(d_model, d_model, bias=False)
        self.W_gate = nn.Linear(d_model * 2, d_model)
        self.W_norm = nn.LayerNorm(d_model)

        # Type I (memory) - Hierarchical cell
        self.I_cell = HierarchicalTemporalCell(d_model, d_state, d_meta)
        self.I_from_W = nn.Linear(d_model, d_model, bias=False)
        self.I_from_I = nn.Linear(d_model, d_model, bias=False)
        self.I_from_A = nn.Linear(d_model, d_model, bias=False)

        # Abstraction bottleneck
        self.abstract = nn.Linear(d_model * 2, bottleneck_dim)
        self.expand = nn.Linear(bottleneck_dim, d_model)

        # Type A (action)
        self.A_from_W = nn.Linear(d_model, d_model, bias=False)
        self.A_from_I = nn.Linear(d_model, d_model, bias=False)
        self.A_from_A = nn.Linear(d_model, d_model, bias=False)
        self.A_gate = nn.Linear(d_model * 2, d_model)
        self.A_norm = nn.LayerNorm(d_model)

        self.phi = nn.Linear(d_model, output_dim)

        # States
        self.W_state: Optional[torch.Tensor] = None
        self.I_fast: Optional[torch.Tensor] = None
        self.I_slow: Optional[torch.Tensor] = None
        self.I_meta: Optional[torch.Tensor] = None
        self.A_state: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'norm' in name or 'A_log' in name or 'D' in name:
                continue
            elif param.dim() >= 2:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='linear')
                fan_in = param.shape[1]
                scale = 0.8 / math.sqrt(fan_in)
                with torch.no_grad():
                    param.mul_(scale)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None):
        if device is None:
            device = next(self.parameters()).device
        self.W_state = torch.zeros(batch_size, self.d_model, device=device)
        self.I_fast = torch.zeros(batch_size, self.d_model, device=device)
        self.I_slow = torch.zeros(batch_size, self.d_model, self.d_state, device=device)
        self.I_meta = torch.zeros(batch_size, self.d_meta, device=device)
        self.A_state = torch.zeros(batch_size, self.d_model, device=device)

    def forward(self, x: torch.Tensor, return_states: bool = False) -> Dict[str, torch.Tensor]:
        batch, seq_len, _ = x.shape
        device = x.device
        self.reset(batch, device)

        outputs = []
        for t in range(seq_len):
            obs = x[:, t]

            # W update
            W_enc = torch.tanh(self.W_enc(obs))
            W_coupled = (
                self.W_from_W(self.W_state) +
                self.W_from_I(self.I_fast) +
                self.W_from_A(self.A_state)
            )
            W_gate = torch.sigmoid(self.W_gate(torch.cat([W_enc, W_coupled], dim=-1)))
            W_new = self.W_norm(W_gate * W_enc + (1 - W_gate) * self.W_state)

            # I update (Hierarchical cell)
            I_input = (
                self.I_from_W(W_new) +
                self.I_from_I(self.I_fast) +
                self.I_from_A(self.A_state)
            )
            I_output, I_fast_new, I_slow_new, I_meta_new = self.I_cell(
                I_input, self.I_fast, self.I_slow, self.I_meta
            )

            # Abstraction
            abstract = torch.tanh(self.abstract(torch.cat([W_new, I_output], dim=-1)))
            expanded = torch.tanh(self.expand(abstract))

            # A update
            A_coupled = (
                self.A_from_W(W_new) +
                self.A_from_I(I_output) +
                self.A_from_A(self.A_state)
            )
            A_input = A_coupled + expanded
            A_gate = torch.sigmoid(self.A_gate(torch.cat([A_input, self.A_state], dim=-1)))
            A_new = self.A_norm(A_gate * A_input + (1 - A_gate) * self.A_state)

            self.W_state = W_new
            self.I_fast = I_fast_new
            self.I_slow = I_slow_new
            self.I_meta = I_meta_new
            self.A_state = A_new

            outputs.append(self.phi(A_new))

        return {'output': torch.stack(outputs, dim=1)}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
