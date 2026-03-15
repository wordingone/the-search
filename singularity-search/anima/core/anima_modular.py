"""
AnimaModular: Separate Circuits for Logic vs Memory

Core Insight: Superposition-specialization trade-off is TASK-DEPENDENT.
- Logic tasks (conditional, XOR): Need dedicated circuits (LOW superposition)
- Memory tasks (delay, copy): Need efficient packing (HIGH superposition)

Current architectures force a single shared state to handle both,
resulting in inevitable trade-offs.

Causal Design:
- ATR has highest superposition (0.82) but worst logic (31.7%)
- EvolvedV2 has lowest superposition (0.57) but best logic (96%)
- Correlation: r = -0.68 between superposition and logic performance

Solution: Two separate modules with task-dependent routing.
- Logic Module: Low superposition, dedicated gates, sharp decisions
- Memory Module: High superposition, SSM dynamics, efficient packing
- Fusion Layer: Soft routing based on input characteristics

Target Improvements:
- Best-of-both: 96% logic (like V2) AND 84% memory (like Anima)
- No catastrophic forgetting between task types
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LogicModule(nn.Module):
    """
    Logic Module: Low superposition, dedicated circuits.

    Optimized for:
    - Threshold decisions (conditional, XOR)
    - Binary classification
    - Crisp gate activations

    Key Design:
    - No state expansion (flat [d_model])
    - Coupled gates (alpha + (1-alpha) = 1) for clean decisions
    - Pre-LayerNorm for sharp activations
    """

    def __init__(self, d_model: int, bias: bool = True):
        super().__init__()
        self.d_model = d_model

        # Pre-norm for sharp gate activations (like base Anima)
        self.pre_norm = nn.LayerNorm(d_model)

        # Coupled gates (GRU-like)
        self.W_z = nn.Linear(d_model * 2, d_model, bias=bias)  # Update gate
        self.W_r = nn.Linear(d_model * 2, d_model, bias=bias)  # Reset gate
        self.W_h = nn.Linear(d_model * 2, d_model, bias=bias)  # Candidate

        # Output
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'norm' in name:
                continue
            elif param.dim() >= 2:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='linear')
                fan_in = param.shape[1]
                with torch.no_grad():
                    param.mul_(0.8 / math.sqrt(fan_in))
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm input for sharp activations
        x_norm = self.pre_norm(x)
        gate_input = torch.cat([h, x_norm], dim=-1)

        # GRU-style coupled update
        z = torch.sigmoid(self.W_z(gate_input))
        r = torch.sigmoid(self.W_r(gate_input))
        h_candidate = torch.tanh(self.W_h(torch.cat([r * h, x_norm], dim=-1)))

        # Coupled: z controls both preservation and integration
        h_new = (1 - z) * h + z * h_candidate

        return self.norm(h_new), h_new


class MemoryModule(nn.Module):
    """
    Memory Module: High superposition, SSM dynamics.

    Optimized for:
    - Long-term retention (delay, sequence)
    - Pattern copying
    - Efficient feature packing

    Key Design:
    - Structured state [d_model, d_state] for multi-scale memory
    - Independent gates for accumulation capability
    - State-dependent discretization for intentional updates
    """

    def __init__(self, d_model: int, d_state: int = 16, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Independent gates (V1-style)
        self.W_alpha = nn.Linear(d_model * 2, d_model, bias=bias)
        self.W_beta = nn.Linear(d_model * 2, d_model, bias=bias)
        self.W_delta = nn.Linear(d_model * 2, d_model, bias=bias)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_model, -1).clone())
        self.W_B = nn.Linear(d_model * 2, d_model * d_state, bias=bias)
        self.W_C = nn.Linear(d_model * 2, d_model * d_state, bias=bias)
        self.D = nn.Parameter(torch.ones(d_model))

        # Output
        self.norm = nn.LayerNorm(d_model)

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

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        h_flat = h.mean(dim=-1)
        gate_input = torch.cat([h_flat, x], dim=-1)

        # Independent gates
        alpha = torch.sigmoid(self.W_alpha(gate_input))
        beta = torch.sigmoid(self.W_beta(gate_input))
        delta = F.softplus(self.W_delta(gate_input)) * 0.1

        # SSM dynamics
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
        B = self.W_B(gate_input).view(batch, self.d_model, self.d_state)
        B_bar = delta.unsqueeze(-1) * B

        x_expanded = x.unsqueeze(-1)
        h_preserved = alpha.unsqueeze(-1) * (A_bar * h)
        h_integrated = beta.unsqueeze(-1) * (B_bar * x_expanded)
        h_new = h_preserved + h_integrated

        # Output
        C = self.W_C(gate_input).view(batch, self.d_model, self.d_state)
        y = (C * h_new).sum(dim=-1)
        y = self.norm(y + self.D * x)

        return y, h_new


class ModularFusion(nn.Module):
    """
    Fusion Layer: Task-dependent routing between Logic and Memory modules.

    Uses SOFT routing (unlike Router's hard Gumbel-Softmax):
    - Preserves gradients through both paths
    - Learns smooth task-dependent mixing
    - Avoids Router's induction destruction

    Key Design:
    - Input characteristics determine routing
    - Entropy-based weighting (high variance -> logic, low variance -> memory)
    - Learned refinement on top of heuristic
    """

    def __init__(self, d_model: int, bias: bool = True):
        super().__init__()
        self.d_model = d_model

        # Routing network
        self.route_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model, bias=bias),
            nn.Tanh(),
            nn.Linear(d_model, 2, bias=bias),
        )

        # Output mixing
        self.W_out = nn.Linear(d_model, d_model, bias=bias)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() >= 2:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='linear')
                fan_in = param.shape[1]
                with torch.no_grad():
                    param.mul_(0.5 / (fan_in ** 0.5))

    def forward(
        self,
        x: torch.Tensor,
        y_logic: torch.Tensor,
        y_memory: torch.Tensor
    ) -> torch.Tensor:
        # Compute input characteristics for routing
        # High magnitude variance suggests threshold task (logic)
        # Smooth patterns suggest memory task

        route_input = torch.cat([x, y_logic, y_memory], dim=-1)
        route_weights = F.softmax(self.route_net(route_input), dim=-1)

        # Soft mixing
        p_logic = route_weights[:, 0:1]
        p_memory = route_weights[:, 1:2]

        y_mixed = p_logic * y_logic + p_memory * y_memory
        y_out = self.norm(self.W_out(y_mixed))

        return y_out


class ModularTemporalCell(nn.Module):
    """
    Modular Temporal Cell combining Logic and Memory modules.

    Maintains separate states:
    - h_logic: [batch, d_model] - Flat state for logic
    - h_memory: [batch, d_model, d_state] - Structured state for memory

    The fusion layer decides how to combine outputs.
    """

    def __init__(self, d_model: int, d_state: int = 16, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.logic = LogicModule(d_model, bias)
        self.memory = MemoryModule(d_model, d_state, bias)
        self.fusion = ModularFusion(d_model, bias)

        # Output gate
        self.W_gamma = nn.Linear(d_model * 2, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        h_logic: torch.Tensor,
        h_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Process through both modules
        y_logic, h_logic_new = self.logic(x, h_logic)
        y_memory, h_memory_new = self.memory(x, h_memory)

        # Fuse outputs
        y_fused = self.fusion(x, y_logic, y_memory)

        # Output gate
        gamma = torch.sigmoid(self.W_gamma(torch.cat([y_fused, x], dim=-1)))
        y = gamma * y_fused

        return y, h_logic_new, h_memory_new


class AnimaModular(nn.Module):
    """
    AnimaModular: Separate circuits for logic vs memory.

    Key Innovation: Explicit specialization without shared bottleneck.
    - Logic Module: Low superposition, coupled gates, sharp decisions
    - Memory Module: High superposition, SSM dynamics, efficient packing
    - Fusion Layer: Soft task-dependent routing

    Addresses the superposition-specialization trade-off:
    - ATR (0.82 superposition, 31.7% logic) vs EvolvedV2 (0.57, 96%)
    - Correlation r=-0.68 shows fundamental conflict
    - Modular design eliminates this trade-off

    Expected: Best of both worlds
    - Logic: ~96% (matching V2's coupled gates)
    - Memory: ~84% (matching Anima's physics)
    """

    def __init__(
        self,
        sensory_dim: int = 8,
        d_model: int = 32,
        bottleneck_dim: int = 16,
        output_dim: int = 4,
        d_state: int = 16,
    ):
        super().__init__()

        self.sensory_dim = sensory_dim
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.d_state = d_state

        # Type W (sensing)
        self.W_enc = nn.Linear(sensory_dim, d_model)
        self.W_from_W = nn.Linear(d_model, d_model, bias=False)
        self.W_from_I = nn.Linear(d_model, d_model, bias=False)
        self.W_from_A = nn.Linear(d_model, d_model, bias=False)
        self.W_gate = nn.Linear(d_model * 2, d_model)
        self.W_norm = nn.LayerNorm(d_model)

        # Type I (memory) - Modular cell
        self.I_cell = ModularTemporalCell(d_model, d_state)
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
        self.I_logic: Optional[torch.Tensor] = None
        self.I_memory: Optional[torch.Tensor] = None
        self.A_state: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'norm' in name or 'A_log' in name or 'D' in name or 'I_cell' in name:
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
        self.I_logic = torch.zeros(batch_size, self.d_model, device=device)
        self.I_memory = torch.zeros(batch_size, self.d_model, self.d_state, device=device)
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
                self.W_from_I(self.I_logic) +
                self.W_from_A(self.A_state)
            )
            W_gate = torch.sigmoid(self.W_gate(torch.cat([W_enc, W_coupled], dim=-1)))
            W_new = self.W_norm(W_gate * W_enc + (1 - W_gate) * self.W_state)

            # I update (Modular cell)
            I_input = (
                self.I_from_W(W_new) +
                self.I_from_I(self.I_logic) +
                self.I_from_A(self.A_state)
            )
            I_output, I_logic_new, I_memory_new = self.I_cell(
                I_input, self.I_logic, self.I_memory
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
            self.I_logic = I_logic_new
            self.I_memory = I_memory_new
            self.A_state = A_new

            outputs.append(self.phi(A_new))

        return {'output': torch.stack(outputs, dim=1)}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
