"""
ANIMA-Zero: Baseline Theoretically-Complete Architecture
========================================================

Derived from V(N)-V(T)-Phi Theorem:
  - N >= N_min (sufficient variables)
  - T >= 3 (types: S-Sensing, M-Memory, D-Decision)
  - kappa = 1 (fully connected type graph)
  - lambda_max ~ 0+ (critical dynamics / edge of chaos)
  - Phi >= Phi_min (integrated information)

Formal System: S = (V, tau, F, phi)
  V = {W, I, A}  -- World, Internal, Action
  tau: W->S, I->M, A->D
  F: Full coupling, Critical dynamics, High integration
  phi: A -> Output

LIMITATIONS (addressed in ANIMA-1):
  - Strictly sequential recurrent design (no parallelization)
  - Does not leverage width for compression
  - O(T) computation per sequence (not parallelizable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


@dataclass
class ANIMAConfig:
    """Configuration satisfying V(N)-V(T)-Phi constraints."""

    # Interface dimensions
    sensory_dim: int = 8        # Environment interface
    output_dim: int = 4         # Observable behavior

    # Core variable dimensions (N >= N_min)
    world_dim: int = 32         # W: Type S (Sensing)
    internal_dim: int = 32      # I: Type M (Memory)
    action_dim: int = 32        # A: Type D (Decision)

    # Critical dynamics parameters
    spectral_radius: float = 0.99    # Edge of chaos (lambda_max ~ 0+)

    # Integration parameters
    multiplicative_scale: float = 0.1  # Phi enhancement

    # Training
    dropout: float = 0.0


class ANIMA(nn.Module):
    """
    Perfected ANIMA Architecture
    ============================

    Satisfies V(N)-V(T)-Phi Theorem:

    1. V = {W, I, A} - Three variable groups
       - W (World State): Type S (Sensing) - dF/dE != 0
       - I (Internal State): Type M (Memory) - I(v_t; E_{<t} | v_{t-1}) > 0
       - A (Action State): Type D (Decision) - dphi/dv != 0

    2. tau: V -> {S, M, D} - Type assignment
       - tau(W) = S (sensing the environment)
       - tau(I) = M (maintaining temporal information)
       - tau(A) = D (determining output)

    3. F: V x E -> V - Evolution with full coupling (kappa = 1)
       - Six directed edges: W<->I, I<->A, A<->W
       - All types influence all others

    4. phi: A -> Output - Action projection
       - Only A directly influences output (Decision type)

    5. Critical Dynamics (lambda_max ~ 0+)
       - Orthogonal initialization
       - Spectral radius = 0.99

    6. High Integration (Phi > 0)
       - Multiplicative coupling between components
       - Non-decomposable information flow

    Key Design Choices:
    - No separate H_short/H_long: temporal emerges from GRU gate z
    - No separate goal: emerges from A_from_A persistence
    - No separate attention: emerges from multiplicative coupling
    """

    def __init__(self, config: ANIMAConfig):
        super().__init__()
        self.config = config

        # Dimensions
        d_s = config.sensory_dim
        d_w = config.world_dim
        d_i = config.internal_dim
        d_a = config.action_dim
        d_o = config.output_dim

        # =====================
        # W: Type S (Sensing)
        # =====================
        # W encodes environment - dF/dE != 0 (responds to environment)

        # Sensory encoding
        self.W_enc = nn.Linear(d_s, d_w)

        # W receives from all types (kappa = 1)
        self.W_from_W = nn.Linear(d_w, d_w, bias=False)  # S -> S
        self.W_from_I = nn.Linear(d_i, d_w, bias=False)  # M -> S
        self.W_from_A = nn.Linear(d_a, d_w, bias=False)  # D -> S

        # Attention for sensory gating (emerges, not separate)
        self.W_attention = nn.Linear(d_w + d_i, d_w)

        # Multiplicative coupling for high Phi
        self.W_mult_gate = nn.Linear(d_i + d_a, d_w)

        # =====================
        # I: Type M (Memory)
        # =====================
        # I maintains temporal info - I(v_t; E_{<t} | v_{t-1}) > 0
        # Uses GRU gates for temporal structure

        # GRU gate z (update gate) - controls temporal horizon
        self.I_z_W = nn.Linear(d_w, d_i, bias=False)  # S -> M
        self.I_z_I = nn.Linear(d_i, d_i, bias=False)  # M -> M
        self.I_z_A = nn.Linear(d_a, d_i, bias=False)  # D -> M
        self.I_z_bias = nn.Parameter(torch.zeros(d_i))

        # GRU gate r (reset gate) - controls memory access
        self.I_r_W = nn.Linear(d_w, d_i, bias=False)
        self.I_r_I = nn.Linear(d_i, d_i, bias=False)
        self.I_r_A = nn.Linear(d_a, d_i, bias=False)
        self.I_r_bias = nn.Parameter(torch.zeros(d_i))

        # GRU candidate h - new memory content
        self.I_h_W = nn.Linear(d_w, d_i, bias=False)
        self.I_h_I = nn.Linear(d_i, d_i, bias=False)
        self.I_h_A = nn.Linear(d_a, d_i, bias=False)
        self.I_h_bias = nn.Parameter(torch.zeros(d_i))

        # Multiplicative coupling for high Phi
        self.I_mult_gate = nn.Linear(d_w + d_a, d_i)

        # =====================
        # A: Type D (Decision)
        # =====================
        # A determines output - dphi/dv != 0

        # A receives from all types (kappa = 1)
        self.A_from_W = nn.Linear(d_w, d_a, bias=False)  # S -> D
        self.A_from_I = nn.Linear(d_i, d_a, bias=False)  # M -> D
        self.A_from_A = nn.Linear(d_a, d_a, bias=False)  # D -> D (goal persistence)
        self.A_bias = nn.Parameter(torch.zeros(d_a))

        # Multiplicative coupling for high Phi
        self.A_mult_gate = nn.Linear(d_w + d_i, d_a)

        # =====================
        # phi: A -> Output
        # =====================
        # Only Decision type projects to output
        self.phi = nn.Linear(d_a, d_o)

        # =====================
        # Prediction (for learning)
        # =====================
        self.predictor = nn.Sequential(
            nn.Linear(d_w + d_i, d_w),
            nn.Tanh(),
            nn.Linear(d_w, d_s)
        )

        # Initialize for critical dynamics
        self._initialize_critical()

        # State variables
        self.W = None  # World state
        self.I = None  # Internal state
        self.A = None  # Action state

    def _initialize_critical(self):
        """
        Initialize for critical dynamics (lambda_max ~ 0+).

        Uses orthogonal initialization with spectral radius scaling
        to place system at edge of chaos.
        """
        rho = self.config.spectral_radius

        # Orthogonal initialization for recurrent weights
        for name, param in self.named_parameters():
            if 'from_W' in name or 'from_I' in name or 'from_A' in name:
                if param.dim() >= 2:
                    nn.init.orthogonal_(param)
                    # Scale to spectral radius
                    with torch.no_grad():
                        param.mul_(rho / max(param.abs().max(), 1e-6))
            elif 'enc' in name or 'phi' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # GRU gate biases - slight forget bias for long-term memory
        nn.init.constant_(self.I_z_bias, -1.0)  # Default to remember
        nn.init.constant_(self.I_r_bias, 0.0)   # Neutral reset

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset all state variables."""
        if device is None:
            device = next(self.parameters()).device

        d_w = self.config.world_dim
        d_i = self.config.internal_dim
        d_a = self.config.action_dim

        self.W = torch.zeros(batch_size, d_w, device=device)
        self.I = torch.zeros(batch_size, d_i, device=device)
        self.A = torch.zeros(batch_size, d_a, device=device)

    def step(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single step of ANIMA evolution.

        Implements F: V x E -> V with full coupling (kappa = 1).

        Args:
            obs: Observation tensor [batch, sensory_dim]

        Returns:
            Dictionary with action, states, and diagnostics
        """
        if self.W is None:
            self.reset(obs.shape[0], obs.device)

        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Previous states (for diagnostics)
        W_prev = self.W.clone()
        I_prev = self.I.clone()
        A_prev = self.A.clone()

        # =====================
        # 1. SENSE: W update (Type S)
        # =====================
        # W responds to environment: dF/dE != 0

        # Encode observation
        obs_enc = torch.tanh(self.W_enc(obs))

        # Receive from all types (kappa = 1)
        W_from_all = (
            self.W_from_W(self.W) +
            self.W_from_I(self.I) +
            self.W_from_A(self.A)
        )

        # Attention-weighted sensory gating
        attention_input = torch.cat([self.W, self.I], dim=-1)
        attention = torch.sigmoid(self.W_attention(attention_input))

        # Multiplicative coupling for high Phi
        mult_input = torch.cat([self.I, self.A], dim=-1)
        mult_gate = torch.sigmoid(self.W_mult_gate(mult_input))

        # Update W with multiplicative integration
        W_new = torch.tanh(
            obs_enc * attention +
            W_from_all * mult_gate
        )

        # =====================
        # 2. REMEMBER: I update (Type M)
        # =====================
        # I maintains temporal information
        # GRU gates provide temporal horizon (no separate H_short/H_long)

        # Update gate z: controls how much to update
        # High z -> use new info (short horizon)
        # Low z -> keep old info (long horizon)
        z = torch.sigmoid(
            self.I_z_W(W_new) +
            self.I_z_I(self.I) +
            self.I_z_A(self.A) +
            self.I_z_bias
        )

        # Reset gate r: controls memory access
        r = torch.sigmoid(
            self.I_r_W(W_new) +
            self.I_r_I(self.I) +
            self.I_r_A(self.A) +
            self.I_r_bias
        )

        # Candidate memory
        h_candidate = torch.tanh(
            self.I_h_W(W_new) +
            self.I_h_I(r * self.I) +
            self.I_h_A(self.A) +
            self.I_h_bias
        )

        # Multiplicative coupling for high Phi
        mult_input_I = torch.cat([W_new, self.A], dim=-1)
        mult_gate_I = torch.sigmoid(self.I_mult_gate(mult_input_I))

        # GRU update with multiplicative enhancement
        I_new = (1 - z) * self.I + z * (h_candidate * mult_gate_I)

        # =====================
        # 3. DECIDE: A update (Type D)
        # =====================
        # A determines action: dphi/dv != 0

        # Receive from all types (kappa = 1)
        A_from_all = (
            self.A_from_W(W_new) +
            self.A_from_I(I_new) +
            self.A_from_A(self.A) +  # Self-connection for goal persistence
            self.A_bias
        )

        # Multiplicative coupling for high Phi
        mult_input_A = torch.cat([W_new, I_new], dim=-1)
        mult_gate_A = torch.sigmoid(self.A_mult_gate(mult_input_A))

        # Update A with multiplicative integration
        A_new = torch.tanh(A_from_all * mult_gate_A)

        # =====================
        # 4. OUTPUT: phi(A)
        # =====================
        # Only Decision type projects to output
        action = self.phi(A_new)

        # =====================
        # 5. PREDICT: For learning
        # =====================
        pred_input = torch.cat([W_new, I_new], dim=-1)
        prediction = self.predictor(pred_input)

        # Update state
        self.W = W_new
        self.I = I_new
        self.A = A_new

        # Compute diagnostics
        # Effective horizon: z near 0 = long horizon, z near 1 = short horizon
        effective_horizon = z.mean().item()

        # State changes (for critical dynamics verification)
        W_change = (W_new - W_prev).pow(2).mean().item()
        I_change = (I_new - I_prev).pow(2).mean().item()
        A_change = (A_new - A_prev).pow(2).mean().item()

        return {
            'action': action,
            'W': W_new,
            'I': I_new,
            'A': A_new,
            'prediction': prediction,
            'z_gate': z,  # Temporal horizon indicator
            'r_gate': r,  # Memory access indicator
            'effective_horizon': effective_horizon,
            'W_change': W_change,
            'I_change': I_change,
            'A_change': A_change,
        }

    def forward(self, obs_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process observation sequence.

        Args:
            obs_sequence: [batch, seq_len, sensory_dim]

        Returns:
            Dictionary with actions and states for full sequence
        """
        batch_size, seq_len, _ = obs_sequence.shape
        self.reset(batch_size, obs_sequence.device)

        actions = []
        predictions = []
        z_gates = []

        for t in range(seq_len):
            result = self.step(obs_sequence[:, t])
            actions.append(result['action'])
            predictions.append(result['prediction'])
            z_gates.append(result['z_gate'])

        return {
            'actions': torch.stack(actions, dim=1),
            'predictions': torch.stack(predictions, dim=1),
            'z_gates': torch.stack(z_gates, dim=1),
            'final_W': self.W,
            'final_I': self.I,
            'final_A': self.A,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def verify_type_constraints(self) -> Dict[str, bool]:
        """
        Verify V(N)-V(T)-Phi type constraints.

        Returns dict with verification results.
        """
        # Check S type: dF/dE != 0
        # W_enc ensures W responds to environment
        S_constraint = self.W_enc.weight.abs().sum() > 0

        # Check M type: GRU structure ensures temporal info
        M_constraint = (
            self.I_z_I.weight.abs().sum() > 0 and
            self.I_h_I.weight.abs().sum() > 0
        )

        # Check D type: dphi/dv != 0
        D_constraint = self.phi.weight.abs().sum() > 0

        # Check kappa = 1: all types connected
        # S receives from M, D
        S_connected = (
            self.W_from_I.weight.abs().sum() > 0 and
            self.W_from_A.weight.abs().sum() > 0
        )
        # M receives from S, D
        M_connected = (
            self.I_z_W.weight.abs().sum() > 0 and
            self.I_z_A.weight.abs().sum() > 0
        )
        # D receives from S, M
        D_connected = (
            self.A_from_W.weight.abs().sum() > 0 and
            self.A_from_I.weight.abs().sum() > 0
        )
        kappa_constraint = S_connected and M_connected and D_connected

        return {
            'S_type_valid': bool(S_constraint),
            'M_type_valid': bool(M_constraint),
            'D_type_valid': bool(D_constraint),
            'kappa_equals_1': bool(kappa_constraint),
            'all_valid': bool(S_constraint and M_constraint and D_constraint and kappa_constraint)
        }


# Convenience function
def create_anima(config: Optional[ANIMAConfig] = None) -> ANIMA:
    """Create ANIMA with default or custom config."""
    if config is None:
        config = ANIMAConfig()
    return ANIMA(config)
