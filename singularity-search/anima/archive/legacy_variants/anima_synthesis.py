"""
ANIMA-Synthesis: Unified Architecture Combining Zero + One Strengths
====================================================================

DESIGN RATIONALE:
-----------------
Benchmark analysis reveals complementary strengths:
- ANIMA-Zero: 100% physics (GRU memory for temporal coherence)
- ANIMA-One: 98% reasoning (bottleneck compression for abstraction)

ANIMA-Synthesis unifies both via:
1. Dual-pathway processing (temporal + compressed)
2. Adaptive gating to blend pathways based on input
3. Minimal additional parameters (no HTC complexity)

TARGET: 99%+ overall by combining best of both architectures.

THEORETICAL BASIS:
------------------
V(N) constraint: Full type coupling (W <-> I <-> A)
V(T) constraint: GRU ensures temporal coherence
Φ constraint: Bottleneck forces information integration

The synthesis adds: Adaptive pathway selection based on input dynamics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ANIMASynthesis(nn.Module):
    """
    ANIMA-Synthesis: Dual-pathway architecture with adaptive gating.

    Combines:
    - Zero's GRU memory (temporal/physics excellence)
    - One's bottleneck compression (reasoning/abstraction excellence)
    - Adaptive pathway blending (input-dependent)
    """

    def __init__(
        self,
        sensory_dim: int = 8,
        d_model: int = 24,
        bottleneck_dim: int = 8,
        output_dim: int = 4,
    ):
        super().__init__()

        self.sensory_dim = sensory_dim
        self.d = d_model
        self.b = bottleneck_dim
        self.output_dim = output_dim

        # === INPUT ENCODING ===
        self.input_enc = nn.Linear(sensory_dim, d_model)

        # === PATHWAY 1: ZERO-STYLE (Temporal/Physics) ===
        # Full GRU for temporal coherence
        self.gru_z = nn.Linear(d_model * 2, d_model)
        self.gru_r = nn.Linear(d_model * 2, d_model)
        self.gru_h = nn.Linear(d_model * 2, d_model)

        # === PATHWAY 2: ONE-STYLE (Compressed/Reasoning) ===
        # Bottleneck for information integration
        self.compress = nn.Linear(d_model, bottleneck_dim)
        self.expand = nn.Linear(bottleneck_dim, d_model)

        # === ADAPTIVE GATING ===
        # Learn when to use temporal vs compressed pathway
        self.pathway_gate = nn.Linear(d_model * 2, 2)  # 2 pathways

        # === TYPE COUPLING (W-I-A) ===
        # Sensing (W)
        self.W_combine = nn.Linear(d_model * 2, d_model)

        # Memory (I) - receives from both pathways
        self.I_from_temporal = nn.Linear(d_model, d_model, bias=False)
        self.I_from_compressed = nn.Linear(d_model, d_model, bias=False)
        self.I_gate = nn.Linear(d_model * 2, d_model)

        # Action (A)
        self.A_from_W = nn.Linear(d_model, d_model, bias=False)
        self.A_from_I = nn.Linear(d_model, d_model, bias=False)
        self.A_gate = nn.Linear(d_model * 2, d_model)

        # === OUTPUT ===
        self.phi = nn.Linear(d_model, output_dim)

        # === STATE ===
        self.W = None  # Sensing state
        self.I = None  # Memory state (GRU hidden)
        self.A = None  # Action state

        self._init_weights()

    def _init_weights(self):
        """Initialize for stability at criticality (λ ~ 0⁺)."""
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.orthogonal_(param)
                with torch.no_grad():
                    param.mul_(0.95 / max(param.abs().max(), 1e-6))
            elif 'bias' in name:
                nn.init.zeros_(param)

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset internal state."""
        if device is None:
            device = next(self.parameters()).device
        self.W = torch.zeros(batch_size, self.d, device=device)
        self.I = torch.zeros(batch_size, self.d, device=device)
        self.A = torch.zeros(batch_size, self.d, device=device)

    def step(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single step of ANIMA-Synthesis.

        Args:
            obs: [batch, sensory_dim] observation

        Returns:
            Dict with 'action' tensor
        """
        if self.W is None:
            self.reset(obs.shape[0], obs.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # === ENCODE INPUT ===
        x = torch.tanh(self.input_enc(obs))

        # === PATHWAY 1: TEMPORAL (Zero-style GRU) ===
        gru_input = torch.cat([x, self.I], dim=-1)
        z = torch.sigmoid(self.gru_z(gru_input))
        r = torch.sigmoid(self.gru_r(gru_input))
        h_candidate = torch.tanh(self.gru_h(torch.cat([x, r * self.I], dim=-1)))
        temporal_out = (1 - z) * self.I + z * h_candidate

        # === PATHWAY 2: COMPRESSED (One-style bottleneck) ===
        compressed = torch.tanh(self.compress(x))
        compressed_out = torch.tanh(self.expand(compressed))

        # === ADAPTIVE GATING ===
        # Learn to blend pathways based on input + memory
        gate_input = torch.cat([x, self.I], dim=-1)
        pathway_weights = torch.softmax(self.pathway_gate(gate_input), dim=-1)

        # Blend pathways
        blended = (
            pathway_weights[:, 0:1] * temporal_out +
            pathway_weights[:, 1:2] * compressed_out
        )

        # === TYPE COUPLING ===
        # W (Sensing): Combine input with previous action
        W_new = torch.tanh(self.W_combine(torch.cat([x, self.A], dim=-1)))

        # I (Memory): Integrate both pathways
        I_temporal = self.I_from_temporal(temporal_out)
        I_compressed = self.I_from_compressed(compressed_out)
        I_combined = I_temporal + I_compressed
        I_gate = torch.sigmoid(self.I_gate(torch.cat([W_new, blended], dim=-1)))
        I_new = torch.tanh(I_combined) * I_gate + (1 - I_gate) * self.I

        # A (Action): Full type coupling
        A_input = self.A_from_W(W_new) + self.A_from_I(I_new)
        A_gate = torch.sigmoid(self.A_gate(torch.cat([W_new, I_new], dim=-1)))
        A_new = torch.tanh(A_input) * A_gate + 0.1 * torch.tanh(A_input)

        # Update state
        self.W = W_new
        self.I = I_new
        self.A = A_new

        return {
            'action': self.phi(A_new),
            'pathway_weights': pathway_weights,  # For analysis
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batch forward for sequence processing.

        Args:
            x: [batch, seq_len, sensory_dim]

        Returns:
            [batch, seq_len, output_dim]
        """
        batch, seq_len, _ = x.shape
        self.reset(batch, x.device)

        outputs = []
        for t in range(seq_len):
            result = self.step(x[:, t])
            outputs.append(result['action'])

        return torch.stack(outputs, dim=1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ANIMASynthesisExact(ANIMASynthesis):
    """
    ANIMA-Synthesis scaled to exact target parameter count.
    Used for fair benchmarking.
    """

    def __init__(
        self,
        target_params: int,
        sensory_dim: int = 8,
        output_dim: int = 4,
    ):
        # Find optimal dimensions
        d, b = self._find_dimensions(target_params, sensory_dim, output_dim)

        super().__init__(
            sensory_dim=sensory_dim,
            d_model=d,
            bottleneck_dim=b,
            output_dim=output_dim,
        )

        # Add padding for exact match
        current = self.count_parameters()
        diff = target_params - current
        if diff > 0:
            self.register_parameter('_pad', nn.Parameter(torch.zeros(diff)))

    def _find_dimensions(
        self,
        target: int,
        s_dim: int,
        o_dim: int
    ) -> Tuple[int, int]:
        """Find d_model and bottleneck_dim for target param count."""
        best_config = (16, 8)
        best_diff = float('inf')

        for d in range(8, 64):
            for b in range(4, d):
                # Create temp model to count actual params
                class TempModel(nn.Module):
                    def __init__(self, dim, bdim):
                        super().__init__()
                        self.input_enc = nn.Linear(s_dim, dim)
                        self.gru_z = nn.Linear(dim * 2, dim)
                        self.gru_r = nn.Linear(dim * 2, dim)
                        self.gru_h = nn.Linear(dim * 2, dim)
                        self.compress = nn.Linear(dim, bdim)
                        self.expand = nn.Linear(bdim, dim)
                        self.pathway_gate = nn.Linear(dim * 2, 2)
                        self.W_combine = nn.Linear(dim * 2, dim)
                        self.I_from_temporal = nn.Linear(dim, dim, bias=False)
                        self.I_from_compressed = nn.Linear(dim, dim, bias=False)
                        self.I_gate = nn.Linear(dim * 2, dim)
                        self.A_from_W = nn.Linear(dim, dim, bias=False)
                        self.A_from_I = nn.Linear(dim, dim, bias=False)
                        self.A_gate = nn.Linear(dim * 2, dim)
                        self.phi = nn.Linear(dim, o_dim)

                temp = TempModel(d, b)
                params = sum(p.numel() for p in temp.parameters())
                del temp

                diff = abs(params - target)
                if diff < best_diff:
                    best_diff = diff
                    best_config = (d, b)

        return best_config


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_anima_synthesis(
    target_params: Optional[int] = None,
    sensory_dim: int = 8,
    d_model: int = 24,
    bottleneck_dim: int = 8,
    output_dim: int = 4,
) -> ANIMASynthesis:
    """
    Factory function for ANIMA-Synthesis.

    Args:
        target_params: If specified, scale to exact param count
        sensory_dim: Input dimension
        d_model: Hidden dimension (ignored if target_params set)
        bottleneck_dim: Bottleneck dimension (ignored if target_params set)
        output_dim: Output dimension

    Returns:
        ANIMASynthesis or ANIMASynthesisExact instance
    """
    if target_params is not None:
        return ANIMASynthesisExact(
            target_params=target_params,
            sensory_dim=sensory_dim,
            output_dim=output_dim,
        )
    else:
        return ANIMASynthesis(
            sensory_dim=sensory_dim,
            d_model=d_model,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
        )


if __name__ == "__main__":
    # Quick test
    model = ANIMASynthesis()
    print(f"ANIMA-Synthesis parameters: {model.count_parameters():,}")

    # Test forward
    x = torch.randn(2, 8, 8)  # batch=2, seq=8, sensory=8
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    # Test exact param matching
    target = 8964
    exact = ANIMASynthesisExact(target)
    print(f"Exact model ({target} target): {exact.count_parameters():,} params")
