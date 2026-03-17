#!/usr/bin/env python3
"""
The Self-Referential Codebook — Phase 2, Step 1

30 lines where the state encodes the operations and the operations
read from the state.

The system chains through its own codebook:
  x → match → w0 → re-match → w1 → action
The chain IS the computation. The codebook IS the program.
Different geometries → different chains → different computations.

Step 181 (iterated k-NN, 100% on Rule 110) was the hint: feed the
output back as input. That was static codebook, no self-modification.
This adds: attract during chain traversal. The chain modifies what
it traverses. The codebook rewrites itself by being read.

R1: No external objective (dot products + argmax, no loss)
R2: Update (attract) IS the computation (matching)
R3: V is the only state. All of V modified by attract + spawn.
R4: Each chain step tests: is winner consistent with query?
    High sim → small update (stable). Low sim → large update (learn).
R5: The interpreter (match-chain-attract) is frozen.
R6: Remove match → blind. Remove chain → no self-reference.
    Remove attract → no learning. Remove spawn → no growth.

It will get 0% on everything. That is correct.
The goal is a point inside six walls, not accuracy.
"""

import torch
import torch.nn.functional as F


class SelfRef:
    """The substrate. 30 lines. State = V. V encodes the program."""

    def __init__(self, d, device='cuda'):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.device = device

    def step(self, x, n_actions):
        """
        One step. The state reads itself, modifies itself, returns an action.

        x: (d,) observation vector.
        n_actions: int, number of possible actions.
        returns: int, the action.
        """
        x_n = F.normalize(x.to(self.device).float(), dim=0)

        # Bootstrap: need at least 2 entries for self-reference
        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])
            return self.V.shape[0] - 1

        # ── MATCH: observation → pattern ──
        sims = self.V @ x_n
        w0 = sims.argmax().item()

        # ── CHAIN: pattern → instruction (self-reference) ──
        # The winner is re-matched against the codebook.
        # The codebook interprets its own entry.
        ref = self.V @ self.V[w0]
        ref[w0] = -float('inf')                    # no self-loop
        w1 = ref.argmax().item()

        # ── ACTION: instruction index encodes behavior ──
        action = w1 % n_actions

        # ── UPDATE: attract during traversal (R2: update IS computation) ──
        # Pattern learns from observation
        lr0 = (1.0 - sims[w0].clamp(0, 1)).item()
        self.V[w0] = F.normalize(
            self.V[w0] + lr0 * (x_n - self.V[w0]), dim=0)

        # Instruction learns from pattern
        lr1 = (1.0 - ref[w1].clamp(0, 1)).item()
        self.V[w1] = F.normalize(
            self.V[w1] + lr1 * (self.V[w0] - self.V[w1]), dim=0)

        # ── SPAWN: self-derived threshold (R3: threshold from V) ──
        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        thresh = G.max(dim=1).values.median().item()

        if sims[w0].item() < thresh:
            self.V = torch.cat([self.V, x_n.unsqueeze(0)])

        return action
