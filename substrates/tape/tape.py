"""
The Tape Machine — Phase 2, a point inside R1-R6 that is NOT vectors.

State: a sparse tape of integers. NOT a codebook. NOT vectors on a hypersphere.
The tape IS the program. Cells reference other cells via their values.
Reading a cell and following its pointer IS the chain.
Writing to a cell IS self-modification.

Match:  hash(input) → tape address          (not dot product)
Chain:  cell value → next address            (not re-matching V @ V[w])
Attract: cell = f(old_value, input)          (not normalize + lerp)
Spawn:  visiting blank cell = implicit spawn (not threshold comparison)

R1: No external objective. Tape reads/writes/chains. No loss.
R2: Write depends on read. Same step. Adaptation IS computation.
R3: Tape is the ONLY state. ALL cells writable. Chain addresses, actions,
    update rules — all determined by tape content. FULLY self-modifying.
R4: Revisit = self-test. Changed cell → different action → implicit comparison.
R5: The step function (8 lines) is frozen Python.
R6: Remove read → blind. Remove chain → no self-reference. Remove write → no
    learning. Each component is load-bearing.

S1-S21 DO NOT APPLY. No cosine. No saturation. No Goldilocks zone.
The failure modes of this system are entirely different from LVQ.
"""

import torch


class TapeMachine:

    def __init__(self, K=256, addr_bits=8):
        self.K = K                              # symbol alphabet size
        self.mask = (1 << addr_bits) - 1        # address space: 256 cells
        # Pre-populated tape: every cell non-zero from step 1.
        # Chain hits meaningful values immediately. No blank-cell problem.
        # This initialization IS a frozen frame (1 line).
        self.tape = {i: (i * 7 + 13) % K for i in range(1 << addr_bits)}

    def _read(self, addr):
        return self.tape.get(addr & self.mask, 0)

    def _write(self, addr, val):
        self.tape[addr & self.mask] = val % self.K

    def step(self, x, n_actions):
        """8 lines. The tape IS the program."""

        # Discretize input → integer key (frozen frame: the hash function)
        key = hash(tuple(x.topk(min(3, len(x))).indices.tolist())) & self.mask

        # MATCH: key → tape address → read symbol
        symbol = self._read(key)

        # CHAIN: symbol → next address → read instruction (self-reference)
        next_addr = (key + symbol + 1) & self.mask
        instruction = self._read(next_addr)

        # ACTION: from chain endpoint
        action = instruction % n_actions

        # ATTRACT: update both cells (read + write = one operation)
        self._write(key, symbol + (key & 0xFF) + 1)
        self._write(next_addr, instruction + (symbol & 0xFF) + 1)

        return action

    @property
    def size(self):
        """Number of visited tape cells."""
        return len(self.tape)
