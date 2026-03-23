"""
plain_lsh.py — PlainLSH: baseline control substrate.

Plain LSH k=12 + running-mean edge counts. No 674 refinement.
Control for Group B experiments. Isolates contribution of ℓ_π refinement
to R3 dynamics.

R3 status: FAIL — 2 U elements (hash planes, k).
Compare to 674 (9 U elements): PlainLSH has fewer U elements but ALSO
fewer M elements (no ref_hyperplanes, no aliased_set, no fine graph).
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM

K_NAV = 12


class PlainLSH(BaseSubstrate):
    """Plain LSH k=12 + edge-count running mean. No refinement, no aliasing.

    Mechanism: identical to 674 coarse graph, but WITHOUT:
      - Transition-triggered refinement (ref dict stays empty)
      - Aliased-cell detection (aliased set stays empty)
      - Fine graph (G_fine stays empty)

    This isolates the contribution of ℓ_π refinement to R3 dynamics.
    R3 hyp: PlainLSH R3 dynamic score < 674 R3 dynamic score because
    ref_hyperplanes (M in 674) never change in PlainLSH.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._init_state(seed)

    def _init_state(self, seed: int) -> None:
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.G = {}         # (cell, action) -> {successor: count}
        self.live = set()
        self._pn = self._pa = None
        self._cn = None
        self.t = 0

    def _hash_nav(self, x: np.ndarray) -> int:
        return int(np.packbits(
            (self.H_nav @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def process(self, observation) -> int:
        x = _enc_frame(observation)
        n = self._hash_nav(x)
        self.live.add(n)
        self.t += 1

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1

        self._cn = n
        best_a, best_s = 0, float('inf')
        for a in range(self._n_actions):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a

        self._pn = n
        self._pa = best_a
        return best_a

    def get_state(self) -> dict:
        import copy
        return {
            "G_size": len(self.G),
            "live_count": len(self.live),
            "t": self.t,
            "H_nav": self.H_nav.copy(),
            # Full state for set_state / R3 counterfactual
            "G": copy.deepcopy(self.G),
            "live": set(self.live),
            "_pn": self._pn,
            "_pa": self._pa,
            "_cn": self._cn,
        }

    def set_state(self, state: dict) -> None:
        """Restore full internal state from get_state() snapshot."""
        import copy
        self.H_nav = state["H_nav"].copy()
        self.G = copy.deepcopy(state["G"])
        self.live = set(state["live"])
        self.t = state["t"]
        self._pn = state["_pn"]
        self._pa = state["_pa"]
        self._cn = state["_cn"]

    def frozen_elements(self) -> list:
        return [
            {"name": "H_nav_planes", "class": "U",
             "justification": "k=12 random LSH planes. System doesn't choose direction or count."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign projection for cell identity. Removing destroys all graph structure."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Removing -> random walk -> L1 never reached."},
            {"name": "edge_count_update", "class": "M",
             "justification": "G updated with each (prev_cell, action, curr_cell) transition. System-driven."},
            {"name": "k_12", "class": "U",
             "justification": "k=12 hash width. Could be 8, 16, 24. System doesn't choose."},
        ]

    def reset(self, seed: int) -> None:
        self._init_state(seed * 1000)

    def on_level_transition(self) -> None:
        self._pn = None
        self._pa = None

    @property
    def n_actions(self) -> int:
        return self._n_actions
