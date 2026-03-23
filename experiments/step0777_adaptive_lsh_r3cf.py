"""
step0777_adaptive_lsh_r3cf.py — Adaptive LSH: attempt to pass R3_counterfactual.

R3 hypothesis:
  H_nav_planes (K hash planes) are U in 674. If the substrate can DERIVE its
  hash planes from observed data (e.g., PCA directions of seen observations),
  then H_nav_planes becomes M (modified by dynamics). U count drops by 1.

  If R3_counterfactual also passes: pretraining on task T gives better planes
  → P_warm > P_cold on T.

Design:
  - Run N observations through PCA
  - Use top-K PCA directions as hash planes (instead of random)
  - PCA planes are computed from a RUNNING COVARIANCE (M element)
  - Action selection: same argmin-edge-count as 674 (I element)
  - U elements eliminated: H_nav_planes (now derived from data)
  - Remaining U: channel_0_only, avgpool16, mean_centering, K (number of planes)

If this passes R3_counterfactual: pretraining gives better planes → warm eval
shows more consistent action selection.

This is the MINIMAL modification to 674 that eliminates one U element.

Expected results:
  R3_static: FAIL — still has K, channel_0_only, avgpool16, mean_centering as U
  R3_counterfactual: PASS (if PCA planes transfer) or FAIL (if planes don't help)
  R2: PASS (covariance matrix M changes every step)

If R3_cf PASSES: PCA planes ARE the R3 mechanism. The substrate learns HOW to
distinguish states (not just WHAT states it has been to).
If R3_cf FAILS: PCA planes don't transfer either — the problem is deeper.
"""
import sys, os, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM, K_NAV

print("=" * 70)
print("STEP 777 — ADAPTIVE LSH: R3_COUNTERFACTUAL ATTEMPT")
print("=" * 70)
print()


class AdaptiveLSHSubstrate(BaseSubstrate):
    """674 variant where H_nav_planes are derived from PCA of seen observations.

    H_nav_planes = top-K eigenvectors of running covariance matrix.
    Update covariance online (M element).
    Re-derive planes every UPDATE_EVERY steps.

    R3 hypothesis: H_nav_planes is now M (derived from covariance). If
    pretraining stabilizes the planes, P_warm > P_cold on R3 counterfactual.

    Remaining U elements:
      - channel_0_only: U (still uses first channel only)
      - avgpool16: U (still uses 16×16 pooling)
      - mean_centering: U (still subtracts mean)
      - K_NAV: U (K=12 planes, could be 8 or 16)
      - UPDATE_EVERY: U (how often to recompute planes)

    So 5 U elements → R3_static still FAILS.
    But R3_counterfactual should PASS if covariance transfer helps.
    """

    UPDATE_EVERY = 100   # U: re-derive planes every N steps

    def __init__(self, n_actions: int = 4, seed: int = 0, k: int = K_NAV):
        self._n_actions = n_actions
        self._k = k
        self._seed = seed
        self._t = 0
        self._init_state(seed)

    def _init_state(self, seed: int):
        rng = np.random.RandomState(seed)
        # Running covariance: M (accumulates from observations)
        self.cov_sum = np.zeros((DIM, DIM), dtype=np.float64)  # M
        self.obs_count = 0  # M (observation counter for covariance)
        self.obs_mean = np.zeros(DIM, dtype=np.float64)  # M (running mean)

        # Hash planes: start random, updated from covariance
        # Initially U (random), then M after enough observations
        self.H_nav = rng.randn(self._k, DIM).astype(np.float32)  # starts random

        # Navigation graph (same as 674)
        self.G = {}       # M: edge counts
        self.live = set() # M: active cells
        self._pn = None
        self._pa = None
        self._cn = None

    def _hash(self, x: np.ndarray) -> int:
        bits = (self.H_nav @ x > 0).astype(np.uint8)
        return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)

    def _update_covariance(self, x: np.ndarray):
        """Update running covariance online. M element."""
        self.obs_count += 1
        n = self.obs_count
        # Welford-style update of mean
        delta = x.astype(np.float64) - self.obs_mean
        self.obs_mean += delta / n
        delta2 = x.astype(np.float64) - self.obs_mean
        # Rank-1 update of scatter matrix (unnormalized cov)
        self.cov_sum += np.outer(delta, delta2)

    def _rederive_planes(self):
        """Derive H_nav from top-K eigenvectors of covariance. M→M update."""
        if self.obs_count < self._k + 1:
            return  # Not enough observations for stable PCA
        try:
            cov = self.cov_sum / max(self.obs_count - 1, 1)
            # Use partial SVD on cov (symmetric → eigendecomposition)
            # For speed: use power iteration approximation (top-k only)
            vals, vecs = np.linalg.eigh(cov)
            # Take top-K eigenvectors (largest eigenvalues)
            idx = np.argsort(vals)[::-1][:self._k]
            self.H_nav = vecs[:, idx].T.astype(np.float32)  # (K, DIM)
        except Exception:
            pass  # Keep existing planes on numerical failure

    def process(self, observation) -> int:
        x = _enc_frame(observation)
        self._t += 1

        # Update running covariance (M element)
        self._update_covariance(x)

        # Re-derive planes periodically
        if self._t % self.UPDATE_EVERY == 0:
            self._rederive_planes()

        # Navigation (same as 674 coarse graph)
        n = self._hash(x)
        self.live.add(n)

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1

        self._cn = n

        # Action selection: argmin outgoing edges (I element — same as 674)
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
        return {
            "cov_sum": self.cov_sum.copy(),
            "obs_mean": self.obs_mean.copy(),
            "obs_count": self.obs_count,
            "H_nav": self.H_nav.copy(),
            "G": copy.deepcopy(self.G),
            "live": set(self.live),
            "_pn": self._pn, "_pa": self._pa, "_cn": self._cn,
            "t": self._t,
        }

    def set_state(self, state: dict) -> None:
        self.cov_sum = state["cov_sum"].copy()
        self.obs_mean = state["obs_mean"].copy()
        self.obs_count = state["obs_count"]
        self.H_nav = state["H_nav"].copy()
        self.G = copy.deepcopy(state["G"])
        self.live = set(state["live"])
        self._pn = state["_pn"]
        self._pa = state["_pa"]
        self._cn = state["_cn"]
        self._t = state["t"]

    def frozen_elements(self) -> list:
        return [
            {"name": "covariance_matrix", "class": "M",
             "justification": "Running covariance accumulates from observations. System-driven."},
            {"name": "H_nav_planes", "class": "M",
             "justification": "Derived from top-K PCA eigenvectors of covariance. System-derived, not random."},
            {"name": "G_edge_counts", "class": "M",
             "justification": "Graph edges accumulate from transitions. System-driven."},
            {"name": "live_cells", "class": "M",
             "justification": "Set of visited cells grows. System-driven."},
            {"name": "binary_hash", "class": "I",
             "justification": "Sign of projection. Removing destroys cell identity."},
            {"name": "argmin_edge_count", "class": "I",
             "justification": "Argmin of outgoing edges. Removing degrades L1 rate (same as 674)."},
            {"name": "pca_eigenvectors", "class": "I",
             "justification": "Eigenvectors of covariance. Removing destroys the M derivation mechanism."},
            {"name": "channel_0_only", "class": "U",
             "justification": "Uses only first channel of 3D obs. Could use all channels. System doesn't choose."},
            {"name": "avgpool16", "class": "U",
             "justification": "16×16 average pooling. Could be 8×8 or 32×32. System doesn't choose."},
            {"name": "mean_centering", "class": "U",
             "justification": "Subtract pixel mean. Could normalize by std. System doesn't choose."},
            {"name": "K_NAV_12", "class": "U",
             "justification": "K=12 hash planes. Could be 6, 16. System doesn't choose."},
            {"name": "UPDATE_EVERY_100", "class": "U",
             "justification": "Recompute planes every 100 steps. Could be 50, 500. System doesn't choose."},
        ]

    def reset(self, seed: int) -> None:
        self._t = 0
        self._init_state(seed)

    def on_level_transition(self) -> None:
        self._pn = None
        self._pa = None

    @property
    def n_actions(self) -> int:
        return self._n_actions


# ── Run judge audit ────────────────────────────────────────────────────────
from substrates.judge import ConstitutionalJudge

print("Auditing AdaptiveLSH with ConstitutionalJudge...")
judge = ConstitutionalJudge()
t0 = time.time()
results = judge.audit(AdaptiveLSHSubstrate, n_audit_steps=500)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print()

print("=" * 70)
print("STEP 777 RESULTS — AdaptiveLSH")
print("=" * 70)
print()

for check in ['R1', 'R2', 'R3', 'R3_counterfactual']:
    r = results.get(check, {})
    passed = r.get('pass')
    detail = r.get('detail', '')
    if check == 'R3_counterfactual':
        p_cold = r.get('P_cold', '?')
        p_warm = r.get('P_warm', '?')
        impr = r.get('improvement', '?')
        print(f"  {check}: pass={passed} P_cold={p_cold} P_warm={p_warm} improvement={impr}")
    else:
        print(f"  {check}: pass={passed} | {detail[:80]}")

fe = results.get('frozen_elements', [])
u = [e for e in fe if e.get('class') == 'U']
m = [e for e in fe if e.get('class') == 'M']
i_ = [e for e in fe if e.get('class') == 'I']
print(f"\n  frozen_elements: U={len(u)} M={len(m)} I={len(i_)}")
print(f"  U elements: {[e['name'] for e in u]}")
print(f"  M elements: {[e['name'] for e in m]}")
print()
print(f"  Summary: {results.get('summary', {})}")

# ── Compare to 674 on R3_counterfactual ─────────────────────────────────
print()
print("=" * 70)
print("COMPARISON: AdaptiveLSH vs 674 on R3_counterfactual")
print("=" * 70)
from substrates.step0674 import TransitionTriggered674

print("Running 674 counterfactual...")
t0 = time.time()
r674 = judge._check_r3_counterfactual(TransitionTriggered674)
elapsed674 = time.time() - t0
print(f"Done in {elapsed674:.1f}s")

r_alsh = results.get('R3_counterfactual', {})
print()
print(f"{'Substrate':<25} {'P_cold':<8} {'P_warm':<8} {'Improvement':<12} {'Pass'}")
print("-" * 65)
print(f"{'674 bootloader':<25} {str(r674.get('P_cold','?')):<8} {str(r674.get('P_warm','?')):<8} "
      f"{str(r674.get('improvement','?')):<12} {r674.get('pass')}")
print(f"{'AdaptiveLSH':<25} {str(r_alsh.get('P_cold','?')):<8} {str(r_alsh.get('P_warm','?')):<8} "
      f"{str(r_alsh.get('improvement','?')):<12} {r_alsh.get('pass')}")
print()

# Interpretation
alsh_p = r_alsh.get('pass')
if alsh_p is True:
    print("FINDING: AdaptiveLSH PASSES R3_counterfactual.")
    print("PCA planes ARE transferable. Covariance encodes useful structure.")
    print("This is the first R3_cf-passing substrate in the search.")
elif alsh_p is None:
    print("FINDING: R3_counterfactual SKIPPED (R1 fail or error).")
else:
    print("FINDING: AdaptiveLSH FAILS R3_counterfactual.")
    print("PCA planes don't transfer better than random planes.")
    print("The R3_cf failure is not due to random planes — it's structural.")

print()
print("STEP 777 DONE")
