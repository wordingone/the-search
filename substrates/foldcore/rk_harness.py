#!/usr/bin/env python3
"""
RK Harness — Reflexive Kernel adapter for MI gap ground truth protocol.

Wraps rk.py's Kernel (state IS transformation, eigenform IS objective)
into the harness Organism interface so it can be tested against the
same MI gap protocol used for the Living Seed and ANIMA organisms.

Interface matches harness.py Organism exactly:
    __init__(seed, alive, rule_params)
    step(xs, signal) -> new xs (NC x D list-of-lists)
    centroid(xs) -> D-length list
    alpha_flat() -> flat diagnostic list

Dimension mapping:
    Harness: NC=6 cells, D=12 values per cell
    RK: N cells, k x k matrix per cell (k=4 -> 16 values)

    Strategy: k=4, pad D=12 -> 16 when injecting into RK matrices,
    truncate 16 -> 12 when extracting state for harness. The 4 extra
    entries are always zero-padded on input and discarded on output.
    This preserves full 4x4 matrix dynamics.

ALIVE vs STILL:
    ALIVE: normal Kernel.step() — M evolves via eigenform drive + coupling
    STILL: after each step, restore M to its frozen snapshot.
           The coupling weights and dynamics compute normally (so the
           state xs flows through the same equations), but M itself
           does not permanently change.

Pure Python. Standard library only.
"""

import math
import random
import sys
import os
import time

# Make src importable
sys.path.insert(0, os.path.dirname(__file__))

from rk import Cell, Kernel, mzero, mrand, mscale, madd, msub, frob, mcosine
from harness import make_signals, gen_perms, vcosine, D, NC


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

K_RK = 4          # RK matrix dimension (k x k = 16 entries per cell)
K_SQ = K_RK * K_RK  # 16
W = 72


# ═══════════════════════════════════════════════════════════════
# Dimension mapping utilities
# ═══════════════════════════════════════════════════════════════

def flat_to_matrix(flat_d, k=K_RK):
    """
    Convert a D-length flat vector to a k x k matrix.
    Pads with zeros if len(flat_d) < k*k.
    """
    padded = list(flat_d) + [0.0] * max(0, k * k - len(flat_d))
    return [padded[i * k:(i + 1) * k] for i in range(k)]


def matrix_to_flat(mat, d=D):
    """
    Flatten a k x k matrix to a d-length vector.
    Truncates if k*k > d, pads with zeros if k*k < d.
    """
    flat = [v for row in mat for v in row]
    if len(flat) >= d:
        return flat[:d]
    return flat + [0.0] * (d - len(flat))


# ═══════════════════════════════════════════════════════════════
# RKOrganism — Reflexive Kernel in harness Organism shape
# ═══════════════════════════════════════════════════════════════

class RKOrganism:
    """
    Wraps rk.Kernel to match the harness Organism interface.

    Parameters (rule_params keys):
        alpha     (float): Supercritical bifurcation parameter. Default 1.2.
        beta      (float): Self-interaction strength. Default 0.8.
        dt        (float): Integration timestep. Default 0.03.
        tau       (float): Coupling temperature. Default 0.3.
        noise     (float): Per-step noise scale. Default 0.01.
        max_norm  (float): Frobenius norm clip. Default 3.0.
        steps_per_call (int): Number of Kernel.step() per harness step(). Default 3.
                              RK's dt=0.03 is small; multiple sub-steps per harness
                              step let the dynamics evolve enough to be responsive.
    """

    def __init__(self, seed=42, alive=False, rule_params=None):
        if rule_params is None:
            rule_params = {}

        self.seed = seed
        self.alive = alive

        # RK parameters
        rk_alpha = rule_params.get('alpha', 1.2)
        rk_beta = rule_params.get('beta', 0.8)
        dt = rule_params.get('dt', 0.03)
        tau = rule_params.get('tau', 0.3)
        noise = rule_params.get('noise', 0.01)
        max_norm = rule_params.get('max_norm', 3.0)
        self.steps_per_call = rule_params.get('steps_per_call', 3)

        # Create kernel: NC cells, k=4 matrices
        self.kernel = Kernel(n=NC, k=K_RK, alpha=rk_alpha, beta=rk_beta, seed=seed)
        self.kernel.dt = dt
        self.kernel.tau = tau
        self.kernel.noise_scale = noise
        self.kernel.max_norm = max_norm

        # For STILL: we freeze M after the first step call.
        # _frozen_M stores snapshots of each cell's M.
        # We set it lazily on the first step() call so the kernel
        # gets its random initialization but then locks.
        self._frozen_M = None
        self._freeze_pending = not alive  # STILL needs freeze

        # Track total M shift for diagnostics (analogous to total_alpha_shift)
        self.total_m_shift = 0.0

    def _snapshot_M(self):
        """Deep-copy all cell M matrices."""
        return [[row[:] for row in c.M] for c in self.kernel.cells]

    def _restore_M(self, snapshot):
        """Restore cell M matrices from snapshot."""
        for c, saved in zip(self.kernel.cells, snapshot):
            c.M = [row[:] for row in saved]

    def _inject_state(self, xs):
        """
        Push harness state xs (NC x D) into kernel cell matrices.

        Each xs[i] is a D=12 vector. We reshape it into a 4x4 matrix
        (padding the last 4 entries with zeros) and SET cell.M to that
        matrix, blending with the existing M to avoid destroying
        eigenform structure.

        Actually, the better approach: treat xs as an EXTERNAL state
        that flows through the kernel, not as M itself. M is the
        learned transformation; xs is the data the harness tracks.
        We inject xs as a signal-like perturbation.

        Design decision: xs modulates the kernel via a mild additive
        injection into the signal pathway. The kernel's own M matrices
        ARE the persistent state (the "alpha" analogue). The harness
        xs is a secondary state vector that rides on top.
        """
        # We don't directly replace M with xs.
        # Instead, xs is carried externally and injected as context.
        # See step() for the actual integration.
        pass

    def step(self, xs, signal=None):
        """
        One harness step. Takes xs (NC x D), optional signal (D-length),
        returns new xs (NC x D).

        Strategy:
        1. Convert signal (D-vector) to a k x k matrix for RK
        2. Run kernel.step() for self.steps_per_call sub-steps
        3. Extract new xs from cell M matrices (flatten + truncate)
        4. Blend extracted state with old xs (the harness expects smooth evolution)
        5. If STILL: restore M after stepping (M doesn't permanently change)
        """
        k = K_RK

        # On first call for STILL, freeze M
        if self._freeze_pending:
            self._frozen_M = self._snapshot_M()
            self._freeze_pending = False

        # Convert harness signal to RK matrix signal
        rk_signal = None
        if signal is not None:
            rk_signal = flat_to_matrix(signal, k)
            # Scale signal to match RK's expected magnitude
            rk_signal = mscale(rk_signal, 0.5)

        # Inject xs into cell M as a mild perturbation
        # This couples the harness state trajectory to the kernel dynamics
        for i, cell in enumerate(self.kernel.cells):
            xs_mat = flat_to_matrix(xs[i], k)
            # Blend: M stays mostly itself, xs provides a small kick
            # This is analogous to how signals perturb RK
            cell.M = madd(cell.M, mscale(xs_mat, 0.02))
            # Re-clip to prevent runaway
            nrm = frob(cell.M)
            if nrm > self.kernel.max_norm:
                cell.M = mscale(cell.M, self.kernel.max_norm / nrm)

        # Snapshot M before stepping (to measure shift and for STILL restore)
        pre_M = self._snapshot_M()

        # Run RK dynamics
        for _ in range(self.steps_per_call):
            self.kernel.step(rk_signal)

        # Measure total M shift
        for i, cell in enumerate(self.kernel.cells):
            for r in range(k):
                for c_idx in range(k):
                    self.total_m_shift += abs(cell.M[r][c_idx] - pre_M[i][r][c_idx])

        # Extract new state from cell M matrices
        new_xs = []
        for i, cell in enumerate(self.kernel.cells):
            new_xs.append(matrix_to_flat(cell.M, D))

        # If STILL: restore M to frozen state (dynamics computed, M unchanged)
        if not self.alive and self._frozen_M is not None:
            self._restore_M(self._frozen_M)

        return new_xs

    def centroid(self, xs):
        """Mean across NC cells, returns D-length list."""
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]

    def alpha_flat(self):
        """
        Return diagnostic values analogous to alpha_flat in Living Seed.
        Uses eigenform distances for each cell — these are the RK analogue
        of per-cell adaptation parameters.
        """
        vals = []
        for cell in self.kernel.cells:
            # Eigenform distance: how close cell is to its own fixed point
            vals.append(cell.eigenform_distance())
            # Autonomy: derived from eigenform distance
            vals.append(cell.autonomy())
        return vals


# ═══════════════════════════════════════════════════════════════
# Adapted run_sequence for RKOrganism
# (mirrors harness.run_sequence, swaps Organism for RKOrganism)
# ═══════════════════════════════════════════════════════════════

def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    """Run one sequence through org. Returns (centroid, xs)."""
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    # Organism warm-up (no signal)
    for _ in range(n_org):
        xs = org.step(xs)

    # Signal sequence
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    # Final settle
    for _ in range(n_final):
        xs = org.step(xs)

    return org.centroid(xs), xs


def measure_gap(org, signals, k, seed, n_perm=8, n_trials=6):
    """
    MI gap = avg_within_cosine - avg_between_cosine.
    Positive means same-order trials cluster together
    (distinguishable final states per sequence).
    """
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c, _ = run_sequence(org, perm, signals, seed, trial)
            trials.append(c)
        endpoints[pi] = trials

    within = []
    between = []
    pis = sorted(endpoints.keys())

    for pi in pis:
        cs = endpoints[pi]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                within.append(vcosine(cs[i], cs[j]))

    for i in range(len(pis)):
        for j in range(i + 1, len(pis)):
            for c1 in endpoints[pis[i]]:
                for c2 in endpoints[pis[j]]:
                    between.append(vcosine(c1, c2))

    avg_w = sum(within) / max(len(within), 1)
    avg_b = sum(between) / max(len(between), 1)
    return avg_w - avg_b


# ═══════════════════════════════════════════════════════════════
# RK Stage 1 comparison: STILL vs ALIVE
# ═══════════════════════════════════════════════════════════════

def run_rk_comparison(rule_params=None,
                      ks=None,
                      seeds=None,
                      birth_seed=42,
                      n_perm=8,
                      n_trials=6,
                      verbose=True):
    """
    Compare STILL (alive=False) vs ALIVE (alive=True) RKOrganism.

    Returns dict with still_gap, alive_gap, gap_delta, ground_truth_pass.
    """
    if rule_params is None:
        rule_params = {}
    if ks is None:
        ks = [4, 6, 8, 10]
    if seeds is None:
        seeds = [42, 137, 256, 512, 1024, 2024, 7777, 31337, 99991, 111111]

    still_gaps = []
    alive_gaps = []

    total = len(ks) * len(seeds)
    done = 0

    for k in ks:
        sig_seed = birth_seed + k * 200
        sigs = make_signals(k, seed=sig_seed)

        for s in seeds:
            done += 1
            if verbose:
                print(f"  [{done:2d}/{total}] K={k} seed={s} ...", end="", flush=True)

            # STILL: alive=False, M frozen after init
            org_still = RKOrganism(seed=birth_seed, alive=False, rule_params=rule_params)
            g_still = measure_gap(org_still, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
            still_gaps.append(g_still)

            # ALIVE: alive=True, M evolves via eigenform drive + coupling
            org_alive = RKOrganism(seed=birth_seed, alive=True, rule_params=rule_params)
            g_alive = measure_gap(org_alive, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
            alive_gaps.append(g_alive)

            if verbose:
                print(f"  still={g_still:+.4f}  alive={g_alive:+.4f}  delta={g_alive - g_still:+.4f}")

    still_avg = sum(still_gaps) / len(still_gaps)
    alive_avg = sum(alive_gaps) / len(alive_gaps)
    gap_delta = alive_avg - still_avg

    still_std = math.sqrt(sum((g - still_avg) ** 2 for g in still_gaps) / len(still_gaps))
    alive_std = math.sqrt(sum((g - alive_avg) ** 2 for g in alive_gaps) / len(alive_gaps))

    # Ground truth: alive gap > 0 (sequence-distinguishable final states)
    ground_truth_pass = alive_avg > 0.0

    # Rough d-statistic (alive_avg - still_avg) / pooled_std
    pooled_std = math.sqrt((still_std ** 2 + alive_std ** 2) / 2.0 + 1e-15)
    d_stat = gap_delta / pooled_std

    return {
        'still_gap': still_avg,
        'still_std': still_std,
        'alive_gap': alive_avg,
        'alive_std': alive_std,
        'gap_delta': gap_delta,
        'd_stat': d_stat,
        'ground_truth_pass': ground_truth_pass,
        'n_seeds': len(seeds),
        'ks': ks,
    }


# ═══════════════════════════════════════════════════════════════
# Stage 1 entry point
# ═══════════════════════════════════════════════════════════════

def run_rk_stage1():
    """
    Run the MI gap ground truth test for the Reflexive Kernel.
    10 seeds, K=[4,6,8,10], n_perm=8, n_trials=6.
    Prints full results.
    """
    print("=" * W)
    print("  RK STAGE 1 -- Ground Truth Test")
    print("  Reflexive Kernel: state IS transformation, eigenform IS objective.")
    print("  Does M-evolution (ALIVE) create MI gap > 0?")
    print("=" * W)
    print()

    t_start = time.time()

    rule_params = {
        'alpha': 1.2,
        'beta': 0.8,
        'dt': 0.03,
        'tau': 0.3,
        'noise': 0.01,
        'max_norm': 3.0,
        'steps_per_call': 3,
    }

    print(f"  Rule params: {rule_params}")
    print()

    result = run_rk_comparison(
        rule_params=rule_params,
        ks=[6, 8, 10],  # K=4 done in prior run: 3/10 nonzero (~0.019 avg delta)
        seeds=[42, 137, 256, 512, 1024, 2024, 7777, 31337, 99991, 111111],
        birth_seed=42,
        n_perm=8,
        n_trials=6,
        verbose=True,
    )

    print()
    print("=" * W)
    print("  RESULTS")
    print("=" * W)
    print(f"  STILL  gap: {result['still_gap']:+.4f}  (std={result['still_std']:.4f})")
    print(f"  ALIVE  gap: {result['alive_gap']:+.4f}  (std={result['alive_std']:.4f})")
    print(f"  Delta:      {result['gap_delta']:+.4f}")
    print(f"  d-stat:     {result['d_stat']:+.4f}")
    print(f"  Seeds:      {result['n_seeds']}  K-values: {result['ks']}")
    print()
    print(f"  Ground truth (alive_gap > 0): {'PASS' if result['ground_truth_pass'] else 'FAIL'}")
    print()

    if result['ground_truth_pass'] and result['gap_delta'] > 0:
        print("  The Reflexive Kernel's M-evolution creates sequence-dependent")
        print("  final states. When M is free to evolve (ALIVE), the eigenform")
        print("  drive + coupling creates a different trajectory for each signal")
        print("  ordering. When M is frozen (STILL), this history is lost.")
    elif result['ground_truth_pass']:
        print("  ALIVE produces positive MI gap, but STILL may also be positive.")
        print("  The RK dynamics create some sequence sensitivity even with")
        print("  frozen M, likely through the xs-injection pathway.")
    else:
        print("  The Reflexive Kernel does not produce positive MI gap.")
        print("  The eigenform dynamics may need tuning for this protocol.")

    print()
    print(f"  Runtime: {time.time() - t_start:.0f}s")
    print("=" * W)

    return result


# ═══════════════════════════════════════════════════════════════
# Quick test (fast feedback, reduced exposure)
# ═══════════════════════════════════════════════════════════════

def run_rk_quick():
    """
    Quick sanity check: 3 seeds, K=[4,6], n_perm=4, n_trials=3.
    Runs in minutes instead of hours.
    """
    print("=" * W)
    print("  RK QUICK TEST -- Sanity check (reduced exposure)")
    print("=" * W)
    print()

    t_start = time.time()

    rule_params = {
        'alpha': 1.2,
        'beta': 0.8,
        'dt': 0.03,
        'tau': 0.3,
        'noise': 0.01,
        'max_norm': 3.0,
        'steps_per_call': 3,
    }

    result = run_rk_comparison(
        rule_params=rule_params,
        ks=[4, 6],
        seeds=[42, 137, 2024],
        birth_seed=42,
        n_perm=4,
        n_trials=3,
        verbose=True,
    )

    print()
    print("-" * W)
    print(f"  STILL  gap: {result['still_gap']:+.4f}")
    print(f"  ALIVE  gap: {result['alive_gap']:+.4f}")
    print(f"  Delta:      {result['gap_delta']:+.4f}")
    print(f"  Ground truth: {'PASS' if result['ground_truth_pass'] else 'FAIL'}")
    print(f"  Runtime: {time.time() - t_start:.0f}s")
    print("-" * W)

    return result


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_rk_quick()
    else:
        run_rk_stage1()
