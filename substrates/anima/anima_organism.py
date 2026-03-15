#!/usr/bin/env python3
"""
ANIMA Stage 1 Organism — W + I dynamics in Search harness format.

Translates ANIMA's W (world model) and I (internal memory/accumulator)
into the harness Organism interface. T (temporal phase) is omitted for
Stage 1 — the key test is whether I accumulation creates sequence-dependent
final states (MI gap > 0).

Interface matches harness.py Organism exactly:
    __init__(seed, alive, rule_params)
    step(xs, signal) -> new xs (NC x D list-of-lists)
    centroid(xs) -> D-length list

Key ANIMA idea mapped to cellular automaton:
    W[i][k]: world model — learns to predict neighbor interaction
    I[i][k]: internal memory — accumulates prediction error (irreversible)

    When alive=True:  I accumulates error from each signal step
    When alive=False: I stays zero — no accumulation — control condition

The critical property: after a signal sequence A-B-C, the I state encodes
the sequence history. A different order C-B-A produces a different I state.
This makes the final centroid sequence-dependent, producing MI gap > 0.
"""

import math
import random

D = 12
NC = 6


def _tanh(x):
    return math.tanh(max(-20.0, min(20.0, x)))


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class AnimaOrganism:
    """
    ANIMA-style W+I organism for Stage 1 ground truth test.

    Parameters (rule_params keys):
        w_lr      (float): world model learning rate. Default 0.0003 (interior optimum,
                           Session 19 extended sweep — inverted-U: 0.0001→+0.0819,
                           0.0003→+0.0948, 0.001→+0.0616, 0.003→+0.0423, 0.01→+0.0274).
        tau       (float): I accumulation rate in [0,1]. Default 0.3.
        tau_slow  (float): slow I accumulation rate in [0,1]. Default 0.0. At 0.0,
                           mem_slow stays zero and dynamics are identical to the
                           single-timescale case (backward compatible). When > 0,
                           a second memory accumulator (mem_slow) integrates prediction
                           error on a slower timescale and feeds back into state dynamics
                           in parallel with the fast I drive.
        gamma     (float): signal coupling strength. Default 3.0 (boundary-optimal, Session 19).
        w_clip    (float): clip bound for w values. Default 2.0.
        noise     (float): state noise. Default 0.005.
        delta     (float): state update blend (1=pure new, 0=pure old). Default 1.0.
    """

    def __init__(self, seed=42, alive=False, rule_params=None):
        if rule_params is None:
            rule_params = {}

        self.seed = seed
        self.alive = alive

        # ANIMA parameters
        self.w_lr = rule_params.get('w_lr', 0.0003)  # Session 19: interior optimum at 0.0003
        self.tau = rule_params.get('tau', 0.3)
        self.tau_slow = rule_params.get('tau_slow', 0.0)  # Session 22: dual-timescale I
        self.gamma = rule_params.get('gamma', 3.0)   # Session 19: boundary-optimal, new canonical
        self.w_clip = rule_params.get('w_clip', 2.0)
        self.noise = rule_params.get('noise', 0.005)
        self.delta = rule_params.get('delta', 1.0)  # Session 16: delta=1.0 canonical

        # Initialize state
        random.seed(seed)

        # x: cellular state NC x D
        self.x = [
            [random.gauss(0, 0.5) for _ in range(D)]
            for _ in range(NC)
        ]

        # W: world model weights NC x D (one per cell, learns neighbor prediction)
        self.w = [
            [random.gauss(0, 0.1) for _ in range(D)]
            for _ in range(NC)
        ]

        # I: internal memory / prediction error accumulator NC x D
        # alive=True: accumulates; alive=False: stays zero
        self.mem = [
            [0.0] * D
            for _ in range(NC)
        ]

        # I_slow: slow-timescale memory accumulator NC x D
        # Only active when tau_slow > 0. Stays zero otherwise (backward compatible).
        self.mem_slow = [
            [0.0] * D
            for _ in range(NC)
        ]

    def step(self, xs, signal=None):
        """
        One step of W+I dynamics.

        W learns neighbor interaction; I accumulates prediction error
        (only when alive=True). Final state x depends on the history
        of I, making it sequence-sensitive.

        xs: NC x D list-of-lists (current state)
        signal: D-length list or None
        returns: NC x D list-of-lists (new state)
        """
        new_xs = []

        # Compute global signal mean field once (if signal present)
        if signal is not None:
            sig_mean = sum(signal) / D
        else:
            sig_mean = 0.0

        for i in range(NC):
            new_row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D

                # Neighbor interaction (ring topology)
                neighbor = xs[i][kp] * xs[i][km]

                # W predicts the neighbor interaction
                w_pred = self.w[i][k] * neighbor

                # Actual dynamics with signal modulation
                if signal is not None:
                    # Signal couples into neighbor term via gamma
                    s_kp = signal[kp]
                    s_km = signal[km]
                    neighbor_sig = (xs[i][kp] + self.gamma * s_kp) * (xs[i][km] + self.gamma * s_km)
                    actual = neighbor_sig
                    # Prediction error: difference between signal-modulated reality and prediction
                    err = actual - w_pred
                else:
                    err = 0.0

                # Update W (world model learns from prediction error)
                if signal is not None:
                    dw = self.w_lr * _tanh(err)
                    self.w[i][k] += dw
                    self.w[i][k] = _clip(self.w[i][k], -self.w_clip, self.w_clip)

                # Update I (internal memory accumulates prediction error, only when alive)
                if self.alive and signal is not None:
                    self.mem[i][k] = (1.0 - self.tau) * self.mem[i][k] + self.tau * _tanh(err)
                # else: mem stays at zero (or frozen at last value if alive just toggled)

                # Update I_slow (slow-timescale accumulator, only when alive and tau_slow > 0)
                if self.alive and signal is not None and self.tau_slow > 0:
                    self.mem_slow[i][k] = (1.0 - self.tau_slow) * self.mem_slow[i][k] + self.tau_slow * _tanh(err)

                # State update: x driven by W-predicted dynamics + I contribution
                # W term: world model drives base dynamics
                w_drive = self.w[i][k] * neighbor

                # I term: accumulated prediction error biases current state (fast timescale)
                i_drive = self.mem[i][k] * xs[i][k]

                # I_slow term: slow-timescale drive (zero when tau_slow=0.0)
                i_slow_drive = self.mem_slow[i][k] * xs[i][k]

                # New state via tanh nonlinearity
                pre_act = xs[i][k] + w_drive + i_drive + i_slow_drive
                phi = _tanh(pre_act)

                # Blend (delta=1.0 canonical → pure new state)
                v = (1.0 - self.delta) * xs[i][k] + self.delta * phi
                v += random.gauss(0, self.noise)

                new_row.append(v)
            new_xs.append(new_row)

        return new_xs

    def centroid(self, xs):
        """Mean across NC cells, returns D-length list."""
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]
