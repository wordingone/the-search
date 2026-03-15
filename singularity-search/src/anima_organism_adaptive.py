#!/usr/bin/env python3
"""
ANIMA Stage 2 — Phase-dependent adaptive w_lr organism.

Stage 2 requirement: The system's own computation produces a signal that,
if used to modify the system, improves performance.

Target parameter: w_lr (world model learning rate)
Interior optimum confirmed at w_lr≈0.0003 (Session 19 extended sweep).

Adaptation signal: mean_abs_err from the previous step (one float, internal state).
Using the previous step's value avoids causality issues within the (i,k) loop.

Adaptation rule:
    w_lr_eff = w_lr_base / (1.0 + err_scale * prev_mean_abs_err)

When err is HIGH (signal phase): w_lr_eff drops → W learns slowly → errors persist
  → I accumulates richly → sequence-distinguishable state.
When err is LOW (settling phase): w_lr_eff rises → W consolidates quickly.

This beats any fixed w_lr because the optimal rate varies across computation phases.
A fixed w_lr must trade off between these regimes; adaptive w_lr serves both.

Principle II compliance:
    prev_mean_abs_err is computed from err — the same variable already used to
    update W and I. No external evaluator. It is a byproduct of the computation.

c009 analog:
    w_lr_base and err_scale are EXTERNAL meta-parameters, not modified by the
    adaptation rule itself. The rule output w_lr_eff does not feed back into
    w_lr_base or err_scale.
"""

import math
import random

D = 12
NC = 6


def _tanh(x):
    return math.tanh(max(-20.0, min(20.0, x)))


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class AnimaOrganismAdaptive:
    """
    ANIMA W+I organism with phase-dependent adaptive w_lr.

    Parameters (rule_params keys):
        w_lr_base   (float): max w_lr when err→0. Default 0.001.
        err_scale   (float): sensitivity of w_lr to err magnitude. Default 3.0.
        tau         (float): I accumulation rate. Default 0.3.
        gamma       (float): signal coupling strength. Default 0.9.
        w_clip      (float): clip bound for W values. Default 2.0.
        noise       (float): state noise. Default 0.005.
        delta       (float): state blend. Default 1.0.
    """

    def __init__(self, seed=42, alive=False, rule_params=None):
        if rule_params is None:
            rule_params = {}

        self.seed = seed
        self.alive = alive

        # Adaptive w_lr parameters
        self.w_lr_base = rule_params.get('w_lr_base', 0.001)
        self.err_scale = rule_params.get('err_scale', 3.0)

        # Other ANIMA parameters
        self.tau = rule_params.get('tau', 0.3)
        self.gamma = rule_params.get('gamma', 0.9)
        self.w_clip = rule_params.get('w_clip', 2.0)
        self.noise = rule_params.get('noise', 0.005)
        self.delta = rule_params.get('delta', 1.0)

        # Adaptation state — one float, previous step's mean |err|
        self.prev_mean_abs_err = 0.0

        # Initialize cell/weight state
        random.seed(seed)

        self.x = [
            [random.gauss(0, 0.5) for _ in range(D)]
            for _ in range(NC)
        ]
        self.w = [
            [random.gauss(0, 0.1) for _ in range(D)]
            for _ in range(NC)
        ]
        self.mem = [
            [0.0] * D
            for _ in range(NC)
        ]

        # Diagnostics
        self.w_lr_eff_history = []

    def step(self, xs, signal=None):
        """
        One step of W+I dynamics with phase-dependent adaptive w_lr.

        w_lr_eff = w_lr_base / (1 + err_scale * prev_mean_abs_err)

        Computed at the START of the step from the PREVIOUS step's mean |err|,
        avoiding causality issues within the (i,k) loop.
        """
        # Effective w_lr for this step — uses PREVIOUS step's error
        w_lr_eff = self.w_lr_base / (1.0 + self.err_scale * self.prev_mean_abs_err)
        self.w_lr_eff_history.append(w_lr_eff)

        new_xs = []
        step_abs_err_sum = 0.0
        step_err_count = 0

        for i in range(NC):
            new_row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D

                neighbor = xs[i][kp] * xs[i][km]
                w_pred = self.w[i][k] * neighbor

                if signal is not None:
                    s_kp = signal[kp]
                    s_km = signal[km]
                    neighbor_sig = (xs[i][kp] + self.gamma * s_kp) * (xs[i][km] + self.gamma * s_km)
                    actual = neighbor_sig
                    err = actual - w_pred
                else:
                    err = 0.0

                # W update using effective (phase-dependent) w_lr
                if signal is not None:
                    dw = w_lr_eff * _tanh(err)
                    self.w[i][k] += dw
                    self.w[i][k] = _clip(self.w[i][k], -self.w_clip, self.w_clip)
                    step_abs_err_sum += abs(err)
                    step_err_count += 1

                # I update (alive only)
                if self.alive and signal is not None:
                    self.mem[i][k] = (1.0 - self.tau) * self.mem[i][k] + self.tau * _tanh(err)

                # State update
                w_drive = self.w[i][k] * neighbor
                i_drive = self.mem[i][k] * xs[i][k]
                pre_act = xs[i][k] + w_drive + i_drive
                phi = _tanh(pre_act)
                v = (1.0 - self.delta) * xs[i][k] + self.delta * phi
                v += random.gauss(0, self.noise)
                new_row.append(v)
            new_xs.append(new_row)

        # Store mean |err| for NEXT step's w_lr computation
        if signal is not None and step_err_count > 0:
            self.prev_mean_abs_err = step_abs_err_sum / step_err_count
        else:
            self.prev_mean_abs_err = 0.0

        return new_xs

    def centroid(self, xs):
        """Mean across NC cells, returns D-length list."""
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]

    def get_w_lr_eff_stats(self):
        """Return effective w_lr trajectory statistics."""
        h = self.w_lr_eff_history
        if not h:
            return {'mean': self.w_lr_base, 'std': 0.0, 'min': self.w_lr_base,
                    'max': self.w_lr_base, 'n': 0}
        m = sum(h) / len(h)
        s = math.sqrt(sum((v - m)**2 for v in h) / max(len(h) - 1, 1))
        return {'mean': m, 'std': s, 'min': min(h), 'max': max(h), 'n': len(h)}
