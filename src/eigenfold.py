#!/usr/bin/env python3
"""
EigenFold — Matrix codebook with eigenform dynamics.

Escapes the Hopfield trap: instead of softmax attention over vector similarities
(which IS modern Hopfield), uses matrix cross-application (noncommutative) and
eigenform perturbation stability for classification.

Each codebook element is a k×k matrix that:
1. Transforms input via cross-application Ψ(M_i, R)
2. Seeks eigenform Φ(M) = tanh(αM + βM²/k)
3. Classifies by STABILITY: most stable element under perturbation = best match

The fold lifecycle operates on the matrix codebook:
- Spawn: when no element is stable enough (perturbation exceeds threshold)
- Update: winning element absorbs perturbation, then recovers toward eigenform
- Merge: (not yet implemented) redundant elements with similar eigenforms fuse

Classification IS state change. The perturbation that classifies the input
is the same event that updates the codebook. No separate train/inference modes.

Dependencies: rk.py (matrix utilities)
"""
import math
import random
from rk import mzero, madd, msub, mscale, mmul, mtanh, frob, mcosine, mclip, mrand


class EigenFold:
    """
    Matrix codebook with eigenform dynamics.

    Parameters
    ----------
    k : int
        Matrix dimension (k×k elements). Default 4.
    alpha : float
        Eigenform linear coefficient. α > 1 makes zero unstable.
    beta : float
        Eigenform quadratic coefficient. Self-interaction strength.
    lr : float
        Winner update step size.
    recovery_steps : int
        Eigenform recovery steps after update.
    dt : float
        Eigenform step size.
    max_norm : float
        Matrix element clipping bound.
    init_steps : int
        Eigenform steps when initializing new element from input.
    spawn_threshold : float
        Spawn if min perturbation exceeds this.
    """

    def __init__(self, k=4, alpha=1.2, beta=0.8, lr=0.1,
                 recovery_steps=5, dt=0.03, max_norm=3.0,
                 init_steps=20, spawn_threshold=1.0):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.recovery_steps = recovery_steps
        self.dt = dt
        self.max_norm = max_norm
        self.init_steps = init_steps
        self.spawn_threshold = spawn_threshold
        self.elements = []  # list of (M, label)

    def phi(self, M):
        """Eigenform map: Φ(M) = tanh(αM + βM²/k)"""
        linear = mscale(M, self.alpha)
        quad = mscale(mmul(M, M), self.beta / self.k)
        return mtanh(madd(linear, quad))

    def psi(self, Mi, Mj):
        """Cross-application: Ψ(Mi,Mj) = tanh(α(Mi+Mj)/2 + βMiMj/k)"""
        avg = mscale(madd(Mi, Mj), self.alpha / 2.0)
        prod = mscale(mmul(Mi, Mj), self.beta / self.k)
        return mtanh(madd(avg, prod))

    def _delta(self, M, R):
        """Perturbation magnitude: how much does R destabilize M?"""
        return frob(msub(self.psi(M, R), M))

    def _eigenform_recover(self, M, steps=None):
        """Drive M toward eigenform."""
        steps = steps or self.recovery_steps
        for _ in range(steps):
            M = madd(M, mscale(msub(self.phi(M), M), self.dt))
            M = mclip(M, self.max_norm)
        return M

    def step(self, R, label=None):
        """
        One fold step. Classify + update in one event.

        Parameters
        ----------
        R : matrix (k×k)
            Input projected to matrix space.
        label : any, optional
            Class label (stored at spawn time).

        Returns
        -------
        pred_label : predicted class label (or None if empty)
        delta : perturbation magnitude of winning element
        """
        if not self.elements:
            self.elements.append((self._eigenform_recover(R, self.init_steps), label))
            return label, 0.0

        deltas = [self._delta(M, R) for M, _ in self.elements]
        win_idx = min(range(len(deltas)), key=lambda i: deltas[i])
        min_d = deltas[win_idx]
        pred_label = self.elements[win_idx][1]

        if min_d > self.spawn_threshold:
            self.elements.append((self._eigenform_recover(R, self.init_steps), label))
        else:
            M_w, lbl_w = self.elements[win_idx]
            delta_vec = msub(self.psi(M_w, R), M_w)
            M_w = madd(M_w, mscale(delta_vec, self.lr))
            M_w = self._eigenform_recover(M_w)
            M_w = mclip(M_w, self.max_norm)
            self.elements[win_idx] = (M_w, lbl_w)

        return pred_label, min_d

    def classify(self, R):
        """Read-only classification (for evaluation)."""
        if not self.elements:
            return None, None
        deltas = [self._delta(M, R) for M, _ in self.elements]
        best = min(range(len(deltas)), key=lambda i: deltas[i])
        return self.elements[best][1], deltas[best]

    def stats(self):
        return {
            'n_elements': len(self.elements),
            'labels': [lbl for _, lbl in self.elements],
            'eigenform_dists': [frob(msub(self.phi(M), M)) for M, _ in self.elements],
        }
