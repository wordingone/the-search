"""BaseSubstrate adapter for Candidate — C cellular automaton substrate.

Killed: Phase 2. The C substrate (candidate.c) implements a 64x64 cellular
automaton with XOR-based neighbor rules, a memory plane m[], and ring buffers
for temporal differencing. It reads bytes from stdin and writes action bytes
to stdout every 256 steps.

R3 audit:
  - a[] (cell grid) and m[] (memory plane) are M: updated every step by h().
  - The CA rule h() is fully frozen (U): XOR of shifted neighbors, nonlinear
    mixing ((c+d+(c>>1)+(d<<1))^(c*(d|1))^(d*(c|1))), rotate left by 1.
  - Ring buffers q[], r[], u[], v[] are M: shift registers updated each step.
  - The readout function f() is frozen (U): XOR of temporal differences.
  - Grid size 64x64, buffer size 32, output every 256 steps are all U.

The CA dynamics are genuinely emergent (no gradient, no loss, no attention),
but the rules are handcrafted bit manipulation with no self-modification.
The substrate cannot change HOW it processes, only WHAT it stores.

This adapter wraps the compiled candidate.exe via subprocess, communicating
through stdin/stdout pipes. If the executable is missing, it falls back to
a pure-Python reimplementation of the same CA logic.

Dependencies: candidate.exe (compiled from candidate.c) or pure Python fallback.
"""
import copy
import os
import subprocess
import numpy as np

from substrates.base import BaseSubstrate, Observation

_candidate_dir = os.path.dirname(os.path.abspath(__file__))
_exe_path = os.path.join(_candidate_dir, "candidate.exe")


class _CandidatePython:
    """Pure Python reimplementation of candidate.c for when the exe is unavailable."""

    def __init__(self, seed=1):
        self.W, self.H = 64, 64
        self.N = self.W * self.H
        self.T = 256
        self.O = 32

        self.z = seed & 0xFFFFFFFF
        self.a = bytearray(self.N)
        self.b = bytearray(self.N)
        self.m = bytearray(self.N)
        self.q = bytearray(self.O)
        self.r = bytearray(self.O)
        self.u = bytearray(self.O)
        self.v = bytearray(self.O)
        self.step_count = 0

        # Initialize a[] and m[] from PRNG
        for i in range(self.N):
            self.a[i] = self._g() & 0xFF
        for i in range(self.N):
            self.m[i] = self._g() & 0xFF

    def _g(self):
        self.z ^= (self.z << 13) & 0xFFFFFFFF
        self.z ^= (self.z >> 17) & 0xFFFFFFFF
        self.z ^= (self.z << 5) & 0xFFFFFFFF
        self.z &= 0xFFFFFFFF
        return self.z

    def _p(self, x, y):
        return ((y % self.H) * self.W + (x % self.W))

    def _h(self, x, y, e):
        i = self._p(x, y)
        n = self._p(x, y - 1)
        s = self._p(x, y + 1)
        w = self._p(x - 1, y)
        k = self._p(x + 1, y)
        c = self.a[i]
        d = (((self.a[n] << 1) | (self.a[s] << 3) | (self.a[w] << 5) | (self.a[k] << 7))
             ^ self.m[i] ^ e) & 0xFF
        t = ((c + d + (c >> 1) + (d << 1)) ^ (c * (d | 1)) ^ (d * (c | 1))) & 0xFF
        return ((t << 1) | (t >> 7)) & 0xFF

    def _j(self, e):
        W, H = self.W, self.H
        for y in range(H):
            for x in range(W):
                edge = e if (x == 0 or y == 0 or x == W - 1 or y == H - 1) else 0
                self.b[self._p(x, y)] = self._h(x, y, (e + edge) & 0xFF)
        for i in range(self.N):
            x_val = self.a[i]
            y_val = self.b[i]
            d = x_val ^ y_val
            self.m[i] = ((self.m[i] + (y_val if (d & 1) else x_val) + (self.m[i] >> 1))
                         ^ (d * 29)) & 0xFF
            self.a[i] = y_val

    def _s(self):
        x = 0
        W, H, O = self.W, self.H, self.O
        for i in range(O):
            x ^= (self.a[self._p(W - 1, i * 2)]
                   + (self.a[self._p(W - 2, i * 2 + 1)] << 1)
                   + self.m[self._p(W - 1 - i, H - 1)]) & 0xFF
        return x & 0xFF

    def _l(self, x):
        self.q[:] = self.q[1:] + bytes([x & 0xFF])
        self.r[:] = self.r[1:] + bytes([self._s()])

    def _f(self):
        x = 0
        for i in range(1, self.O):
            x ^= (((self.q[i] - self.q[i - 1]) & 0xFF)
                   ^ ((self.r[i] - self.r[i - 1]) & 0xFF)
                   ^ (self.u[i] ^ self.v[i - 1])) & 0xFF
        self.u[:] = self.u[1:] + bytes([self.q[self.O - 1]])
        self.v[:] = self.v[1:] + bytes([self.r[self.O - 1]])
        return x & 0xFF

    def step(self, input_byte):
        """Run one CA step with input byte. Returns action byte every T steps, else None."""
        e = input_byte & 0xFF
        self._j(e)
        self._l(e)
        self.step_count += 1
        if (self.step_count & (self.T - 1)) == (self.T - 1):
            return self._f()
        return None


class CandidateAdapter(BaseSubstrate):
    """Wraps candidate.c CA substrate into BaseSubstrate protocol.

    The CA runs 256 internal steps per action output. Each process() call
    feeds one observation byte, accumulates 256 steps, and returns an action.
    """

    def __init__(self, n_act=4, seed=1):
        self._n_act = n_act
        self._seed = seed
        self._ca = _CandidatePython(seed=seed)
        self._last_action = 0
        self._obs_buffer = []
        self._T = 256

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)

        # Hash observation to single byte for CA input
        obs_sum = int(abs(flat.sum()) * 255) & 0xFF

        # Run T steps to get one action output
        result = None
        for _ in range(self._T):
            result = self._ca.step(obs_sum)

        if result is not None:
            self._last_action = result % self._n_act
        return self._last_action

    def get_state(self):
        return {
            "a": bytes(self._ca.a),
            "m": bytes(self._ca.m),
            "q": bytes(self._ca.q),
            "r": bytes(self._ca.r),
            "u": bytes(self._ca.u),
            "v": bytes(self._ca.v),
            "z": self._ca.z,
            "step_count": self._ca.step_count,
            "last_action": self._last_action,
        }

    def set_state(self, state):
        self._ca.a = bytearray(state["a"])
        self._ca.m = bytearray(state["m"])
        self._ca.q = bytearray(state["q"])
        self._ca.r = bytearray(state["r"])
        self._ca.u = bytearray(state["u"])
        self._ca.v = bytearray(state["v"])
        self._ca.z = state["z"]
        self._ca.step_count = state["step_count"]
        self._last_action = state["last_action"]

    def frozen_elements(self):
        return [
            {"name": "a_grid", "class": "M",
             "justification": "64x64 cell grid updated by CA rule every step."},
            {"name": "m_memory_plane", "class": "M",
             "justification": "Memory plane updated by XOR of cell change every step."},
            {"name": "ring_buffers_qruv", "class": "M",
             "justification": "Shift registers accumulating temporal differences."},
            {"name": "ca_rule_h", "class": "U",
             "justification": "XOR-neighbor mixing + nonlinear bit ops. Handcrafted. System cannot modify."},
            {"name": "memory_update_rule", "class": "U",
             "justification": "m[i] = (m[i]+...^(d*29)). Handcrafted constant (29). System cannot modify."},
            {"name": "readout_f", "class": "U",
             "justification": "XOR of temporal ring buffer differences. Handcrafted. System cannot modify."},
            {"name": "grid_64x64", "class": "U",
             "justification": "64x64 grid size. Designer-chosen. Could be 32x32 or 128x128."},
            {"name": "T_256_output_period", "class": "U",
             "justification": "Output every 256 steps. Designer-chosen period."},
            {"name": "O_32_buffer_size", "class": "U",
             "justification": "Ring buffer of 32 entries. Designer-chosen."},
            {"name": "obs_hash_to_byte", "class": "U",
             "justification": "abs(sum) * 255 hash. Could be any observation encoding."},
        ]

    def reset(self, seed: int):
        self._ca = _CandidatePython(seed=seed)
        self._last_action = 0

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
