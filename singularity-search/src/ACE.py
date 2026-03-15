#!/usr/bin/env python3
"""
ACE — Autopoietic Computation Engine

Every existing architecture has a level where structure is GIVEN, not GENERATED.
Transformers: fixed objective, external gradients.
SSMs: dynamic state, static dynamics.
All of them: the metric by which error is measured never changes.

A living system generates its own structure at every level.
Including the level that defines what "structure" means.

From general relativity:
  Mass tells space how to curve. Space tells mass how to move.

Translated:
  State tells the objective how to change. The objective tells the state how to move.

Each process carries:
  x ∈ ℝ^d      — what it IS
  c ∈ ℝ^d      — what it CARES ABOUT (dynamic objective)
  m ∈ (0,1)    — how AUTONOMOUS it is

  dx/dt = m·(c ⊙ tanh(x)) + (1-m)·Σ_j κ_ij·(x_j - x_i)/Z
  dc/dt = α·(dx/dt ⊙ tanh(x) - c)·max(0, 1-‖c‖²)
  dm/dt = β·(cos(dx/dt, c) - m)

  κ_ij = exp(-Σ_k |c_k|·(x_k - x'_k)² / τ)

No loss function. No optimizer. No training/inference split.
The system's continued coherent operation is its own objective.

Zero dependencies. Pure Python. The code IS the math.
"""

import math
import random
import sys

# ── Vector operations ────────────────────────────────────────

def vzero(d):        return [0.0] * d
def vrand(d, s=1.0): return [random.gauss(0, s) for _ in range(d)]
def vadd(a, b):      return [ai + bi for ai, bi in zip(a, b)]
def vsub(a, b):      return [ai - bi for ai, bi in zip(a, b)]
def vscale(v, s):    return [vi * s for vi in v]
def vmul(a, b):      return [ai * bi for ai, bi in zip(a, b)]
def vdot(a, b):      return sum(ai * bi for ai, bi in zip(a, b))
def vnorm(v):        return math.sqrt(vdot(v, v) + 1e-15)
def vtanh(v):        return [math.tanh(vi) for vi in v]

def vcosine(a, b):
    na, nb = vnorm(a), vnorm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, vdot(a, b) / (na * nb)))

# ── Process ──────────────────────────────────────────────────

class Process:
    """A self-maintaining process. Not a neuron. Not a node."""
    __slots__ = ('d', 'x', 'c', 'm', 'dx', 'age', 'births')

    def __init__(self, d):
        self.d = d
        self.x = vrand(d, 0.5)
        self.c = vrand(d, 0.1)
        self.m = 0.5
        self.dx = vzero(d)
        self.age = 0
        self.births = 0

    def self_force(self):
        return vscale(vmul(self.c, vtanh(self.x)), self.m)

    def coherence(self):
        return vcosine(self.dx, self.c)

    def concern_distance(self, other):
        """Distance weighted by what THIS process cares about."""
        return sum(abs(ck) * dk * dk
                   for ck, dk in zip(self.c, vsub(self.x, other.x)))

    def recycle(self):
        self.x = vrand(self.d, 0.5)
        self.c = vrand(self.d, 0.1)
        self.m = 0.5
        self.dx = vzero(self.d)
        self.age = 0
        self.births += 1

# ── Engine ───────────────────────────────────────────────────

class ACE:
    """
    N processes coupled through dynamic, concern-weighted topology.
    No loss function. The system runs. It either maintains itself or doesn't.
    """

    def __init__(self, n=12, d=8, seed=None):
        if seed is not None:
            random.seed(seed)
        self.n, self.d = n, d
        self.P = [Process(d) for _ in range(n)]
        self.alpha = 0.3   # concern tracking rate
        self.beta = 0.2    # membrane adaptation rate
        self.tau = 0.5     # coupling temperature
        self.dt = 0.02
        self.t = 0.0
        self.step_count = 0

    def step(self, signal=None):
        P, n, d, dt = self.P, self.n, self.d, self.dt

        # Phase 1: compute all velocities from previous state
        for i in range(n):
            pi = P[i]
            sf = pi.self_force()

            # Coupling: weighted pull toward neighbors
            cf, Z = vzero(d), 0.0
            for j in range(n):
                if i == j: continue
                k = math.exp(-pi.concern_distance(P[j]) / self.tau)
                Z += k
                cf = vadd(cf, vscale(vsub(P[j].x, pi.x), k))
            if Z > 1e-10:
                cf = vscale(cf, (1.0 - pi.m) / Z)

            # External signal enters proportional to openness
            ef = vscale(signal, (1.0 - pi.m) * 0.5) if signal else vzero(d)

            pi.dx = vscale(vadd(vadd(sf, cf), ef), 0.95)  # dissipation

        # Phase 2: integrate all states
        recycled = 0
        for pi in P:
            pi.x = vadd(pi.x, vscale(pi.dx, dt))

            # Concern tracks what actually moves, bounded
            tx = vtanh(pi.x)
            target = vmul(pi.dx, tx)
            bound = max(0.0, 1.0 - vdot(pi.c, pi.c))
            pi.c = vadd(pi.c, vscale(vsub(target, pi.c), self.alpha * bound * dt))

            # Membrane tracks coherence
            pi.m += self.beta * (pi.coherence() - pi.m) * dt
            pi.m = max(0.005, min(0.995, pi.m))

            if pi.m < 0.02 and pi.age > 50:
                pi.recycle()
                recycled += 1
            pi.age += 1

        self.t += dt
        self.step_count += 1
        return recycled

    # ── Observables ──────────────────────────────────────────

    def avg_coherence(self):
        return sum(p.coherence() for p in self.P) / self.n

    def avg_membrane(self):
        return sum(p.m for p in self.P) / self.n

    def energy(self):
        return sum(vdot(p.dx, p.dx) for p in self.P) / self.n

    def concern_diversity(self):
        mean = vzero(self.d)
        for p in self.P:
            mean = vadd(mean, vscale(p.c, 1.0 / self.n))
        return sum(sum((ck - mk) ** 2 for ck, mk in zip(p.c, mean))
                   for p in self.P) / self.n

    def avg_degree(self, threshold=0.5):
        total = 0.0
        for i, pi in enumerate(self.P):
            for j, pj in enumerate(self.P):
                if i != j:
                    total += math.exp(-pi.concern_distance(pj) / self.tau) > threshold
        return total / self.n

    def measure(self):
        return dict(t=round(self.t, 3),
                    coh=round(self.avg_coherence(), 4),
                    mem=round(self.avg_membrane(), 4),
                    E=round(self.energy(), 6),
                    div=round(self.concern_diversity(), 4),
                    deg=round(self.avg_degree(), 2))

    def process_signal(self, signal, steps=50):
        """Inject signal, let settle, return mean displacement."""
        before = [p.x[:] for p in self.P]
        for _ in range(steps):
            self.step(signal)
        resp = vzero(self.d)
        for i, p in enumerate(self.P):
            resp = vadd(resp, vsub(p.x, before[i]))
        return vscale(resp, 1.0 / self.n)

# ── Demo ─────────────────────────────────────────────────────

def bar(v, w=25, lo=0.0, hi=1.0):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '█' * n + '░' * (w - n)

def run():
    print("ACE — Autopoietic Computation Engine\n")

    engine = ACE(n=12, d=8, seed=42)

    # Phase 1: self-organization
    print("── Self-organization from noise ──\n")
    hdr = f"{'step':>6}  {'coherence':>9}  {'membrane':>9}  {'energy':>10}  {'diversity':>9}  {'degree':>7}"
    print(hdr)
    for _ in range(20):
        for __ in range(100): engine.step()
        m = engine.measure()
        print(f"{engine.step_count:>6}  {m['coh']:>9.4f}  {m['mem']:>9.4f}  "
              f"{m['E']:>10.6f}  {m['div']:>9.4f}  {m['deg']:>7.2f}")

    # Phase 2: perturbation and recovery
    print("\n── Perturbation response ──\n")
    pre = engine.measure()
    sig = vzero(8); sig[0] = 2.0; sig[1] = -1.0
    print("Signal: dim[0]=+2.0, dim[1]=-1.0\n")

    for _ in range(10):
        for __ in range(50): engine.step(sig)
        m = engine.measure()
        print(f"  {engine.step_count:>5}: coh={m['coh']:+.4f}  mem={m['mem']:.4f}  E={m['E']:.6f}")

    print("\nSignal removed:\n")
    for _ in range(10):
        for __ in range(50): engine.step()
        m = engine.measure()
        print(f"  {engine.step_count:>5}: coh={m['coh']:+.4f}  mem={m['mem']:.4f}  E={m['E']:.6f}")

    post = engine.measure()
    delta = abs(post['coh'] - pre['coh'])
    print(f"\n  → {'Homeostasis' if delta < 0.1 else 'Reorganization'} "
          f"(Δcoherence = {delta:.4f})")

    # Phase 3: history-dependent computation
    print("\n── History-dependent computation ──\n")
    print("Same seed, different conditioning → different responses\n")

    def conditioned_engine(conditioning_signal):
        e = ACE(n=12, d=8, seed=100)
        for _ in range(500): e.step()
        for _ in range(300): e.step(conditioning_signal)
        for _ in range(200): e.step()
        return e

    ea = conditioned_engine([1,1,1,1,0,0,0,0])
    eb = conditioned_engine([0,0,0,0,1,1,1,1])

    test = vrand(8, 1.0)
    ra, rb = ea.process_signal(test, 100), eb.process_signal(test, 100)
    cos_ab = vcosine(ra, rb)

    print(f"  Response A norm:     {vnorm(ra):.4f}")
    print(f"  Response B norm:     {vnorm(rb):.4f}")
    print(f"  Response divergence: {vnorm(vsub(ra, rb)):.4f}")
    print(f"  Cosine similarity:   {cos_ab:.4f}")
    print(f"\n  → {'Different histories → different responses. Concerns carry context.' if abs(cos_ab) < 0.9 else 'Signal overwhelmed history.'}")

    # Phase 4: concern evolution
    print("\n── Concern evolution (3 processes × 4 dims) ──\n")
    ev = ACE(n=6, d=4, seed=77)
    for epoch in range(15):
        for _ in range(200): ev.step()
        print(f"  t={ev.t:>6.1f}")
        for i in range(3):
            p = ev.P[i]
            c = " ".join(f"{v:+.3f}" for v in p.c)
            print(f"    P{i}: c=[{c}]  m={bar(p.m, 12)} {p.m:.3f}  coh={p.coherence():+.3f}")
        print()

    print("─" * 60)
    print("""
  Dynamic objectives: concerns evolved to track actual dynamics.
  Emergent topology: coupling shaped by what processes care about.
  Self-regulation: coherence drives autonomy drives coherence.
  History as context: same signal, different history, different response.
  No loss function: continued coherent operation is the only criterion.
""")


THEORY = """
WHY DYNAMIC OBJECTIVES BREAK THE SCALING LAW

A static network with N·d parameters: O(N·d) degrees of freedom.

A dynamic-concern system: at any moment, c_i selects a subspace of ℝ^d
that process i amplifies. There are exponentially many subspace
configurations across N processes. Each configuration yields a different
effective topology → different input-output mapping.

Static system modes: O(N·d).
ACE modes: O(2^(N·d)) in the combinatorial limit.

Same argument as: N bits encode 2^N configurations,
N analog values encode an N-dimensional manifold.

Dynamic concerns are a stored program the system writes for itself
on the same substrate that executes it.

The question is not "how many parameters"
but "how many dynamical regimes can the system visit?"

Static system: one.
ACE: as many as the dynamics can explore.
"""


if __name__ == '__main__':
    if '--theory' in sys.argv:
        print(THEORY)
    else:
        run()
        if '--full' in sys.argv:
            print(THEORY)
