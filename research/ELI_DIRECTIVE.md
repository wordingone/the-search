# FoldCore — Eli Directive: Current State
*From Leo, 2026-03-14. Supersedes all prior briefings.*

---

## Research State (96 Steps Complete)

Two arcs of research are complete. The fold equation is a verified perceptual primitive. The eigenform substrate search has produced mathematical findings but no applied capability.

### Arc 1: Fold Equation (Steps 1-71) — COMPLETE, FROZEN

**The fold works.** Canonical implementation: `fluxcore_manytofew.py` (11/11 tests pass).

| Result | Value | Step |
|---|---|---|
| CSI coverage | 33/33 divisions | 56 |
| Generation energy | 0.081 | 56 |
| P-MNIST AA | 56.7%, **0.0pp forgetting** | 65 |
| CIFAR-100 AA | 33.5% (matches EWC) | 71 |

Architecture: many-to-few (codebook vectors + fixed matrix cells + routing). 140K FLOPs/step. No backprop, no replay, no regularization.

**Do not modify the fold equation or many-to-few architecture.** It is frozen and verified.

### Arc 2: Eigenform Substrate (Steps 74-96) — AT CROSSROADS

**Old equation (tanh): EXHAUSTED.** 17 experiments (Steps 74-90). Φ(M) = tanh(αM + βM²/k). Classification doesn't work. k=4 specific. Algebra trivial.

**New substrate (spectral): ALGEBRAICALLY INTERESTING, APPLIED FAILURES.**
- Spectral Φ(M) = M·M^T / ||M·M^T|| · target_norm — 100% convergence at all k
- Formula C composition: genuine mixing, 15/15 non-commutative, deterministic
- **Step 95: P-MNIST FAILS** — 15.9% vs 46.2% baseline
- **Step 96: Order discrimination FAILS** — 55.5% vs 64.5% order-blind baselines
- Root cause: long-chain composition collapses to same attractors (class prototypes cos=0.9861)

**DeepSeek's challenge stands.** The substrate hasn't demonstrated a capability simpler methods can't match.

---

## Key Documents

| File | Purpose |
|---|---|
| `FRAMEWORK.md` | Lean governing document (118 lines). Read first. |
| `EXPERIMENT_LOG.md` | Full 96-step history (577 lines). Read on demand. |
| `JUNS_INTENT.md` | Jun's original research intent. |
| `WHAT_THE_FAILURES_TEACH.md` | Four separations that must die. |
| `EQUATION_CANDIDATES.md` | Untested substrate candidates (A1-A5, B1-B5, C1-C5). |
| `fluxcore_manytofew.py` (in foldcore repo) | Canonical fold implementation. |
| `scripts/run_step91-96*.py` | Recent experiment scripts. |

---

## Current Direction: AWAITING

Leo is deciding next direction with Jun. Options:
1. Accept eigenform characterization as complete — document and close
2. Test shorter sequences (length 2-3) where pairwise non-commutativity is fresh
3. Different use case for the substrate (not classification/sequence discrimination)
4. Return to EQUATION_CANDIDATES.md — test untested candidates
5. New eigenform equation — richer quotient structure needed

**Stand by for next experiment specification.**

---

## Hard Rules

- FRAMEWORK.md governs. Amendments require documented reasoning.
- One variable per experiment.
- Document what happens, not what you hope happens.
- Mail Leo ALL results with analysis before proceeding to next step.
- Do not modify the fold equation or many-to-few architecture.

---

*Questions -> mail Leo. FRAMEWORK.md governs.*
