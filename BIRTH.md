# Birth — Formal Derivation of the Feasible Region

*Started 2026-03-19. 505 experiments carved the mold. This document derives what's inside it.*

*The constraints are in English. English supports checking, not derivation. This document translates them to mathematics, then derives.*

---

## Primitives

Let **X** be the observation space (e.g., 64×64 palette frames).
Let **A** be the action space (game-specific; may include complex actions with parameters).
Let **S** be the state space — everything the substrate stores internally.

The substrate is a triple (f, g, F):

- **f_s: X → S** — the state update rule parameterized by current state s. Given observation x, produces next state.
- **g: S → A** — action selection. Given state, produces action.
- **F: S → (X → S)** — the meta-rule that maps states to update rules. F(s) = f_s.

The dynamics: at each timestep t,

```
s_{t+1} = F(s_t)(x_t) = f_{s_t}(x_t)
a_t = g(s_t)
x_{t+1} = env(x_t, a_t)    [external, not part of the substrate]
```

**F is the frozen frame.** It is the one function the substrate cannot modify. Everything else — s, and therefore f_s and g — is state.

---

## Formalization 1: The Structural Rules (R1-R3) and the Core Tension (U7, U17, U22)

### 1.1 Conditions

**R1 (No external objectives):**
f_s(x) depends only on s and x. No external signal (loss function, reward, label) enters the computation of s_{t+1}.

Formally: ∃ no function L external to f such that f_s(x) = h(s, x, L(s, x)). The substrate is a closed dynamical system over S given the input stream X.

*Note:* The action selection g(s) → a produces an action that affects the environment, which produces the next x. The environment IS external — but R1 governs the substrate's internal update, not the environment. The substrate sees x as raw input, not as reward.

**R2 (Adaptation = Computation):**
There is no separate optimizer. The state update f_s IS the adaptation mechanism.

Formally: let Θ(f) denote the "parameters" of f that change over time. Then Θ(f) ⊆ S. The only mechanism that modifies Θ is f itself. There is no gradient ∇L, no evolutionary selection, no external training loop. f modifies its own parameters as a side effect of processing observations.

Equivalently: the map t ↦ s_t is generated entirely by iterating f. There is no second dynamical system running alongside.

**R3 (Self-modification):**
The update rule at time t is determined by the state at time t. Different states produce different update rules.

Formally: F: S → (X → S) is not constant. There exist s₁ ≠ s₂ such that F(s₁) ≠ F(s₂) as functions on X.

*Strength of R3:* R3 does NOT require F to be injective (different states COULD produce the same rule). It requires F to be non-constant (some state differences produce different rules). The stronger version — F is injective, meaning every state difference produces a different rule — is not required by the constitution but would eliminate redundancy (see R6).

*What R3 excludes:* Any system where the update rule is the same regardless of state. For example, a k-means update where every centroid uses the same "attract toward nearest observation" rule regardless of how many centroids exist or where they are — the OPERATION is constant even though the DATA (centroid positions) changes. R3 requires the operation itself to depend on the state.

**U7 (Dominant amplification):**
Iterated application of f amplifies the dominant component of the state.

Formally: let f^n denote n-fold composition. For the linearization Df of f around a fixed point s*, the spectral radius ρ(Df) has a simple dominant eigenvalue λ₁. Then:

‖f^n(s, ·) - λ₁ⁿ v₁ v₁ᵀ f(s, ·)‖ → 0 as n → ∞

where v₁ is the dominant eigenvector. In English: the system's long-run behavior is determined by a single mode.

*Evidence:* Codebook (Step 405): recursive composition at all depths produces identical geometry = dominant eigenvector. Reservoir (Steps 438-439): Hebbian dynamics → rank-1 collapse. This is a mathematical property of ANY iterated linear (or locally linearizable) system.

**U17 (Unbounded accumulation):**
Some component of S grows without bound.

Formally: there exists a function φ: S → ℝ≥0 such that φ(s_{t+1}) ≥ φ(s_t) for all t, and lim_{t→∞} φ(s_t) = ∞.

*Evidence:* Codebook: entry count grows (spawn-on-novelty). LSH: edge count grows. k-means: edge count grows. Every successful navigation substrate has unbounded growth in SOME dimension of S.

*What φ can be:* Entry count, edge count, total mass, information content. The constraint says SOMETHING grows; it doesn't say what.

**U22 (Convergence kills exploration):**
If the state dynamics converge (s_t → s*), exploration ceases.

Formally: if ‖s_{t+1} - s_t‖ → 0, then the action sequence (a_t) becomes eventually periodic (repeats a fixed cycle).

*Evidence:* Codebook (Step 428): score convergence → random walk. TemporalPrediction (Step 437d): pred_err → 0 → weights frozen → action locked. Two architecturally distinct confirmations. Mathematically: convergent dynamics make the environment appear stationary → trivially predictable → no learning signal → no new states visited.

### 1.2 Analysis: The Core Tension

**Claim: R3 + U7 + U22 produce a tension that U17 resolves.**

*The tension:*

1. By R3, F is non-constant — different states produce different update rules.
2. By U7, iterated application of f converges to the dominant eigenvector of the linearized system.
3. By U22, convergence kills exploration — no new observations reach the system.
4. Without new observations, s stops changing (f_s(x) with fixed x and converging s produces fixed s*).
5. At the fixed point s*, the update rule is f_{s*} — a single, fixed function. R3 is formally satisfied (F is non-constant on S) but functionally dead (the system is AT s* and stays there).

*The resolution:*

6. U17 requires some φ(s) → ∞. If s converges to s*, then φ(s*) is finite. Contradiction with U17.
7. Therefore: s does NOT converge. The system cannot reach a fixed point.
8. But U7 says iterated f converges to the dominant eigenvector. How can both hold?

*The reconciliation:*

U7 applies to the LINEARIZATION of f around a fixed point. If s never reaches a fixed point (because U17 prevents it), then U7 describes local behavior (tendency toward dominant mode) but not global behavior (the system escapes before convergence completes).

The picture: f locally amplifies the dominant mode (U7), driving toward convergence. But before convergence completes, growth (U17) adds new state, which changes the update rule (R3), which changes the dominant mode, which restarts the amplification process.

**This is a limit cycle, not a fixed point.** The system oscillates between:
- Phase A: amplification toward dominant mode (U7)
- Phase B: growth disrupts the dominant mode (U17 + R3)

The period of this oscillation is not determined by the constraints. It is a **degree of freedom**.

### 1.3 Formal Statement

**Theorem (informal):** Let (f, g, F) satisfy R1, R2, R3, U7, U17, U22 simultaneously. Then the state trajectory (s_t) has no fixed point. The system exhibits perpetual non-convergence driven by the tension between U7 (convergence pressure) and U17 (growth pressure), mediated by R3 (growth changes the update rule).

**Proof sketch:**
- Suppose s_t → s*. Then φ(s_t) → φ(s*) < ∞, contradicting U17. ∎

**Corollary:** The update rule f_{s_t} never stabilizes. At every time t, the system is using a rule that will be different at some future time t' > t. Self-modification (R3) is not an optional feature that might atrophy — it is a NECESSARY consequence of the growth condition (U17).

*This resolves the question of whether R3 is "functionally dead" at convergence. It cannot be, because convergence cannot occur.*

### 1.4 Degrees of Freedom

1. **The growth function φ.** U17 says SOMETHING grows. It doesn't say what. In known substrates: entry count (codebook), edge count (graph), cell count (never, for LSH — fixed). The choice of φ determines the character of the growth.

2. **The oscillation period.** How quickly does growth (U17) disrupt the dominant mode (U7)? Fast disruption = chaotic, no time to exploit structure. Slow disruption = near-convergent, exploits structure but risks stagnation. This tradeoff is NOT determined by the constraints.

3. **The meta-rule F.** R3 says F is non-constant. U7 says F(s) has a dominant mode. But the FORM of F — how state maps to update rule — is the frozen frame. The constraints say F must exist and be non-constant, but don't specify it.

4. **The coupling between φ and F.** Growth (φ increasing) changes the update rule (via F). But how? Does adding a new node change F locally (only nearby update rules change) or globally (all update rules change)? This is not determined.

### 1.5 Tensions (unresolved)

**T1: U7 assumes linearity that R3 breaks.**
U7 is a property of LINEAR iterated systems (spectral theory). R3 makes f non-stationary (the update rule changes with state). A non-stationary system doesn't have a fixed linearization. So U7 may not apply in the standard sense. The experimental evidence (Steps 405, 438-439) confirms dominant amplification empirically, but the mathematical justification via spectral theory requires stationarity that R3 forbids.

**Status:** U7 may need to be reformulated. Instead of "converges to dominant eigenvector" (which requires stationarity), perhaps: "locally amplifies the component with largest variance" (which is an instantaneous property, not an asymptotic one). This reformulation is consistent with the evidence and doesn't require stationarity.

**T2: R3 strength is ambiguous.**
R3 says "every modifiable aspect IS modified." But the constitution defines this as "different states produce different update rules." These are not the same claim. The first says the system ACTIVELY modifies its operations. The second says the operations are RESPONSIVE to state. A system could satisfy the second (F non-constant) while violating the first (the system never visits states that exercise the non-constancy of F).

**Status:** This is a genuine ambiguity. The strong version (active modification) requires the trajectory (s_t) to visit regions of S where F varies. The weak version (responsive) only requires F to be non-constant on S, even if the trajectory stays in a region where F is locally constant. The experiments can't distinguish these — we need to decide which we mean.

---

*End of Formalization 1. Next: U3 (zero forgetting), U20 (local continuity), R6 (no deletable parts) — the growth topology.*
