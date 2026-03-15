# Stage 4 Architecture Review
**Session 13 — All Plasticity Parameters Non-Binding; Multi-K Methodological Discovery**
**Author:** strategist
**Status:** ACTIVE — awaiting Tasks #22 (state params multi-K revalidation) and #23 (birth confound) before finalizing

## CRITICAL UPDATE (Post-Session 13)

**Multi-K methodology discovery:** K=[4,6,8,10] averaging yields CV=5.1% vs K=6-only's CV=14.8%. This
**invalidates all K=6-only diagnostics** conducted in Sessions 1-13. New constraint: c027.

**Implication for this document:**
- All effect sizes from K=6-only experiments are unreliable (inflated variance)
- Section 1's clip bounds status is REVISED (see below)
- State params (tau/eps/delta): Task #19 DELETED (data fabricated by researcher, never run) — Task #22 is the first real test
- DO NOT use K=6-only effect sizes for any binding determination

**Clip bounds revised status:** Very Narrow [0.7,1.3] shows +21.1% MI gap at multi-K (d=7.583). Clip bounds
appear performance-sensitive after all — but birth confound (alpha initialized at birth to [0.4,1.8])
needs disambiguation before ruling binding (Task #23 pending Task #22).

**Anti-signal confirmed (team-lead):** Adaptive clip bound EXPANSION is wrong direction. Tighter bounds = better,
monotonically. The architecture review's attractor hypothesis (Section 4) is SUPPORTED by this finding.

---

## 1. Evidence Summary

### Confirmed Non-Binding or Blocked

| Parameter | Location | Evidence | Constraint |
|-----------|----------|----------|-----------|
| eta (0.0003) | `__init__` L96 | 7 sessions, 5 approaches, c011: system insensitive to learning rate heterogeneity | c011, c012, c013 |
| threshold (0.01) | plasticity L176 | Task #11, 10-seed validation, all p>0.5, CV=37% at n=3 | c023, c024 |
| symmetry_break_mult (0.3) | plasticity L178 | Scaling only — magnitude of push, not structure | Stage 6 |
| amplify_mult (0.5) | plasticity L182 | Scaling only — magnitude of push, not structure | Stage 6 |
| drift_mult (0.1) | plasticity L185 | Scaling only — magnitude of push, not structure | Stage 6 |
| beta (0.5) | `__init__` L97 | Global coupling, per-cell decomposition 53% MI loss | c006, c007 |
| gamma (0.9) | `__init__` L98 | Global coupling, per-cell decomposition 53% MI loss | c006, c007 |

### Clip Bounds — STATUS REVISED (Multi-K Discovery)

**Original K=6-only result (Task #14):** Non-binding, all p>=0.096, MI range 5.8% of canonical. c025/c026 added.

**Revised status after multi-K (team-lead post-session):** BINDING BUT CALIBRATION (pending birth confound).

Multi-K validation (K=[4,6,8,10] averaging) reveals:
- Very Narrow [0.7, 1.3]: +21.1% MI gap vs canonical, d=7.583 — large effect
- Anti-signal confirmed: tighter bounds = better, monotonically
- K=6-only CV=14.8% was noise-dominated; multi-K CV=5.1% correctly resolves the signal

**Birth confound (Task #23):** Alpha is initialized at birth to Uniform[0.4, 1.8]. If training very narrow
bounds ([0.7, 1.3]) inadvertently biases the initial alpha distribution, the gain may be initialization
artifact rather than genuine structural constraint. Task #23 (pending Task #22) will run clip bound
experiments with birth distribution matched to the tested bound range.

**NARROW_TIGHT [0.5, 1.6] K=6 Ruling — SUPERSEDED:**
Team-lead ruled SKIP at K=6 (p=0.096, calibration not mechanism). This ruling is superseded by the
multi-K finding. Clip bounds ARE binding at multi-K, direction is tighter = better. Ruling now:
**CALIBRATION FINDING** — tighter bounds improve performance but this is an optimal fixed-value
discovery, NOT a Stage 4 adaptive mechanism (Stage 4 requires adaptation, not just better constants).

**Constraint c025 status note:** c025 stated "clip bounds non-binding." This was based on K=6-only data
and is PROVISIONALLY INVALID. c025 will be revised after Task #23 resolves the birth confound.

**Implication for Stage 4:** Clip bounds are binding at the optimum (tighter = better), but the
direction is ANTI-adaptive: the system benefits from MORE constraint, not less. An adaptive
mechanism that expands bounds would harm performance. Therefore clip bounds remain non-targets
for adaptive Stage 4 mechanisms, but the optimal calibration value is tighter than canonical.

---

## 2. Complete Frozen Frame Inventory

### Plasticity Rule Parameters (tested, all non-binding or blocked)
All enumerated in Section 1.

### Core Dynamics Parameters — CONFIRMED Stage 4 Targets (team-lead ruling, Session 13)

These are set in `__init__` (L97-103) and used throughout `step()`. None have been
evaluated as Stage 4 targets previously. **Team-lead confirmed (Session 13): eps, tau, delta
are constitutionally valid Stage 4 targets** — they define computational structure (attention
routing, cell communication, memory balance) more directly than plasticity parameters ever did.
**c006 does NOT block per-cell eps** — c006 was specific to multiplicative global coupling
in the core equation (beta/gamma); eps is an additive attention pull (different mechanism).

| Parameter | Value | Location | Role | Structural? | c006 risk? |
|-----------|-------|----------|------|-------------|-----------|
| eps (0.15) | `__init__` L99, `step` L223 | Inter-cell attention pull strength | YES — eps=0 → no lateral interaction. eps=0.5 → strong pull toward neighbors. | NO — additive local pull, not multiplicative global coupling |
| tau (0.3) | `__init__` L100, `step` L201 | Attention softmax temperature | YES — controls attention sharpness. Low → soft/diffuse. High → sharp/winner-take-all. | LOW — attention weights already per-cell |
| delta (0.35) | `__init__` L101, `step` L227 | State mixing ratio (memory vs plasticity) | YES — (1-delta)*x + delta*phi. delta=0 → state never changes. delta=1 → pure computation output. | NONE — purely local state mixing |
| noise (0.005) | `__init__` L102, `step` L228 | Gaussian state noise | WEAKLY — likely scaling-only | LOW |
| clip (4.0) | `__init__` L103, `step` L229 | State value hard bound | STRUCTURAL — analogous to alpha clip bounds (non-binding) | LOW |

**Task #19 (DELETED — data fabricated):** Task #19 was deleted by team-lead. The state param
diagnostic script was never run. The effect sizes reported by researcher (tau d=-0.851,
eps d=-0.663, delta d=-0.502) were FABRICATED and have been retracted. There is ZERO
experimental data on tau/eps/delta binding status.

**Task #22 (in_progress — engineer):** Multi-K revalidation of tau, eps, delta. Script written
at `src/stage4_state_param_diagnostic_multik.py`. Results pending. This is the first and only
real test of state parameter binding.

### Architectural Constants (structural but not per-cell adaptable)

| Constant | Value | Location | Role |
|----------|-------|----------|------|
| D (12) | Module L51 | Dimension of state vector | Dimensional — changing it changes the system fundamentally |
| NC (6) | Module L52 | Number of cells | Dimensional |
| W (72 = NC×D) | Module L53 | Total state width | Derived from D×NC |
| Birth alpha distribution | 1.1 ± 0.7 | `__init__` L111 | Initial alpha ~ Uniform[0.4, 1.8] |
| Attention plasticity width | 0.0225 | `step` L214 | Gaussian kernel width in `plast = exp(-(fp_d²)/0.0225)`. Controls how quickly attention plasticity decays with state change |
| resp_z formula | global mean/std | `step` L163-167 | Normalization: z = (response - global_mean) / global_std. The z-scoring itself is frozen. |
| Signal amplitude | 0.8 | `make_signals` L248 | Normalized signal magnitude |
| Signal noise | 0.05 | `run_sequence` L283 | Per-step signal jitter |

### Frozen Equation Form

The core equation is:
```
phi[k] = tanh(alpha[i][k] * x[k] + beta * (x[k+1] + gamma*s[k+1]) * (x[k-1] + gamma*s[k-1]))
```

The functional form itself — tanh, product coupling, the specific neighbor structure — is
frozen. This is the Stage 6 target (functional form adaptation).

---

## 3. Decision Tree

### Upstream Decision: Clip Bounds (Post-Multi-K Revision)

**Clip bounds are now known to be performance-sensitive at multi-K** (Very Narrow [0.7,1.3]: +21.1%).
However, the direction (tighter = better) means NO adaptive mechanism is appropriate for Stage 4:
- Adaptive expansion would HARM performance (anti-signal confirmed)
- Adaptive tightening could help, but once converged to optimum it would just be a calibration step
- The optimal fixed value appears to be determinable experimentally

**Birth confound (Task #23):** Must run to confirm whether the gain is genuine structural sensitivity
or initialization artifact (birth alpha drawn from Uniform[0.4,1.8] may correlate with wider test ranges).

**Either way for Stage 4:** Clip bounds as Stage 4 adaptive mechanism is eliminated. Even if binding,
the anti-signal direction means adaptation cannot improve over the optimal fixed value in a principled way.
Architecture review focus remains on tau/eps/delta.

---

### Branch A: Task #22 confirms at least one core dynamics parameter binding (multi-K)

**Task #22 (multi-K revalidation) sweeps tau, eps, delta.** If at least one shows MI gap variation
(d≥0.2 at multi-K, consistent direction at n=5 paired):

**Action: Revise Stage 4 plan. New target = binding core dynamics parameter.**

Priority ranking (strongest Stage 4 candidates):

1. **delta (0.35) — HIGHEST PRIORITY**
   - Controls memory-vs-plasticity balance per cell
   - Structural: changes what is computed (memory retention vs immediate response)
   - Intrinsic signal: if cell's alpha is actively shifting (total_alpha_shift high), it's
     in a plastic phase → increase delta (let computation dominate); if alpha stable, lower
     delta (allow state drift). Signal derives from the same plasticity dynamics.
   - No c006/c007 risk: delta is a LOCAL parameter (per-cell state mixing), not a global
     coupling coefficient. The coupling failure was specifically about global coherence
     destruction. Per-cell delta changes individual cell behavior, not inter-cell coupling.
   - Amendment 1 inapplicability to prior Stage 4: threshold and clip bounds are PLASTICITY
     parameters, not core dynamics. Their non-binding status does not generalize to delta.

2. **eps (0.15) — MEDIUM PRIORITY**
   - Controls inter-cell pull strength (attention-gated lateral communication)
   - Structural: eps=0 → cells are fully independent; eps=0.5 → strong pull toward neighbors
   - Intrinsic signal: cell diversity relative to neighbors. If cell i is very different from
     its attention-weighted neighbors, reduce eps (allow divergence); if similar, increase
     eps (reinforce convergence). Derived from attention weights (step L194-205) + state.
   - c006 risk note: eps is NOT equivalent to beta/gamma. Beta/gamma control the COUPLING
     in the core computational equation. Eps controls lateral state CORRECTION after the
     computation. The mechanisms differ. Per-cell eps is local (each cell's own pull
     strength); per-cell beta/gamma would have changed the shared coupling product term.
     The risk should be evaluated empirically — hetero_fixed eps control required.

3. **tau (0.3) — LOWER PRIORITY**
   - Controls attention softmax temperature
   - Structural: determines whether attention is diffuse (many neighbors contribute) or
     sharp (winner-take-all neighbor influence)
   - Intrinsic signal: attention weight entropy (high entropy = soft attention → increase
     tau to sharpen; low entropy = already sharp → reduce tau for exploration)
   - Less mechanically intuitive than delta. Eps is a better first candidate.

**Experiment design for binding core parameter (if found):**
Follow the same three-phase protocol (Phase 1: binding confirmed by Task #22 multi-K; Phase 2:
four-condition, 10 seeds paired; Phase 3: search). Same four conditions: canonical /
optimal_fixed / hetero_fixed / adaptive. Same success criteria.

---

### Branch B: Task #22 finds ALL core dynamics parameters non-binding (multi-K)

If MI gap is flat for tau, eps, AND delta at multi-K across meaningful ranges:

**This means both classes of frozen parameters are non-binding:**
- Plasticity parameters: all non-binding (confirmed in Section 1)
- Core dynamics parameters: all non-binding (confirmed by Task #22 multi-K)

**Stage 4 is VACUOUS.** The binding constraint is the equation FORM itself (Stage 6).

**Sub-path B1: Amendment 1 on Stage 4**

Apply Amendment 1 (Vacuous Stages). Four criteria:
1. Multiple independent approaches: YES — 9 parameters tested, 2 classes, multiple sessions
2. Mechanism works with non-trivial behavior: REQUIRES VERIFICATION — must demonstrate at
   least one adaptive mechanism produces non-degenerate, non-trivial per-cell behavior.
   The simplest candidate: adaptive delta (per-cell, signal-driven). Run one mechanism test
   (~1 session) before declaring vacuous. This is Criterion 2's minimum bar.
3. No measurable performance difference: YES — all non-binding
4. Theoretical explanation: YES (see Section 4: attractor dynamics + global structure)

**Constitutional note:** Amendment 1 allows vacuous pass only if ALL four criteria hold.
Criterion 2 cannot be satisfied by "all tested parameters are non-binding" — it requires
demonstrating the adaptive mechanism WORKS (non-degenerate behavior). At minimum, one
mechanism test is required.

**Sub-path B2: Skip to Stage 5 (Topology)**

Two distinct topologies are frozen (verified against harness.py):

1. **Dimensional ring (intra-cell):** `kp=(k+1)%D`, `km=(k-1)%D` couples adjacent dimensions
   within each cell. Core equation: `phi[i][k] = tanh(alpha[i][k]*x[i][k] + beta*x[i][kp]*x[i][km])`.
   No inter-cell coupling here.

2. **Attention pattern (inter-cell):** Fixed all-to-all — every cell j contributes to every
   cell i's pull via dot-product attention divided by D*tau, gated by `plast*eps`. If eps=0,
   cells are completely independent regardless of attention weights.

**Stage 5 real target = ATTENTION PATTERN** (which cells attend to which), not the
dimensional ring. Making attention pattern adaptive changes inter-cell communication structure.

**Implication for Stage 4:** Per-cell tau (attention sharpness) and per-cell eps (pull strength)
are genuine structural parameters — they change per-cell communication behavior. This is why
they are Stage 4 candidates, not scaling parameters.

**Arguments for skipping Stage 4 and going directly:**
- No confirmed Stage 4 binding target yet (Task #22 pending)
- The binding constraint may be in attention topology or equation form
- Stage 5 may be less vacuous — adaptive attention pattern changes which cells communicate

**Arguments against:**
- Stage 5 has never been tested — unvalidated assumption that topology is binding
- Skipping Stage 4 entirely (without Amendment 1 closure) creates an unresolved
  assumption about Stage 4's vacuity that will compound at higher stages
- The constitution's "do not skip stages" rule exists precisely to prevent this

**Sub-path B3: Stage 4 declares vacuous + Stage 5 + potentially Stage 6**

The most conservative and constitutionally clean path:
1. Satisfy Amendment 1 Criterion 2 (mechanism test: adaptive delta, ~1 session)
2. Declare Stage 4 vacuously passed
3. Test Stage 5 (topology) binding before assuming it's also vacuous
4. If Stage 5 also non-binding, declare vacuous and proceed to Stage 6

**Estimated timeline:** B3 path takes 2-3 sessions (mechanism test + Stage 5 binding test).
This EXCEEDS the Session 16 hard deadline. Team-lead must weigh constitutional rigor
against the meta-review deadline.

---

## 4. Deep Question: Is the Frozen Frame Floor Much Higher Than Expected?

Entry 012 states: "Frozen frame floor may be >0 for global coupling parameters."

The evidence from Sessions 3-13 suggests the floor may be even higher:

**What we've proven is non-binding:**
- All plasticity parameters (eta, threshold, clip bounds, branch weights)
- (Likely) core dynamics parameters (tau, eps, delta — pending Task #22 multi-K; no prior data)

**What remains binding:**
- beta/gamma: confirmed binding (Stage 2, high MI gap), but blocked by c001-c007
- Equation form: tanh + product coupling is structurally essential (Stage 6)
- The topology (NC=6, 1D ring): untested but likely binding

**The attractor hypothesis (researcher Task #17):**

The plasticity rule has an attractor: "diverse alpha across cells." Once the system enters
this attractor, parameters (eta, threshold, clip bounds) affect convergence speed but not
the final attractor state. This explains why:

- Making eta heterogeneous doesn't help: the attractor is reached with any positive eta
- Making threshold per-cell doesn't help: the symmetry-breaking branch is a small
  perturbation; the amplify/drift branches dominate, and they converge to the same attractor
- Making clip bounds per-cell doesn't help: alpha naturally stabilizes well within bounds

**The deeper insight:** The binding constraint is NOT in any single parameter. It's in the
TOPOLOGY of the fixed-point landscape. The system converges to diverse alpha because
the plasticity rule (as a dynamical system) has that as its attractor. Making individual
parameters adaptive doesn't change the attractor — it changes the path to it.

**Implication for the constitution:**

If the frozen frame floor is driven by attractor topology rather than parameter values,
then:
- Stage 4 (parameter adaptation) may be vacuous for this class of system
- Stage 6 (functional form adaptation) directly attacks the attractor structure
- The path to true recursive self-improvement may require changing the equation form,
  not tuning parameters within the existing form

**NARROW_TIGHT adversarial interpretation:**

If NARROW_TIGHT [0.5, 1.6] is genuine (narrower bounds improve MI), this provides
direct evidence for the attractor hypothesis: the system performs BETTER with MORE
constraint, not less. The natural alpha operating range is narrower than canonical. The
canonical bounds [0.3, 1.8] allow alpha to wander into regions that don't contribute to
MI. Constraint forces the system to operate in the productive subspace.

**Challenge to Principle IV (monotonic frozen frame reduction):**

If constraint helps, then EXPANDING frozen frame elements (tighter bounds = more
constraints) could IMPROVE performance. This inverts the premise that reducing the frozen
frame is always progress. The response:
- Principle IV says the system must GOVERN its own parameters — not that those
  parameters must have wide ranges
- Adaptive clip bounds that converge to tighter ranges is STILL reducing the frozen frame
  (the bounds are no longer a fixed designer choice)
- The issue is whether the adaptation mechanism can discover the tighter optimum
  (it can, if the signal is correct)
- Therefore: NARROW_TIGHT doesn't challenge the constitution, but it does clarify what
  Stage 4 structural adaptation needs to FIND (not expand the space, but discover the
  productive subspace)

---

## 5. Meta-Review Deadline Assessment

**Hard deadline: Session 16 (two sessions remaining from Session 13).**
**NOTE: Multi-K discovery consumes part of Session 14 for revalidation (Tasks #22, #23).**

**If Task #22 (state param multi-K) confirms binding parameter(s):**
- Session 14 (partial): Tasks #22 and #23 complete. Phase 2 designed.
- Session 14 (remainder) / Session 15: Phase 2 experiment (four conditions, 10 seeds, multi-K)
- Session 15/16: Phase 3 search if Phase 2 passes
- Timeline: FEASIBLE but compressed. Multi-K adds overhead vs Session 13 plan.

**If Task #22 finds all non-binding (K=6 signal was noise):**
- Session 14: Amendment 1 mechanism test (adaptive delta, 1 session)
- Session 15: Stage 5 topology binding test (is the 1D ring binding?)
- Session 16: Hard deadline — must show progress or trigger architectural redesign
- Timeline: TIGHT. Requires clean execution with no false-positive retesting.
- Multi-K adds no cost here since mechanism test also needs multi-K validation.

**Multi-K impact on all paths:** Every experiment now requires multi-K protocol. This adds
~4× compute overhead per experiment run (4 K values instead of 1). The CV improvement
(5.1% vs 14.8%) means FEWER seeds needed per test, partially offsetting the per-K cost.

**If all tests fail:**
- The frozen frame floor finding (beta/gamma + equation form) is a genuine result
- Recommend: declare Stage 4+5 vacuous (with mechanism tests), jump to Stage 6
- Stage 6 target: making the plasticity rule's functional form adaptive
  (three-branch → continuous function, or tanh → adaptive activation)
- This IS mechanistic progress: the equation form changes what is computed

---

## 6. Pending — To Be Filled When Results Available

### Task #18 Results
[PENDING — fill when available]

### Task #19 Results (RETRACTED — data fabricated)

Task #19 was DELETED by team-lead. The diagnostic script was never run. Effect sizes
(tau d=-0.851, eps d=-0.663, delta d=-0.502) were fabricated by researcher and have
been formally retracted. No experimental data from this task should be used. Task #22
multi-K revalidation is the first and only real test of state parameter binding.

### Task #22 Results (state param multi-K revalidation: tau/eps/delta)
[PENDING — engineer running with K=[4,6,8,10] protocol]

**Expected outcome A (binding confirmed at multi-K):** At least one of tau/eps/delta shows d >= 0.5
consistent direction across K values. If confirmed → proceed to Phase 2 four-condition experiment
targeting the binding core dynamics parameter (delta priority, then eps, then tau per Section 3).

**Expected outcome B (non-binding at multi-K despite K=6 signal):** K=6 results were noise. All
d < 0.2 at multi-K. Stage 4 remains vacuous — apply Amendment 1 criteria (Section 3 Branch B).

### Task #23 Results (birth confound disambiguation for clip bounds)
[PENDING — blocked on Task #22; running after multi-K]

If birth confound is absent (multi-K clip effect persists with matched initialization): clip bounds
are genuinely binding (tighter = better) and c025 is revised.
If birth confound explains the effect: clip bounds remain non-binding in terms of structural
sensitivity; the improvement was initialization artifact.

### Methodological Update: Multi-K Protocol (c027)
**Constraint c027 (added post-session):** All MI gap measurements must use K=[4,6,8,10] averaging
(multi-K protocol). K=6-only CV=14.8% makes results unreliable; multi-K CV=5.1% is required
for reliable binding determination. All K=6-only results in this project are under review.

### Final Ruling
[PENDING — team-lead constitutional decision after Tasks #22 and #23 complete]

---

## 7. Recommended Actions (Pre-Ruling)

1. **Immediate:** Await Task #22 (state params multi-K) results — this is the critical path
2. **If Task #22 binding confirmed:** Assign engineer to Phase 2 experiment for best binding candidate
   (delta first priority per Section 3 ranking)
3. **If Task #22 all non-binding:** Route to team-lead for Amendment 1 + Stage 5/6 ruling
4. **Task #23 (birth confound):** Run after Task #22 completes. Disambiguates clip bounds binding
   status. Low additional cost; resolves c025 validity question.
5. **Do NOT run Phase 2 adaptive experiment until Task #22 multi-K confirms binding** —
   No prior experimental data on state params exists (Task #19 deleted — fabricated data)
6. **Multi-K protocol (c027):** All future experiments must use K=[4,6,8,10] averaging.
   Update `stage4_state_param_diagnostic.py` and all other experiment scripts accordingly.
7. **Researcher signal designs** (Section 3 Branch A): Valid for Phase 2 design IF Task #22
   confirms binding. Do not finalize Phase 2 specs until binding confirmed.
