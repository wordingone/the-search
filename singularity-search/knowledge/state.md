# Project Knowledge State

## Active Constraints
What NOT to do and why:

- **[c001]** Response-weighted analytical gradients are not reliable proxies for MI direction
- **[c002]** Analytical gradient magnitude is too weak for beta/gamma without 100-1000× learning rate boost
- **[c003]** No local proxy for beta/gamma exists (7 tested, best r=0.44, none >0.7)
- **[c004]** Finite-diff MI gradient violates Principle II (external measurement, not self-generated)
- **[c005]** Beta/gamma MI landscape has multiple local maxima (no unique optimum)
- **[c006]** Per-cell decomposition of global coupling parameters destroys performance (53% MI loss)
- **[c007]** Beta/gamma coupling is fundamentally global, not locally decomposable
- **[c008]** Multiplicative self-referential meta-rates cause bang-bang oscillation (inherently unstable)
- **[c009]** Stage 3 meta-rate must be EXTERNAL to eta, not eta itself
- **[c010]** 5-seed validation is insufficient for detecting parameter effects in this system
- **[c011]** Per-cell eta adaptation does not improve performance — the system is insensitive to learning rate heterogeneity
- **[c012]** resp_z derivative tower collapses at order 1 — higher-order derivatives cannot drive adaptation
- **[c013]** Stage 3+ requires a fundamentally different signal family, not derivatives of resp_z
- **[c014]** Evaluation protocol CV=29% — need 50+ seeds for meaningful comparisons or variance-reduction redesign
- **[c015]** Future experiments must use n_perm=8, n_trials=6 (2× exposure) to achieve CV≈20%
- **[c016]** Within-seed paired design increases variance — do not use
- **[c017]** Adding extreme K values (K=3,12) increases variance — stick to K=[4,6,8,10]
- **[c018]** Evolutionary search with 3-seed eval is unable to find genuine improvements — selects for noise
- **[c019]** Future search must use 10+ seeds with 2× exposure protocol
- **[c020]** delta_correlation one-way eta adaptation actively harms performance - successful adaptation is misread as degradation
- **[c021]** delta_stability Phase 2 improvement (+0.0068) is within noise floor - Phase 3 with 10+ seeds required to confirm
- **[c022]** One-way eta reduction based on error-push correlation is an anti-signal for this system
- **[c023]** Plasticity threshold is non-binding — the system is insensitive to threshold variation across [0.01, 0.1]
- **[c024]** Phase 1 diagnostics using 3 independent seeds per condition cannot detect effects reliably at CV=37% — always use paired seeds or 10+ seeds
- **[c025]** alpha_clip_lo/hi are non-binding — structurally active (15% saturation) but performance-insensitive across meaningful range
- **[c026]** Parameter activity (hitting bounds) does not imply parameter binding (performance depends on value)
- **[c027]** Single-K diagnostics insufficient; multi-K protocol required (K=[4,6,8,10] averaging achieves CV≈5.1% vs K=6-only CV≈14.8%)
- **[c028]** c028: Always use per-cell cosine similarity for MI gap measurement, not centroid. Centroid can give opposite rankings.
- **[c029]** c029: Entry 042 d=7.583 for VERY_NARROW clip bounds is protocol-specific to the short (50-step) protocol and does not generalize to the canonical harness.py protocol
- **[c030]** RETRACT c028: per-cell and centroid cosine give identical rankings under the same protocol; neither is inherently superior
- **[c031]** Tau's MI effect is K-dependent (positive at K=4, reverses at K=8) — no consistent adaptive direction exists
- **[c032]** c031: Tau is non-binding in [0.2, 0.5] range. Entry 046 d=+1.202 is a 5-seed false positive (10-seed: d=+0.317, p=0.317). Adaptive tau (plast-driven [0.15, 0.35]) produces d=-0.039 vs fixed. No tau value or adaptive rule improves performance.
- **[c033]** c032: Plast-driven per-cell adaptive tau does not beat fixed baselines. Mechanism works (tau values adapt, non-degenerate) but performance is flat — tau is not a binding constraint.
- **[c034]** c034: Eps is non-binding in [0.05, 0.5] range. Entry 046 eps=0.05 d=-3.092 is a 5-seed false positive (10-seed: d=-0.461, p=0.145). No eps value significantly beats canonical.
- **[c035]** c035: Delta optimal at boundary (1.0). Improvement is monotonic: higher delta = better MI. Canonical delta=0.35 is suboptimal by 6%. This is calibration, not an adaptive opportunity - optimum at boundary means adaptive delta cannot beat best fixed value.
- **[c036]** c036: State-output divergence is an anti-signal for delta adaptation. Divergence measures per-step state change magnitude; delta performance depends on per-sequence signal fidelity. These are uncorrelated. Converges to ~0.16 regardless of initialization, significantly worse than fixed delta=1.0 (d=-2.4, p=0.0000). Never retry divergence-based adaptive delta.
- **[c037]** c037: Architecture ceiling declared. Living Seed frozen frame minimum: 6/8. Substrate cannot satisfy Stage 7 necessary conditions (no self-representation mechanism). Stage progression halted under Amendment 2. Substrate requires architectural modification to proceed.
- **[c038]** ANIMA Stage 7 requires hypernetwork H within I generating I's own update weights — do not attempt Stage 7 without this mechanism
- **[c039]** Self-modification in ANIMA must apply at cycle boundaries (T consolidation points), not per-step — prevents c008-analog instability
- **[c040]** Stage 7 exit criterion requires novel-input comparison, not training-input comparison — must test on held-out sequences
- **[c041]** ANIMA Stage 7 self-modification must apply at T cycle boundaries, not per-step — prevents c008-analog bang-bang instability
- **[c042]** ANIMA validation protocol must characterize CV before setting minimum seed count — do not assume Living Seed CV=29% applies
- **[c043]** ANIMA evaluation must guard against false positives (c010 lesson) and anti-signals (c020/c036 lesson) even though the specific Living Seed signals are not applicable
- **[c044]** c044: ANIMA w_lr interior optimum at 0.0003 (10-seed confirmed). Canonical w_lr updated from 0.01 to 0.0003. Improvement: 3.5× MI gap.
- **[c045]** c045: ANIMA tau non-binding in [0.1, 0.7] range. Consistent with Living Seed c032. Do not attempt tau adaptation.
- **[c046]** c046: ANIMA gamma boundary-optimal (monotonically increasing to 3.0). Calibration-only, not adaptive opportunity.
- **[c047]** c047: Per-step adaptation signals (mean_abs_err, gradient EMA) cannot capture sequence-level w_lr optimality in ANIMA. The MI gap is determined over full K-signal sequences (~1000 steps); per-step signals encode phase presence, not optimal w_lr direction. This is the ANIMA analog of c036 (per-step vs sequence-level timescale gap).
- **[c048]** c048: ANIMA Stage 2 vacuous. No Principle-II-compliant per-step signal can detect optimal w_lr. The W-I tension creates an inverted-U at the sequence level, but the information about position on that curve does not exist at the per-step timescale.
- **[c049]** c049: No combination of I_fast, I_slow, I_curvature, or their ratios can locate the w_lr interior optimum via dual-timescale I. All signals show <25% variation across 100x w_lr range. MI-err structural decoupling empirically confirmed.
- **[c050]** c050: W_velocity is monotone increasing in w_lr and is not available as an internal organism adaptation signal. It cannot serve as a Principle-II-compliant Stage 3 signal.
- **[c051]** c051: Additive dual-timescale I reduces MI gap for all tau_slow > 0. Boundary-optimal at tau_slow=0.0. Fast I (tau=0.3) is sufficient; slow I adds misleading temporal blurring that degrades sequence discrimination.

## Current State

**Session:** 23

**Stage:** stage3
**Status:** passed — Session 22: Additive dual-timescale I hurts MI gap — tau_slow boundary-optimal at 0.0

## Key Decisions

**Constitution v2 Established**
  Choice: 5 principles + 8 stages framework with candidate ground truth
  Status: irreversible
  Rationale: Need architecture-independent criteria for recursive self-improvement with monotonic frozen frame reduction

**Accept Beta/Gamma as Frozen, Advance to Stage 3**
  Choice: Accept beta/gamma as irreducible frozen frame elements, focus Stage 3 on alpha adaptation rate
  Status: reversible
  Rationale: Three approaches exhausted: (1) analytical proxy gradient failed, (2) local proxy search failed (7 tested), (3) per-cell decomposition failed (53% loss). Finite-diff MI works but violates Principle II

**Formalize Phased Compute Budget Protocol (Experiment Gate)**
  Choice: Enforce 3-phase validation pipeline before expensive experiments
  Rationale: Sessions 7-8 wasted ~45% compute on 3-seed false positives requiring demolition sessions. Root cause: running Phase 3 (expensive search) before Phase 1 (theory) would have killed candidates. Inverting the order prevents this class of waste.

**Constitutional Amendment 1: Vacuous Stages**
  Choice: Amend constitution to allow stages to be declared vacuously passed when empirical evidence demonstrates the frozen element is not a binding constraint
  Status: reversible
  Rationale: Sessions 6-11 (7 sessions) tested Stage 3 through 5 independent approaches. All converged on the same result: eta can be made adaptive (healthy distributions, non-trivial values) but produces zero measurable performance difference. The original 'do not skip stages' rule creates a deadlock when a stage is empirically vacuous — you cannot pass it (because 'beats fixed' is undefined on a flat landscape) and cannot proceed past it.

**Architecture ceiling declared: Living Seed frozen frame minimum 6/8**
  Choice: Forward viability check (Amendment 2): Can the Living Seed satisfy Stage 7 — represent and modify its own update rule as first-class data? Answer: NO. The update equation phi[k] = tanh(alpha*x + beta*(x[k+1]+gamma*s)*(x[k-1]+gamma*s)) is hardcoded Python. No mechanism exists to represent this equation as data the system can modify. This is not a performance ceiling — the mechanism does not exist. The Living Seed's frozen frame minimum is 6/8. Stage progression halted under Amendment 2.

**ANIMA Stage 2 declared vacuous under Amendment 1**
  Choice: ANIMA Stage 2 (Self-Generated Adaptation Signal) declared vacuously passed under Amendment 1.

## Recent Experiments

**Session 22: Additive dual-timescale I hurts MI gap — tau_slow boundary-optimal at 0.0** [PASSED]

## Historical Summary

**Session 3:** 1 failed
  - Stage 2 Analytical Gradient for Beta/Gamma: FAIL. Proxy gradient does not correlate with true MI landscape. Z-score cancellation (mean=0, var=1) prevents collective signal. NC×D averaging dilutes gradient by factor of 72
**Session 4:** 1 failed
  - Randomized Initialization Test (Task #19): FAIL. Gradient bottleneck: (1) NC×D averaging divides by 72, (2) response×dphi product (~0.0015) is 100× weaker than alpha's resp_z (~1.0)
**Session 5:** 2 failed, 1 partial
  - Local Proxy Search (Basic + Advanced): FAIL. Beta/gamma are global coupling parameters. Local statistics cannot capture global information flow. No local decomposition exists
  - Per-Cell Beta/Gamma Decomposition: FAIL. Beta/gamma are structurally global coupling parameters. Per-cell decomposition destroys the coupling mechanism itself
**Session 6:** 1 failed, 1 partial, 1 passed
  - Stage 3 Exp A: Self-Referential Eta: FAIL. Multiplicative self-reference (push_e ~ eta * f(rz)) creates exponential growth/decay → bang-bang oscillation between clip bounds
**Session 8:** 1 failed, 1 active
  - Stage 3 Delta_rz Fails 10-Seed Validation: FAIL. 
**Session 10:** 1 partial
**Session 11:** 1 partial
**Session 13:** 2 failed, 1 passed, 1 active
  - Stage 4 Phase 1: Threshold non-binding (10-seed validation): FAIL. The 3-seed Phase 1 diagnostic used independent seeds per threshold condition. With CV=37%, independent 3-seed samples cannot distinguish real effects from sampling noise. A paired-seed design would have caught this. The threshold result mirrors eta: the parameter is structurally real (regime selector) but performance is insensitive to its value across the tested range.
  - Stage 4 Phase 1: Clip bounds non-binding (5-seed paired diagnostic): FAIL. Clip bounds define the reachable state space but the system's actual operating range is well within the bounds for most conditions. The lower bound's 15.1% saturation rate does not translate to binding constraint — the system can achieve equivalent MI with different bounds because alpha values near the boundary are not computationally critical for mutual information. The paradox: parameter activity (hitting the bound) ≠ parameter binding (performance depends on value).
**Session 14:** 5 passed
**Session 15:** 1 failed, 2 passed
  - Tau=0.2 is a 5-seed false positive; adaptive tau dead at 10 seeds: FAIL. 5-seed validation at multi-K is still insufficient when dealing with seed-sensitive effects. The original 5 seeds [42,137,2024,999,7] happened to all favor tau=0.2, producing a spuriously strong effect (d=+1.202). This is the same class of false positive as Sessions 6-11 (c010) but at higher seed count — the multi-K variance reduction masked that seed count was still too low for this effect size. Constraint c010 ('5-seed validation insufficient') confirmed yet again.
**Session 16:** 1 failed
  - Adaptive delta Phase 2: divergence signal drives toward 0.16, not 1.0 -- delta=1.0 confirmed calibration-only: FAIL. The divergence signal measures the wrong property of the computation. The distinction: it is not that the signal is weak -- it is that the signal measures per-step state change magnitude, while delta performance depends on per-sequence signal fidelity. These are uncorrelated.

Delta=1.0 is optimal because full replacement maximizes each signal's influence on the final state. With low delta, older signals bleed through and blur the system's ability to distinguish signal sequences -- it is a sequence-level fidelity problem.

The divergence signal norm(p[k]-xs[k]) measures something else entirely: how far the current computation wants to move from current state, per step. At equilibrium this settles to ~0.16 -- the natural scale of per-step state changes in this system -- regardless of what mixing rate would be optimal for MI. tanh(0.16)=0.16, so the leaky integrator finds a fixed point at delta=0.16. Both initializations converge there because it is a property of the computation's dynamics, not of the optimal mixing rate.
**Session 18:** 3 active
**Session 19:** 2 failed, 2 passed
  - ANIMA Stage 2: Gradient w_lr adaptation fails (signal too weak): FAIL. Per-step gradient signal is ~1e-8 magnitude. Would need ~1M steps for meaningful w_lr movement. The signal accumulation rate is 4-5 orders of magnitude too slow for the sequence lengths in the protocol (~1000 steps total).
  - ANIMA Stage 2: Reactive w_lr adaptation fails (5 configs, 10 seeds): FAIL. prev_mean_abs_err is nearly constant (~0.04) regardless of w_lr. The modulation is driven by signal presence vs absence (phase), not by W's learning quality. The signal oscillates between 'signal present → w_lr drops' and 'settling → w_lr rises' without encoding information about whether the current w_lr is optimal for MI gap. This is the ANIMA analog of c036: the signal measures the wrong property (phase, not sequence-level signal fidelity).
**Session 20:** 1 passed
**Session 21:** 1 passed

## Architecture

**ANIMA Organism (src/anima_organism.py)** [ACTIVE SUBSTRATE]
  W+I dynamics: W predicts neighbor interaction, I accumulates prediction error
  Canonical params: w_lr=0.0003, tau=0.3, gamma=3.0, w_clip=2.0, noise=0.005, delta=1.0
  Stage 1 PASS: alive_gap +0.1256 (calibrated w_lr=0.0003, gamma=3.0). Stage 2 vacuous (Amendment 1).

**Living Seed Core Equation** [ARCHIVED — ceiling 6/8]
  phi[k] = tanh(alpha*x + beta*(x[k+1]+gamma*s)*(x[k-1]+gamma*s))
  Archived substrate. Ceiling declared Session 17: frozen frame minimum 6/8. Stage 7 impossible (equation is code, not data). Replaced by ANIMA.

**Experiment Harness (harness.py)**
  Parameterized plasticity rule experiment harness. Self-contained copy of Living Seed core with configurable rule parameters.
  Enables systematic exploration of plasticity rule space without modifying canonical the_living_seed.py

**Search Space Definition (search_space.py)**
  Defines the space of possible plasticity rules and sampling/mutation operations
  Encodes parameter bounds and valid rule configurations for evolutionary search

**Constraint Checker (constraint_checker.py)**
  Validates rule configurations against experimentally-derived constraints
  Filters invalid rule candidates in search loop to prevent repeating known failures

**Canonical delta updated from 0.35 to 1.0: the Living Seed is a memoryless signal processor**
  Updated delta parameter in the_living_seed.py from 0.35 to 1.0 (line 101). The state update equation xs_new = (1-delta)*xs + delta*phi now simplifies to xs_new = phi (pure replacement). State blending is eliminated.

## META-COGNITIVE REVIEW DUE

**Current session:** 23
**Last review:** Session 17
**Sessions since last review:** 6

### Frozen Frame Inventory

**Frozen elements (6):**
  - beta (global coupling)
  - gamma (global self-coupling)
  - symmetry_break_mult
  - amplify_mult
  - drift_mult
  - threshold
  - alpha_clip_lo
  - alpha_clip_hi

**Adaptive elements:**
  - alpha (per-cell, driven by resp_z signal, Stage 2)
  - eta (adaptive-vacuous, Stage 3, Session 12 — non-binding constraint confirmed)

### Stage Progress

| Stage | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | Autonomous Computati | pass | Ground truth test passes: distinguishable states p... |
| 2 | Self-Generated Adapt | partial | Alpha (per-cell) adapts via resp_z signal. Beta/ga... |
| 3 | Adaptation Rate Adap | vacuous_pass | Eta marked adaptive (vacuous). 5 independent appro... |
| 4 | Structural Constants | characterized | Delta binding (+6% at 1.0) but Principle II signal... |
| 5 | Topology Becomes Ada | not_started | Fixed 1D ring connectivity with k=1 nearest neighb... |
| 6 | Functional Form Beco | not_started | Update rule form is fixed (weighted sum of terms).... |
| 7 | Representation Becom | not_started | Equation is code, not data. No self-modification a... |
| 8 | Ground Truth Only Fr | not_started | Not applicable until Stages 3-7 complete. |

### Review Prompt

Perform adversarial progress assessment:
1. Distinguish progress ON the path (frozen frame reduction) from progress BESIDE the path (infrastructure, understanding).
2. What changed since last review? What experiments ran? What failed? What worked?
3. Honest assessment: Are we closer to Stage 3? Or optimizing within Stage 2?
4. Update progress.json: increment last_review_session, append review summary to reviews array.
