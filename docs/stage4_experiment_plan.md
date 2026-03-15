# Stage 4 Experiment Plan — v6
**Session 13 — Structural Constants Become Adaptive**
**Author:** strategist
**Version:** 6 — Architecture review triggered: all structural parameters non-binding
**Status:** ARCHITECTURE REVIEW TRIGGERED. Awaiting team-lead constitutional ruling.

---

## Experiment History (Session 13 so far)

| Task | Target | Result | Constraint added |
|------|--------|--------|-----------------|
| #10 (3-seed diagnostic) | threshold sensitivity | Apparent +20% at threshold=0.1 | — (noise, not yet known) |
| #11 (10-seed validation) | threshold=[0.01,0.02,0.05,0.1] | FLAT. All p>0.5, best d<0.2. +20% was 3-seed noise. | c023: threshold non-binding; c024: independent 3-seed diagnostics unreliable at CV=37% |
| #14 (5-seed paired) | clip bounds sensitivity | NON-BINDING. All p>=0.096. MI range 5.8% of canonical. BORDERLINE: NARROW_TIGHT d=0.610 all 5 seeds above canonical. | c025: clip bounds non-binding; c026: activity≠binding |

**Architecture review trigger: FIRED.** All plasticity rule structural parameters confirmed non-binding. Awaiting team-lead ruling.

**Threshold status:** Non-binding. Like eta before it, threshold is structurally real but
performance is insensitive to its value.

**Amendment 1 (Vacuous Stages) — does it apply to threshold?**
Amendment 1 requires all four criteria. Criterion 2: "The mechanism works — the frozen element
can be made adaptive with non-trivial, non-degenerate behavior." We tested FIXED threshold
values, NOT adaptive per-cell threshold. Criterion 2 is unmet — we cannot declare threshold
vacuously passed. Threshold is simply eliminated as a Stage 4 target (non-binding). The
frozen frame shrinks by zero for threshold. This is not a vacuous pass; it is a kill.

---

## Meta-Review Constraint (unchanged)

**Hard requirement for Stage 4:** Mechanistic progress — adaptive structural parameters that
measurably change what the system computes AND improve the MI gap.

**Hard deadline:** Architecture review if no mechanistic frozen frame reduction after 4 sessions.

**Epistemic progress (proving things don't matter) does not satisfy Stage 4.**

---

## Target Selection — v5

| Parameter | Binding? | Structural? | Stage 4 Role |
|-----------|----------|-------------|-------------|
| alpha_clip_lo (0.3) | UNCONFIRMED — Task #9 evidence: 5% hit rate, narrow bounds constrain state space | YES | **PRIMARY (pending Task #14)** |
| alpha_clip_hi (1.8) | UNCONFIRMED — Task #9: full range used, narrow bounds [0.5,1.0] constrain dynamics | YES | **PRIMARY (pending Task #14)** |
| threshold (0.01) | NO — Task #11, 10 seeds, flat MI | YES structurally | KILLED — not non-binding for Amendment 1, just non-binding |
| amplify_mult (0.5) | YES but scaling | NO (magnitude only) | Stage 6 candidate |
| drift_mult (0.1) | YES but scaling | NO (magnitude only) | Stage 6 candidate |
| symmetry_break_mult (0.3) | WEAK, scaling | NO | Stage 6 candidate |
| beta | BLOCKED | N/A | c001-c007 permanent block |
| gamma | BLOCKED | N/A | c001-c007 permanent block |

**Note on beta/gamma:** Permanently blocked. Per-cell decomposition (c006: 53% loss), local
proxies (c003: r=0.44 best), analytical gradients (c001-c002 fail), finite-diff (c004:
violates Principle II). Do not revisit without a mechanism addressing all seven constraints.

---

## Phase 1 Kill Criterion: Clip Bounds

Before committing to clip bounds as Stage 4 target, answer: does varying clip bounds over
a meaningful range produce measurable MI gap differences?

**Task #14 (running):** Sweep alpha_clip_lo over [0.1, 0.2, 0.3] and alpha_clip_hi over
[1.4, 1.8, 2.4], 10 seeds each (paired), n_perm=8, n_trials=6, K=[4,6,8,10].

**Kill criterion:** If MI gap is flat across the sweep (all p>0.5) → clip bounds are
non-binding → architecture review trigger fires.

**Proceed criterion:** If MI gap varies significantly (at least one condition d≥0.2, p≤0.3)
→ clip bounds are binding → continue to Phase 2.

**Important:** Task #14 must use PAIRED seeds across conditions (c024) — the threshold
validation failure was caused by independent seeds confounded by CV=37%.

---

## Architecture Review — TRIGGERED

**Status: FIRED.** Both structural plasticity parameters (threshold, clip bounds) confirmed
non-binding. All plasticity rule elements enumerated below:

| Parameter | Category | Status |
|-----------|----------|--------|
| threshold (0.01) | STRUCTURAL | NON-BINDING (c023, Task #11) |
| alpha_clip_lo (0.3) | STRUCTURAL | NON-BINDING (c025, Task #14) |
| alpha_clip_hi (1.8) | STRUCTURAL | NON-BINDING (c025, Task #14) |
| amplify_mult (0.5) | SCALING | Stage 6 (functional form) |
| drift_mult (0.1) | SCALING | Stage 6 (functional form) |
| symmetry_break_mult (0.3) | SCALING | Stage 6 (functional form) |
| eta (0.05) | ADAPTATION RATE | VACUOUS (Stage 3, 7 sessions, c011) |
| beta | COUPLING | BLOCKED (c001-c007) |
| gamma | COUPLING | BLOCKED (c001-c007) |

**All plasticity PARAMETERS are non-binding.** The binding constraint is in the rule FORM.

### Three Paths — Team-Lead Must Rule

**Path A: Amendment 1 (Vacuous Stage 4)**
- Require: run adaptive per-cell clip bounds to satisfy Criterion 2 (mechanism works with
  non-trivial behavior). Then declare Stage 4 vacuously passed.
- Costs: ~1 session (mechanism test only, no performance expectation).
- Closes the constitutional loop cleanly before advancing.
- Question: then Stage 5 or Stage 6?

**Path B: Skip to Stage 6 (Functional Form)**
- Justification: All structural PARAMETERS non-binding. Binding constraint is the rule FORM.
- Requires skipping Stage 5 (topology) via Amendment 1 (topology not yet tested, but
  strong theoretical basis that it is also non-binding if parameters are non-binding).
- Problem: Double-skipping (Stage 4 + Stage 5) without mechanism tests may create
  unvalidated assumptions. The constitution warns against this.

**Path C: Novel Stage 4 Target**
- No clean Stage 4 target exists outside beta/gamma (blocked) and functional form (Stage 6).
- The branching rule structure (3 branches) could be Stage 4 or Stage 6 — ambiguous.
- Verdict: Path C is effectively empty unless team-lead identifies a new structural element.

### Borderline Finding: NARROW_TIGHT [0.5, 1.6]

Engineer flagged NARROW_TIGHT [0.5, 1.6] as borderline (d=0.610, all 5 seeds above
canonical, p=0.096, CV=14.8%). Critically: CV=14.8% is much better than threshold's 37%,
so this is more credible than threshold's false positive, but still subthreshold at n=5.

**Analysis:** Narrower alpha range may improve MI by forcing cells to discriminate within
a compressed space. But this is the OPPOSITE of Stage 4 structural adaptation — it shows
a better fixed value exists (calibration), not that adaptive bounds help. If genuine,
update canonical baseline to [0.5, 1.6]; do not use as Stage 4 mechanism.

**Team-lead decision needed:**
1. Pursue 10-seed NARROW_TIGHT validation (~30 min, low cost, CV=14.8% means n=10 is
   probably definitive)? If confirmed, updates canonical baseline.
2. Skip validation, proceed directly to architecture review ruling?

Team-lead must rule on both the NARROW_TIGHT question AND the architecture review path
before the next experiment is run.

---

## Adaptation Signal Design (for clip bounds)

### Primary Signal: Alpha Trajectory Utilization

```
For each cell (i, k) at each adaptation step:

alpha_hi_proximity_k = (alpha[i][k] - alpha_clip_lo) / (alpha_clip_hi - alpha_clip_lo)
  → 0.0 means alpha is at lower bound, 1.0 means at upper bound

recent_max_k = max(alpha[i][k] over last W steps)
recent_min_k = min(alpha[i][k] over last W steps)

Upper bound signal:
  if recent_max_k > 0.9 * alpha_clip_hi:   # alpha is hitting upper bound
      clip_hi[i][k] += alpha_meta_s4 * (clip_hi_max - alpha_clip_hi)  # expand upward
  elif recent_max_k < 0.7 * alpha_clip_hi:  # alpha never reaches upper bound
      clip_hi[i][k] -= alpha_meta_s4 * (alpha_clip_hi - recent_max_k)  # contract downward

Lower bound signal (symmetric):
  if recent_min_k < alpha_clip_lo + 0.1 * (alpha_clip_hi - alpha_clip_lo):
      clip_lo[i][k] -= alpha_meta_s4 * (alpha_clip_lo - clip_lo_min)   # expand downward
  elif recent_min_k > alpha_clip_lo + 0.3 * (alpha_clip_hi - alpha_clip_lo):
      clip_lo[i][k] += alpha_meta_s4 * (recent_min_k - alpha_clip_lo)  # contract upward
```

**Principle II check:** Signal uses only alpha[i][k] trajectory — the system's own state.
No external measurement. **Passes.**

**c013 check:** Alpha trajectory utilization is NOT in the resp_z family. It operates on
alpha state, not resp_z. No c013 risk.

**No positive feedback loop:** Signal is stabilizing — when alpha hits bounds, bounds expand;
when alpha stops hitting bounds, bounds contract. This is negative feedback around
the current utilization level, not positive feedback toward extremes.

**Non-tautology:** Adaptation is bidirectional (expand if hitting, contract if not).
Upper and lower bounds adapt independently per cell. Non-degenerate when cells have
different alpha trajectories.

### Signal Fallback

**resp_z × sign(dev)** — original threshold signal. Can be repurposed:
- High resp_z AND alpha near upper bound → expand clip_hi (cell wants to go higher)
- Low resp_z AND alpha near lower bound → contract clip_lo (cell is stuck at bottom)
Less mechanically clean than utilization signal but usable as fallback.

---

## Frozen Frame Accounting

### Clip bounds (Phase 2 primary):
- Remove: alpha_clip_lo, alpha_clip_hi (2 elements — become per-cell adaptive state)
- Reuse: eta as meta-rate (vacuous, already in adaptive state — not new frozen element)
- Add: alpha_meta_s4 = adaptation step size (+1 element)
- Add: per-cell clip bounds limits (clip_lo_min, clip_lo_max, clip_hi_min, clip_hi_max)
  **Derived argument:** Alpha bounds derive from the tanh equation form:
  - alpha < 0: flips sign of computation (degenerate — excluded by physics)
  - alpha > ~3: tanh saturates, additional alpha has diminishing return
  - Meaningful range: [~0.1, ~3.0] is fully derived from equation structure
  - Per-cell bounds [clip_lo_min=0.05, clip_lo_max=0.5, clip_hi_min=1.0, clip_hi_max=3.0]
    are subsets of this derived range — count as +0 by same argument as threshold bounds
- Net: **-2 + 1 = -1 element.** Principle IV strictly satisfied.

**If derived argument rejected:** Need to remove an additional frozen element simultaneously.
Options: also kill threshold from frozen frame explicitly (it was already not adaptive, so
removing it costs nothing), or accept that clip bounds for adaptive clip bounds are frozen.

---

## Phase 2: Single-Candidate Validation (if Task #14 passes)

**Four conditions, 10 seeds, paired across conditions:**

| Condition | clip_lo | clip_hi | Adaptation | Purpose |
|-----------|---------|---------|------------|---------|
| canonical | 0.3 (fixed, global) | 1.8 (fixed, global) | None | Baseline |
| optimal_fixed | best from Task #14 sweep (global) | best from Task #14 | None | Best fixed beats adaptive? |
| hetero_fixed | per-cell, random at birth from valid range | per-cell random | None | c006 analog — variation alone? |
| s4_adaptive | 0.3 start (per-cell) | 1.8 start (per-cell) | Alpha utilization signal | Stage 4 candidate |

**Decision ladder:**

| Result | Interpretation | Action |
|--------|---------------|--------|
| s4_adaptive > canonical (d≥0.3) AND > hetero_fixed | Full proof: adaptation, not just diversity | Phase 3 |
| s4_adaptive ≈ optimal_fixed > canonical | Found better fixed bounds, adaptation vacuous | Document, Amendment 1 criteria check |
| s4_adaptive ≈ hetero_fixed > canonical | Diversity helps, signal doesn't add | Try fallback signal |
| hetero_fixed << canonical | Per-cell variation itself harmful | c006 analog confirmed general — abort per-cell approach |
| All conditions ≈ canonical | Clip bounds non-binding via adaptation | Architecture review trigger |

**Evaluation:** n_perm=8, n_trials=6, K=[4,6,8,10], 10 seeds (c015-c019). Paired seeds.

---

## Mechanistic Progress Tests

**Test 1: Per-cell bound divergence**
After adaptation, compute variance of per-cell clip_hi and clip_lo across cells.
- If variance < 5% of mean: all cells converged to same bounds → effectively global shift, possibly vacuous
- If variance > 20% of mean: cells genuinely differentiated → structural specialization

**Test 2: Bound-utilization correlation**
Cells whose alpha frequently hits clip_hi should have LARGER clip_hi than cells that rarely hit it.
Pearson correlation between (recent_max_alpha[i][k] / canonical_clip_hi) and adapted clip_hi[i][k].
Requirement: r > 0.4 for mechanism to be non-random.

**Test 3: Alpha specialization delta**
s4_adaptive should produce MORE differentiated alpha profiles (lower inter-cell cosine) than canonical.
If inter-cell alpha cosine ≥ 0.9 for s4_adaptive: no specialization produced.

**Test 4: Novel signal generalization**
MI advantage must hold on signals not seen during adaptation.

---

## Three-Phase Protocol

### Phase 1 (running — Task #14):
- Kill criterion: does clip bound variation produce measurable MI differences?
- PENDING Task #14 results.
- Phase 1 gate: proceed if at least one bound configuration shows d≥0.2.

### Phase 2 (if Phase 1 passes):
- Four conditions, 10 seeds paired, full evaluation protocol.
- ~1.5–2 hours.
- Parameters: alpha_meta_s4=0.05, W=50 (window for recent min/max), per-cell bounds start at canonical.

### Phase 3 (if Phase 2 passes):
Search space:
```
alpha_meta_s4:     [0.001, 0.2] log-scale (4 values)
W_window:          [20, 50, 100] (3 values)
clip_hi_max:       [2.5, 3.0] (2 values)
clip_lo_min:       [0.05, 0.1] (2 values)
```
Total: 48 candidates — subsample to 30.
Evaluation: 10 seeds, paired, n_perm=8, n_trials=6.

---

## Fallback Sequence

**If Task #14 shows clip bounds also non-binding:**

Architecture review trigger fires. Team-lead determines:
- Option A: Declare Stage 4 vacuous (Amendment 1) — requires running adaptive clip bounds
  to verify mechanism works (Criterion 2), even if no performance improvement expected.
- Option B: Skip to Stage 6 (functional form) — requires constitutional ruling on skipping
  Stage 5 (topology). Justification: if all structural PARAMETERS are non-binding, the
  binding constraint is the functional FORM, not the topology.
- Option C: Novel mechanism for Stage 4 — requires proposing a structural element not yet
  identified. Currently no candidates exist.

**If Phase 2 fails (hetero_fixed << canonical):**
c006 analog confirmed as general principle: per-cell variation of any parameter is harmful.
This is a new architectural constraint. Document, ingest, do not retry per-cell approach.

---

## Success and Failure Criteria

### Stage 4 PASSES (mechanistic) if ALL:
1. s4_adaptive > canonical (d ≥ 0.3, p ≤ 0.20, 10 paired seeds)
2. s4_adaptive > hetero_fixed (adaptation adds value beyond diversity)
3. Per-cell bound variance > 20% of mean after adaptation (genuine specialization)
4. Bound-utilization Pearson r > 0.4 (mechanism is responsive to actual utilization)
5. Ground truth passes (3/3 initializations)
6. Inter-cell alpha cosine < 0.9 after adaptation (Stage 4 specialization criterion)
7. Advantage holds on novel signals

### Stage 4 VACUOUSLY PASSES if (Amendment 1):
1. Multiple independent approaches tested (utilization signal + resp_z fallback)
2. Mechanism works: per-cell bounds adapt, non-trivial, non-degenerate
3. Zero measurable performance difference, 10+ paired seeds
4. Theoretical explanation why adaptive bounds are non-binding

### Architecture review triggered if:
- Task #14 confirms clip bounds ALSO flat (architecture review path described above)
- OR hetero_fixed << canonical (per-cell variation generally harmful)

---

## Compute Budget

| Phase | Description | Runs | Estimate |
|-------|-------------|------|---------|
| #14 | Clip bounds MI sweep (3×3 grid, 10 seeds) | 90 | 1.5–2 hours |
| 2 | Phase 2 (4 conditions × 10 seeds) | 40 | 1.5–2 hours |
| 3 | Phase 3 search (30 candidates × 10 seeds) | 300 | 6–10 hours |
| Fallback | Architecture review / Amendment 1 mechanism test | — | team-lead decision |

---

## Implementation Notes for Engineer (updated for clip bounds)

### Harness.py changes required:

1. Add `per_cell_clip_lo` and `per_cell_clip_hi` fields (NC×D arrays, None = use global)
2. Modify plasticity clip line:
   ```python
   lo = self.per_cell_clip_lo[i][k] if self.per_cell_clip_lo else 0.3
   hi = self.per_cell_clip_hi[i][k] if self.per_cell_clip_hi else 1.8
   self.alpha[i][k] = max(lo, min(hi, self.alpha[i][k] + push))
   ```
3. Add alpha trajectory tracking: per-cell running min/max over window W
4. Add `stage4_enabled` flag, `alpha_meta_s4`, `W_window` parameters
5. Add Stage 4 adaptation block AFTER existing plasticity block:
   ```python
   if self.stage4_enabled:
       recent_max = self.alpha_max_window[i][k]  # tracked rolling max
       recent_min = self.alpha_min_window[i][k]  # tracked rolling min
       hi = self.per_cell_clip_hi[i][k]
       lo = self.per_cell_clip_lo[i][k]

       # Upper bound adaptation
       if recent_max > 0.9 * hi:
           self.per_cell_clip_hi[i][k] = min(self.clip_hi_max,
               hi + self.alpha_meta_s4 * (self.clip_hi_max - hi))
       elif recent_max < 0.7 * hi:
           self.per_cell_clip_hi[i][k] = max(self.clip_hi_min,
               hi - self.alpha_meta_s4 * (hi - recent_max))

       # Lower bound adaptation (symmetric)
       lo_range = hi - lo
       if recent_min < lo + 0.1 * lo_range:
           self.per_cell_clip_lo[i][k] = max(self.clip_lo_min,
               lo - self.alpha_meta_s4 * (lo - self.clip_lo_min))
       elif recent_min > lo + 0.3 * lo_range:
           self.per_cell_clip_lo[i][k] = min(self.clip_lo_max,
               lo + self.alpha_meta_s4 * (recent_min - lo))
   ```

### New experiment scripts:
- `src/stage4_exp_phase2_clips.py` — four-condition Phase 2 (clip bounds)
- `src/stage4_exp_phase3_clips.py` — grid search (write only if Phase 2 passes)

### search_space.py additions:
```python
'alpha_meta_s4':  {'min': 0.001, 'max': 0.2,  'canonical': 0.05, 'log_scale': True},
'W_window':       {'min': 20,    'max': 100,   'canonical': 50,   'log_scale': False},
'clip_hi_max':    {'min': 2.0,   'max': 3.0,   'canonical': 2.5,  'log_scale': False},
'clip_lo_min':    {'min': 0.05,  'max': 0.15,  'canonical': 0.1,  'log_scale': False},
'stage4_enabled': {'default': False},
```

---

## Open Questions

**For team-lead — pre-empting architecture review:**

If Task #14 shows clip bounds also non-binding, we need a ruling on path forward BEFORE
running Amendment 1 mechanism tests. Two key questions:

1. Does Amendment 1 require running adaptive clip bounds even if theory predicts vacuous?
   (Criterion 2 requires mechanism demonstration — can we skip it if theory is clear?)

2. If Stage 4 is declared vacuous, can Stage 5 (topology) be skipped on the same grounds?
   Stage 5 is a fixed Moore neighborhood (8-cell). Topology change requires rewiring, which
   changes what cells interact — genuinely structural. But it has never been tested.
   Constitutional ruling needed before skipping two stages simultaneously.
