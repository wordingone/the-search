# Direction 2: Unconstrained Diagnostic

*Jun directive 2026-03-24. ALL bans lifted, ALL constraints suspended, ALL constitutional limits removed. Purpose: capability ceiling — can we solve EVERY LEVEL of EVERY preview game using everything available?*

**This is NOT a substrate search.** It's a calibration audit. Per-game prescription is allowed. Manual encoding is allowed. Source analysis is allowed. The question: what is the absolute ceiling?

**Gate suspension notice for Eli:** When this spec is active, the following gates are SUSPENDED:
- Codebook ban (Step 416)
- Graph ban (Step 777)
- Per-game tuning prohibition
- One-config rule
- R1 (reward/loss signals allowed)
- R3 requirement
- 5-minute budget cap (extended to 30 min per game for multi-level)
- PRISM randomization (per-game tuning = each game gets its own config)

**Still active:** Experiment must be reproducible (same code + same seeds = same result). Results must be honest.

---

## Sub-mode A: Per-Game Tuned Entities (ceiling per game)

### LS20 — Target: ALL levels (L1-L5+)

**Known:** L1-L3 solved (Steps 572j-572u). L4+ blocked at puq_wall_set bootstrap (Steps 611-612).

**Substrate:** 674+running-mean (frozen bootloader) + mgu pipeline:
- Level-aware background modeling
- Robust object detection via isolated CC
- Hidden-state coverage (POMDP, |S| ≤ 96)
- Dead reckoning + state estimation + sequencing
- puq_wall_set fix for L4+ transition

**Budget:** 30 min per attempt. 5 seeds.
**Expected:** L1-L3 confirmed. L4+ depends on puq_wall_set fix.

### FT09 — Target: ALL 6 levels

**Known:** All 6 levels solved (Step 608b). 75 clicks total, deterministic.

**Substrate:** Codebook + 69-action graph (64 grid + 5 simple) + argmin + prescribed encoding (avgpool16 centered). 674+running-mean for L1 reliability.

**Budget:** 30 min. 5 seeds.
**Expected:** 6/6 levels, all seeds. Previously confirmed deterministic.

### VC33 — Target: ALL 7 levels

**Known:** All 7 levels solved (Step 610). 176 clicks, analytical BFS.

**Substrate:** Mode map + CC zone discovery + analytical BFS path planning. Camera offset correction. Per-level state estimation.

**Budget:** 30 min. 5 seeds.
**Expected:** 7/7 levels, all seeds. Previously confirmed deterministic (0.15s total).

### CIFAR — Target: Maximum classification accuracy

**Known:** 94.48% with external labels (Step 425). 36.2% self-labels (Step 573).

**Substrate:** Codebook + softmax voting + external labels (R1-violating for ceiling).

**Budget:** Standard P-MNIST protocol.
**Expected:** ~94% with labels, ~36% without.

---

## Sub-mode B: Single Entity Through PRISM (unified ceiling)

**The hard question:** Can ONE substrate, with per-game configuration allowed, solve ALL games in randomized order through PRISM?

**Substrate design:**
```
Phase detection:
- On game start: 1000 random steps to profile the game
- If n_actions <= 8 AND frame changes significantly per action → directional game (LS20-type)
- If n_actions > 8 OR frame barely changes → click game (FT09/VC33-type)
- If input is images without game mechanics → classification (CIFAR-type)

For directional games:
- 674+running-mean + graph+argmin
- 4 directional actions

For click games:
- CC zone discovery → action space identification
- Graph+argmin on discovered zones
- 69 actions (64 grid + 5 simple) as fallback

For classification:
- Codebook + softmax voting
- External labels if available, self-labels otherwise

Domain separation: per-domain centering (reset running mean on game switch)
```

**Through PRISM:** Randomized game order. Chain score as metric.
**Budget:** 10K steps per game (standard PRISM). 5 seeds.
**Expected:** CIFAR=100% (bootloader), LS20=20/20 (674+rm), FT09=L1 (69-action graph), VC33=L1 (CC+graph). Chain score: 5/5 L1.

---

## What the diagnostic measures

| Metric | Sub-mode A (per-game) | Sub-mode B (PRISM) | Current constrained |
|--------|----------------------|-------------------|-------------------|
| LS20 L1 | 20/20 (expected) | 20/20 (expected) | 10/10 (100%) |
| LS20 L2+ | L3 confirmed, L4+ open | L1 only (10K budget) | 0 |
| FT09 L1 | 5/5 (expected) | L1 (expected) | 0/10 (0%) |
| FT09 L2+ | 6/6 (confirmed) | L1 only | 0 |
| VC33 L1 | 5/5 (expected) | L1 (expected) | 0/10 (0%) |
| VC33 L2+ | 7/7 (confirmed) | L1 only | 0 |
| CIFAR | 94% (labels) | 100% (bootloader) | 100% (bootloader) |
| Chain score | N/A | 5/5 (expected) | 3/5 |

**The gap = the cost of constraints:**
- FT09: 0% → 100% = graph ban + prescribed encoding cost
- VC33: 0% → 100% = graph ban + CC discovery + prescribed zones cost
- LS20: 100% → 100% = no cost (post-ban mechanism matches pre-ban)
- Multi-level: 0 → 16+ levels = temporal credit + per-game pipeline cost
