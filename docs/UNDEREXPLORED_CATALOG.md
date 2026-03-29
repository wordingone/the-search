# Underexplored Directions Catalog

*Created 2026-03-28 by Jun. Updated 2026-03-29 by Leo (Steps 1334-1348). Full git history audit (1554 commits). Reference before pursuing any "new" direction.*

## Components cataloged as high-priority but never tested in current context

1. **C23 - Mode map + CC zone discovery** (Step 576). VC33 5/5, FT09 5/5 (Step 1023b). **TESTED Step 1338: KILLED.** Zones are spatial (episode-specific), not functional. try1 zones [3,4,4,3] → try2 [0,0,0,0]. Discovery works but discovered structure doesn't transfer.
2. **C30 - Sparse gating (relu threshold)** (Step 959). Dissolves positive lock (Prop 30). Listed as imaginal disc in metamorphosis audit. **Never tested in any complete substrate.**
3. **C33 - Attention retrieval over trajectory buffer** (Step 1007). 1 experiment. Bootstrap failure. Catalog: "needs fixing, not killing." **Never retried.**
4. **C7 - Change-rate detection** (Step 400). Found LS20 active regions autonomously. Era 1 extraction priority #1. **Never combined with post-ban substrates.**
5. **C22 - Self-observation** (Steps 620-629). Only tested as frozen random projection. **Never tested with learned self-observation.**

## Families killed below 20-experiment minimum

6. **Hebbian RNN** - 6 experiments. "Severely under-explored." Return condition: decorrelated representations. **Never tested.**
7. **Attention-trajectory** - 1 experiment. **Never retried.**
8. **Oscillatory** - 3 experiments. Return: "revisit if W_in is learned." **Never revisited.**
9. **GFS** - 3 experiments. Prop 27 falsified within 916 base only. **Never tested outside 916.**
10. **Obs preprocessing** - 2 experiments. **Never tested under different base.**
11. **Multi-horizon** - 2 experiments. Return: "revisit with attention over trajectory buffer" (→ C33).

## Kill file return conditions never tested

12. **Competitive inhibition / anti-Hebbian normalization** - Hebbian W_a kill (Step 960). The fix for positive lock. **Still untested as specified.**
13. **Non-linear action scoring** (distance-based, attention-style) - Same kill file. **Never tested.**
14. **Meta-learned plasticity rules** (Najarro & Risi 2020) - **TESTED Steps 1325, 1339.** Step 1325: theta found correct direction (anti-Hebbian + decay) but credit formula biased. Step 1339: theta frozen — credit signal 2e-9 per update (too small). Fix identified: normalized credit + RMS tracking + eta=0.01. **Underexplored — mechanism correct, calibration wrong.**
15. **Decorrelated representations from recurrent dynamics** - Hebbian W_a return condition. **Never tested.**

## Directions that produced signal and were dropped

16. **D2 pipeline (WHERE→HOW→WHEN→ACT)** - Steps 1023-1035. Mode map discovery works without graph (15/15 VC33, 5/5 FT09). **Partially tested Step 1338** (mode map component). Zone discovery works but zones are spatial/episode-specific. Full D2 pipeline (WHERE+HOW+WHEN+ACT) never rebuilt in current substrate.
17. **MI-detected reactive (v67)** - Step 1161. ARC=0.2000, GAME_2 SOLVED 100%. Highest ARC score in debate. **1 experiment, never replicated.**
18. **Change-rate maximizing (v80)** - Steps 1186-1188. Best debate mechanism: 3.3/5. **3.3/5 ceiling never explained.**
19. **Allosteric softmax / adaptive temperature** - Step 1253. R3+I3 coexisted on VC33. Only selector that reads from W. **Specced as Step 1339 (old) but deprioritized for meta-plasticity. Concept revisited in deliberation context (Jun, 2026-03-29): prediction confidence → action precision. Never implemented on MLP+TP.**
20. **Empowerment** - Step 1152. ARC=0.004, 3/5 L1. **4 experiments, killed. Only tested within 800b-era.**

## Theoretical directions never experimentally tested

21. **Eigenoptions / SR on W_action** - Eli mail 3605. W_action is empirical SR. **Never implemented.**
22. **Infant causal learning** (Gopnik 2024). **Never became experiment.**
23. **Causal discovery from intervention** (Eberhardt 2008). **Never tested.**
24. **Renormalization group / L2 wall as phase transition.** **Never tested.**
25. **Immune system R3 / SHM parallels.** **Never pursued.**
26. **Termite stigmergy → Prop 13.** **Never tested.**
27. **Retinal contrast adaptation → Prop 15.** **Never tested.**
28. **Quorum sensing → Prop 28.** **Never tested.**
29. **Hippocampal place cells → positive lock solution.** **Never tested.**
30. **Prop 18 (eigenform reactivation via encoding).** **Full test never done.**
31. **Prop 24 (active inference).** Killed within 800b only. **Never tested outside 800b.**

## Structural gaps

32. **Self-directed pruning / U3 revisitation.** **Never tested.**
33. **Activity-dependent growth.** **Never tested.**
34. **STDP / three-factor plasticity.** Active direction as of 2026-03-28.
35. **178 experiments without PRISM** (Steps 778-963). **Chain interactions never backfilled.**
36. **"Does the substrate understand what a game is?"** **No experiment has measured internal task-structure representation.**
37. **5 untested MAP.md gap analysis items.** Per-cell action discrimination, 674+raw 64x64, 674 on benchmarks, clean Recode, non-argmin under running-mean. **All untested.**
38. **avgpool4 encoding may be Phase 1 artifact** (Leo, 2026-03-28). Validated under argmin (which doesn't need spatial structure). StochasticGoose (leaderboard leader) uses 2D conv that PRESERVES spatial structure. avgpool4 DESTROYS it (64×64 → 4×4 → flatten → 256D). If the substrate needs spatial awareness for efficient click-game action selection, the encoding is the bottleneck. **Premature to test before forward model gets first experiment.**
39. **Action hierarchy (type → position)** (from StochasticGoose analysis). Our substrate treats all 4103 actions as flat vector. StochasticGoose decomposes: 5 action types, then 64×64 click position. Hierarchical action decomposition = smaller search space. **Never explored.**
40. **Level transition detection** (from StochasticGoose analysis). StochasticGoose resets experience buffer on level transitions = explicit phase awareness. Our substrate has never detected level transitions. **Connected to Jun's "does the substrate understand what a game is?" (#36).**

## New directions discovered (Steps 1334-1348, 2026-03-29)

41. **Deliberation / action decimation** (Jun insight, 2026-03-29). RHAE counts actions not steps — internal simulation is free. Act every K steps instead of every step. K=10 tested (Step 1346). **Running.**
42. **Action-conditional forward model** (Leo, 2026-03-29). Current model is action-blind (predicts same next-state for all actions). Conditioning on action enables model-based selection. Tested Steps 1342-1343 with lossy action encoding → novelty_var≈0. Rich encoding (type+x+y) showed MBPP signal. **Partially tested, ARC encoding issue identified.**
43. **Equilibrium propagation + learned halting** (literature review, 2026-03-29). R2-compliant deep credit via energy-based settling. Substrate decides when to act. Novel combination — not in literature. **Never tested.**
44. **Childhood / multi-game pretraining** (Leo, 2026-03-29). Train on 10+ games before evaluation. Accumulates prior knowledge. Breaks the circle: can't solve games without experience, can't get experience without solving games. **Step 1348 specced, contingent on 1346+1347 failing.**
45. **MLP modality-agnostic encoder** (Step 1337). Replaces CNN. Processes both ARC pixels and MBPP text. cr=0.08 MBPP (first text compression). **CONFIRMED. CNN frozen frame for text.**
46. **Normalized meta-plasticity credit** (Eli, 2026-03-29). Fix for Step 1339 theta freeze: relative credit (loss_before-loss_after)/loss_before instead of absolute. Predicted delta_theta=1.25e-3 per window (visible). **Identified, not yet tested.**

## 11 Signal moments — how many became directions?

| Step | Signal | Experiments | Outcome |
|------|--------|------------|---------|
| 576 | mode map VC33 5/5 | → D2 pipeline | **Dropped at composition era** |
| 954 | ESN alive | → 916 family | Became frozen fixed point |
| 1092 | forward model 30% L1 | 6 more | Killed |
| 1146 | state-conditioned action | 1 | Draw variance |
| 1152 | empowerment ARC=0.004 | 4 | Killed |
| 1157 | action-space-adaptive | 1 | Not followed up |
| 1161 | MI-detected ARC=0.200 | 1 | Not replicated |
| 1177 | patient hold 4/5 | ~3 | Draw variance → killed |
| 1180 | multi-timescale 4/5 | ~3 | Draw variance → killed |
| 1186 | change-rate max 4/5 | ~10 | 3.3/5 ceiling → "learning adds zero" → closed |
| 1252 | first I4 signal | → composition era | **Current** |

**2/11 signals became real directions. 9 tested 1-4 times and dropped.**
