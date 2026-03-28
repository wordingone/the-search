# Underexplored Directions Catalog

*Created 2026-03-28 by Jun. Full git history audit (1554 commits). Reference before pursuing any "new" direction.*

## Components cataloged as high-priority but never tested in current context

1. **C23 - Mode map + CC zone discovery** (Step 576). "HIGHEST EXTRACTION PRIORITY." Only autonomous multi-game discovery mechanism. VC33 5/5, FT09 5/5 (Step 1023b). D2 pipeline built on this. Dropped at composition era. **0 experiments in current substrate.**
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
14. **Meta-learned Hebbian rules** (Najarro & Risi 2020) - Hebbian RNN kill. **Never explored.**
15. **Decorrelated representations from recurrent dynamics** - Hebbian W_a return condition. **Never tested.**

## Directions that produced signal and were dropped

16. **D2 pipeline (WHERE→HOW→WHEN→ACT)** - Steps 1023-1035. Mode map discovery works without graph (15/15 VC33, 5/5 FT09). Most successful autonomous discovery mechanism. **Dropped at composition era.**
17. **MI-detected reactive (v67)** - Step 1161. ARC=0.2000, GAME_2 SOLVED 100%. Highest ARC score in debate. **1 experiment, never replicated.**
18. **Change-rate maximizing (v80)** - Steps 1186-1188. Best debate mechanism: 3.3/5. **3.3/5 ceiling never explained.**
19. **Allosteric softmax** - Step 1253. R3+I3 coexisted on VC33. Only selector that reads from W. **1 experiment. Never revisited with eta_h=0.05.**
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
