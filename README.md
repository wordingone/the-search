# The Search

*Part of the research and design work collected at https://wordingone.github.io/.*

> **At a glance (read this first)**
>
> **What it is.** A logged research record (March to June 2026, 1,000+ experiment steps) asking one question: can a learning system improve itself by criteria it generates, with no external objective? Candidate systems were tested against seven written rules (R0-R6, [`docs/CONSTITUTION.md`](docs/CONSTITUTION.md)), first on ARC-AGI-3 interactive games and MBPP (Era I), then on ARC-AGI-1 program synthesis (Era II).
>
> **Status.** Concluded (Era II was marked concluded 2026-06 in [`docs/STAGE_MAP.md`](docs/STAGE_MAP.md)); last commit to the research itself 2026-06-01. Results are mostly negative. Nothing here is peer reviewed. The research question itself is recorded as **open**: the constitution lists R3 (self-modification) as holding for weights and open for structure ([`docs/CONSTITUTION.md`](docs/CONSTITUTION.md), R3 row). Older text that says "R3 solved" (e.g. [`docs/PAPER.md`](docs/PAPER.md) title) is superseded; see the note at the top of that file.
>
> **Who wrote it.** I designed and directed this work; most of the code was written by AI coding agents under my review. The paper's own disclosure (Author Attribution and Disclosure, [`docs/paper.tex`](docs/paper.tex) line 1830) says LLM agents (Claude Opus and Claude Sonnet) implemented and ran the experiments, and that the per-experiment review was done by the agents, not by independent humans.
>
> **Key measured results** (each is a file you can open):
> - *Era I, Step 1379:* the state-space model's prediction loss was 0.796971 with action inputs and 0.796955 with them masked, ratio 1.00002, verdict `ACTION_BLIND`: the action input had no measurable effect on what the model predicted. Source: [`experiments/results/results_all/results_1379/summary.json`](experiments/results/results_all/results_1379/summary.json) (`action_blind_diagnostic`). Across Era I, the best learned score under the rules was about 1e-4 RHAE(try2), and no trained action head beat random action selection (README findings 15-21 below).
> - *Era I, Step 1251:* the "composed" system passed its self-modification test (R3_pass true in 130 of 145 run files) by a provisional Jacobian threshold, but its state-distinguishing check (`I1_pass`) passed in 0 of 145, so this was not shown to help the task. Source: [`experiments/results/results_all/results_1251/`](experiments/results/results_all/results_1251/), threshold at [`experiments/steps/step1251_composition.py`](experiments/steps/step1251_composition.py) line 844.
> - *Era II, reach baseline:* a program search over a small grid DSL solved 10 of 200 held-out ARC-AGI-1 tasks at its best setting (depth 3, budget 300K, 600 s per task); raising depth, budget and time each added at most one task, then saturated. Source: [`docs/STAGE_MAP.md`](docs/STAGE_MAP.md), "Era II - Reach ladder", and release [`stage-1k-reach-baseline`](https://github.com/wordingone/the-search/releases/tag/stage-1k-reach-baseline).
> - *Era II, last gate:* the grammar self-modification pilot (Stage F3, Part A) failed its pre-registered gate: median reduction 0.0, 2 of 10 tasks improved, 9/10 solved with chunks vs 8/10 without. Source: [`docs/STAGE_MAP.md`](docs/STAGE_MAP.md) four-faces table and commit `8ddca36`. Part B is not reported.
> - *Transfer:* a learned recognition prior gave a warm-minus-cold gain of +0.000; the oracle ceiling with the available seed vocabulary was about 0.7%. Source: [`docs/RESEARCH_STATE.md`](docs/RESEARCH_STATE.md), "R4 Decomposition" section.
>
> **What is not shown.** No system under the rules improved itself on tasks it was not designed for. The recorded kills ([`docs/kills/`](docs/kills/), the "Mechanism Kill Log" in [`docs/RESEARCH_STATE.md`](docs/RESEARCH_STATE.md)) are the main output.
>
> **How to run.** Python 3.10+, `pip install -r requirements.txt`, or `bash experiments/setup.sh`. Then, for a current-era step (1305+): `PYTHONUTF8=1 python experiments/steps/step1392_dendritic_spatial_stdp.py`; details in [`experiments/README.md`](experiments/README.md). Caveats: the ARC-AGI-3 SDK (`arc_agi`, `arcengine`) is not on PyPI and needs an API key (see `requirements.txt`); steps before 1305 import a `substrates/` package that is not in this repo and will not run as-is; some scripts contain absolute Windows paths from the original machine. Era II code is in `incoming/arc-agi1-visa/`.
>
> **Sources for this summary** are the repo files linked above, read at `main` (commit `0c1a964`). Everything below this box is the original README, written at about Step 1381, with the places that described work as current now labelled as of that time (2026-11-05); the dated record in `docs/RESEARCH_STATE.md` and `docs/STAGE_MAP.md` takes precedence.

---

Experimental record of recursive self-improvement research on ARC-AGI-3 and MBPP (Era I). The sections below are a snapshot from about Step 1381 and do not cover Era II.

---

## Question

Can a system improve itself by criteria it generates?

## Feasibility

One known system satisfies all seven constitutional constraints simultaneously: biology. Cells self-modify (R3) through metabolic dynamics that ARE the computation (R2), without external objectives (R1). Environmental selection tests modifications against prior fitness (R4). DNA provides the fixed ground truth (R5). Every organelle is essential (R6). Identical genomes in different environments produce different organisms (R0).

This is not a theoretical exercise. The constitution is extracted from observing what already works. The open question is whether computation (discrete, finite precision) can implement what chemistry (continuous, molecular) does. A single computational substrate satisfying R0-R6 with RHAE > 0 would resolve this.

## Metric

**RHAE(try2) = mean(efficiency²)** across all games, measured on try2 (with weights from try1).

ARC Prize scoring. External judgment. The substrate plays each game twice: try1 (fresh weights), try2 (carrying try1's weights). MBPP (text) always in the game pool. efficiency = optimal_actions / actual_actions. efficiency = 0 when progress never reached.

**Best RHAE(try2) from an R2-compliant substrate as of about Step 1381: ~1e-5 to 1e-4** depending on game draw (Steps 1349-1366). ~10% of randomly drawn games are reachable by random exploration in 2K steps. No architecture reliably exceeds this range — draw variance dominates substrate signal across 33 experiments and 2 paradigms (MLP+TP, SSM+RTRL).

**Reference (R2-violating):** CNN+Adam RHAE=2.4e-5, speedup=10.5x on sp80 (Steps 1305-1324).

## Constitution

Seven simultaneous constraints (R0-R6). Full definitions, falsification conditions, and evidence in `docs/CONSTITUTION.md`.

| Rule | Summary | Status at about Step 1381 |
|------|---------|---------------|
| R0 | Dynamics dominate initial conditions | Partial — prediction consistent, task performance untested |
| R1 | No external objectives | Holds for prediction/navigation |
| R2 | Adaptation IS the computation | Adam violates. LPL/TP comply. Organism interpretation adopted, not falsification-tested |
| R3 | Everything self-modified | Holds for weights. Open for structure |
| R4 | Tested against prior state | **Not satisfied.** Zero genuine transfer across all substrates tested |
| R5 | One fixed ground truth | Holds by construction |
| R6 | No deletable parts | Untested on current architecture |

## Test environment

**ARC-AGI-3:** 150+ game environments. 64x64 pixel grid, 16 colors, multiple levels. 7 keyboard + 4096 click actions. Rules, goals, and action effects unknown to the substrate.

**MBPP:** Text/code generation. 128 ASCII actions. Predicting next character IS selecting next action.

## Confirmed findings

Each finding cites the experiments that support it and states what would falsify it.

| # | Finding | Evidence | Falsified if |
|---|---------|----------|-------------|
| 1 | R2-compliant prediction compresses weakly (3-9%) | LPL cr=0.93 (Steps 1310-1313). K=50 iterations = K=5 (Step 1322). LPL normalized cr=0.91 (Step 1328). | An R2-compliant local update rule achieves cr < 0.5 without target propagation. |
| 2 | Target propagation compression is game-dependent (54-92%) and R2-compliant | TP cr=0.08 on easy games (Steps 1329-1343), cr=0.36-0.46 on harder games (Step 1344). "92% compression" was game-selection artifact. Local targets from forward computation, no global backward pass. | TP compression shown to require a global signal. |
| 3 | CNN+Adam compresses 98% and reaches task progress, violating R2 | cr=0.003 (Steps 1305-1307). RHAE=2.4e-5 (Step 1306). Adam optimizer separable from forward pass. | Adam shown to be non-separable from forward computation, or an R2-compliant method matches RHAE. |
| 4 | Prediction learning does not shift action distribution | action_head(h3) entropy is FLAT (max) throughout training (Step 1350). TP updates h3, h3 feeds action_head, but softmax outputs near-uniform regardless. 10 engineered action selectors also ≈ entropy (Steps 1306-1343). REFLEX vs PURE_RANDOM difference was PRNG artifact, not learned signal. | action_head entropy measurably decreases during training (H_2000 < H_100). |
| 5 | All spatial representations are episode-specific | TP anti-speedup (Steps 1330, 1335). Mode map zones: try1 [3,4,4,3] → try2 [0,0,0,0] (Step 1338). CNN try2 diverges cr=20.76 (Step 1337). | A spatial representation produces speedup > 1 confirmed by seed swap control. |
| 6 | MLP processes text; CNN cannot | MLP MBPP cr=0.08 (Step 1337). CNN MBPP cr=null, wdrift=0. | CNN shown to compress MBPP text, or MLP shown to fail on text. |
| 7 | Single-layer Hebbian degrades prediction | cr=1.44 — prediction gets worse (Step 1309). | Single-layer Hebbian achieves cr < 1.0 on any game. |
| 8 | Seeds are unnecessary | Deterministic orthogonal init produces consistent results (Step 1313). | Deterministic init produces >10% metric variance across runs. |
| 9 | Game reachability is ~10-30% with random actions in 2K steps | 3/30 games reached progress (Step 1349, 10 draws). 3/10 draws non-zero. Both TP and Adam score zero on hard seeds (Steps 1344-1345). | >50% of games reachable by random play, or 0% reachable (1349 was PRNG luck). |
| 10 | Adam also fails on hard games — not credit depth | MLP+Adam RHAE=0 on same 5 hard seeds as MLP+TP (Step 1345). Adam diverges on 65536-dim input. | Adam reaches progress where TP doesn't on matched seeds. |
| 11 | Cross-game features don't transfer (childhood) | cr=1.0 on eval game after 10-game childhood (Step 1348). Weights from 10 random games don't help on new game. | Childhood weights produce cr < 0.5 on first observation of eval game. |
| 12 | Hierarchical action improves reachability 1.64× | HIER 4/10 non-zero, RHAE=7.53e-5 vs FLAT 3/10, 4.59e-5 (Steps 1351-1352). Structural keyboard coverage, not learned. | FLAT matches or exceeds HIER on 10+ draws. |
| 13 | Type_head CAN learn from self-supervised signal | Entropy drops 0.16 with change-magnitude target (Step 1353), 0.11 with info-gain (Step 1354). First self-supervised action distribution shift in the search. | Type entropy stays flat (=max) under any self-supervised target. |
| 14 | Both trained type targets suppress clicks and regress RHAE | Change-magnitude (1353): click_frac 0.094, RHAE=0. Info-gain (1354): click_frac 0.087, RHAE=5e-6. Both worse than untrained HIER (click_frac 0.123, RHAE=7.53e-5). Action space asymmetry: keyboard always wins per-step metrics. | A type target that increases click_frac and improves RHAE simultaneously. |
| 15 | Training the action head HURTS — random actions are optimal | SSM disconnected (1364): RHAE=1.34e-4. Every trained variant scored worse: circular CE (8.5e-6), Gumbel feedback (0.0), surprise REINFORCE (0.0). 4 SSM experiments confirm. | Trained action head outperforms random on matched seeds. |
| 16 | SSM produces 2.92× better features than MLP for random exploration | SSM disconnected 2K (1364): RHAE=1.34e-4, 3/10 non-zero. MLP flat 2K (1349): 4.59e-5, 3/10 non-zero. Same reachability, higher efficiency per game. Sequence structure provides action-conditional prediction structurally. **Note: 2.92× was draw variance — SSM replication (1365) showed 0.19×. Finding unstable.** | MLP matches or exceeds SSM RHAE on 10+ matched draws. |
| 17 | Persistent recurrent state is TOXIC across episodes | SSM COUNT-PERSIST: 0/30 nz vs COUNT-RESET: 8/30 nz (Step 1375). h from try1 actively destroys try2 progress — carries action-strategy-specific state that interferes when try2 uses different actions. Replicated: PERSIST also worse in Step 1374 (RAND try1). | Persistent h outperforms reset h on 30+ paired draws. |
| 18 | Try2 action selection is irrelevant | COUNT-COUNT: 5/30 nz vs COUNT-RAND: 5/30 nz, paired 3-3-24, p=0.656 (Step 1376). With good try1 weights, try2 coverage doesn't matter. Random try2 ≈ COUNT try2. | Try2 action mechanism significantly outperforms random try2 given matched try1. |
| 19 | Disconnected actions = null experiment. Weight transfer is zero. | 100-draw PRNG-fixed paired test (Step 1377). COUNT=RAND exactly: 0-0-100 ties, p=1.0. Same chain_mean (6.2e-5), same nz (19/100). 1375's p=0.090 "transfer" was PRNG artifact (try1 mode shifted try2's random sequence through global RNG coupling — same bug as Step 1350). With disconnected actions, try1 weights CANNOT affect try2 behavior. 17 SSM experiments (1360-1377) measured noise. True baseline: 19% reachability, RHAE=6.2e-5. | Any disconnected-action experiment shows COUNT ≠ RAND with PRNG fix applied. |
| 20 | SSM prediction features ANTI-CORRELATE with task progress | Frozen projection softmax(W_fixed @ h / T=3) produces non-uniform actions (h_norm=0.72, entropy < max on 30/30 draws). But FROZEN 8/30 nz, RHAE=2.91e-5 vs DISCONNECTED 9/30 nz, RHAE=1.22e-3 (Step 1378). Structured h-based actions are WORSE than random. The prediction features encode "what's predictable" not "what leads to progress." Every action mechanism (17 trained + 1 frozen) worse than random = features point the wrong direction. | Frozen projection outperforms disconnected on 30+ paired draws. |
| 21 | **SSM is action-blind by construction** | Prediction loss with action tokens: 0.796971. Without action tokens (zeroed): 0.796955. Ratio: 1.00002 (0.002% difference). Verdict: ACTION_BLIND (Step 1379). The action token has ZERO influence on SSM prediction. h encodes only observation autocorrelation. RTRL optimizes obs→obs, which doesn't need actions. This is the ROOT CAUSE of all SSM failures: no action info in h → no mechanism can extract useful actions from h. | SSM prediction loss changes >5% when action tokens are masked. |
| 22 | Multiplicative action gating degenerates to constant | Gated SSM: x' = (A@x + B@obs) * sigmoid(W_gate @ one_hot(act)). Action-blind ratio: 1.000119 (0.012% — same as ungated). RTRL optimizes gate for obs prediction → optimal gate is constant scaling → action info doesn't enter h (Step 1380). Architecture changes cannot force action conditioning when the objective doesn't need it. | Gated SSM action-blind ratio >5%. |
| 23 | **Inverse dynamics can't fix linear SSM action-blindness** | Inverse head learns to classify actions at 88× random (2.14% accuracy, Step 1381) but through OBSERVATION SHORTCUT (obs_{t+1} pixel changes reveal action) not action-conditional state. Action-blind ratio: 0.999989 — SSM prediction unchanged. RTRL gradient flows through obs encoder, bypassing action path. Linear dynamics x'=a*x+b*u are structurally action-independent (b*act is constant regardless of state). No objective change fixes this. | Inverse dynamics + linear SSM produces action-blind ratio >5%. |

## What doesn't work

| Mechanism | Experiments | Result |
|-----------|-----------|--------|
| Prediction-based action selection (10 variants) | Steps 1306-1343, ~30 experiments | All ≈ entropy. Action-conditional model with rich encoding (1343) showed weak MBPP signal but zero ARC progress. |
| R2-compliant update rules for task progress | Steps 1309-1348, 14+ experiments | Zero RHAE > 0. TP compresses 54-92% (game-dependent) but no level advancement. |
| R2-violating (Adam) on hard games | Step 1345 | Adam ALSO scores RHAE=0 on same hard games as TP. NOT credit depth — game difficulty exceeds budget. |
| Deliberation (fewer actions, more training) | Steps 1346-1347 | K=10 reduces to 200 actions → cr=1.0 (model trains on 200 sparse transitions, memorizes). Model-based selection on cr=1.0 model has no signal. |
| Multi-episode training for transfer | Step 1336 | Worse than single-episode. Diverse episodes produce interference, not invariance. |
| Childhood (multi-game pretraining) | Step 1348 | cr=1.0 on evaluation game. Features from 10 random games don't transfer to new games. Cross-game transfer as dead as within-game transfer. |
| Overfitting detection (internal R4) | Step 1334 | Detection triggers but LR reduction too aggressive. Calibration issue, not concept failure. |
| Meta-learned plasticity | Steps 1325, 1339 | Theta found correct direction (1325) but credit signal too weak. Theta frozen at init (1339) — normalized credit fix identified but untested. |
| Mode map / zone discovery | Step 1338 | Zones are spatial (episode-specific), not functional. try1 zones don't persist to try2. |
| Trained type_head (change-magnitude) | Step 1353 | Entropy drops 0.16 (learns!) but favors keyboard (change=15.6) over click (change=1.7). Regresses RHAE to 0/5. |
| Trained type_head (info-gain) | Step 1354 | Entropy drops 0.11. Also favors keyboard (denser exploration → more learning per step). click_frac drops to 0.087. RHAE=5e-6 (1/5), regression vs untrained HIER. |

## Compression spectrum (empirically mapped)

| Update rule | Compression | R2 status | RHAE(try2) | Key step |
|---|---|---|---|---|
| LPL Hebbian | 5% | Compliant | 0 | 1310 |
| LPL normalized | 9% | Compliant | 0 | 1328 |
| DFA (random backward) | 34% | Compliant | 0 | 1326 |
| Target propagation | 92% | Compliant | 0 | 1329 |
| Adam (full gradient) | 99.7% | Violating | 2.4e-5 | 1324 |

## Open directions at about Step 1381 (from catalog, 46 items)

*Historical: this section describes what was planned at the time. The project later moved to Era II and was concluded; see [`docs/STAGE_MAP.md`](docs/STAGE_MAP.md).*

Tested this session: mode map (#16, killed 1338), meta-plasticity (#14, killed 1339), action-conditional model (#42, partial signal on MBPP), deliberation (#41, killed 1346-1347), childhood (#44, killed 1348).

**Key eliminative finding (Steps 1344-1348):** Game seeds 13440-13444 are unreachable by ANY substrate (TP or Adam) with 2K random actions. The bottleneck on these games is not the update rule, not action selection, not credit depth, not prior knowledge — it's that random exploration in a 4103-action space with 2K steps has near-zero probability of hitting required action sequences.

**Direction at the time: SSM paradigm shift (Step 1360).** MLP+TP exhausted after 25 experiments (1334-1358). Replacing with small Mamba-style SSM + RTRL (online local gradients). Sequence model processes interleaved (obs, act) stream. Recurrent state = history + world model. Actions are tokens in the sequence. RTRL gradient for diagonal SSM is local and Hebbian-like (prediction_error × state). R2-compliant by construction. This was then built and tested in Steps 1360-1381; findings 15-23 above record the result (the SSM was action-blind).

Untested at that time:
- **Normalized meta-plasticity credit (#46)** — fix for 1339 theta freeze. Identified, not yet tested.
- **#32/#33: Self-directed pruning / activity-dependent growth** — architecture emerges from dynamics.
- **#36: "Does the substrate understand what a game is?"** — no experiment has measured internal task-structure representation.

Full catalog: `docs/MAP.md` underexplored directions section (46 items, merged 2026-03-30)

## Competition reference

**ARC-AGI-3 Preview Top 3 (2026-03-26):**
- 1st StochasticGoose (Tufa Labs, 12.58%): CNN + spatial 64×64 click head + "did frame change" binary supervision. 200K experience buffer. R1-adjacent signal.
- 2nd Blind Squirrel (6.71%): Frame-to-graph + ResNet18 value model. Online learning.
- 3rd Graph Exploration (training-free): Pure heuristic graph exploration, 17 levels post-fix. Nearly matches 1st place without learning.

**Key implications:** Spatial click encoding critical (conv preserves 2D; our avgpool4 destroys it). Graph exploration is surprisingly strong (validates defense thesis). "Did frame change" is R1-adjacent but trained via BCE (R1-violating method, R1-compliant signal).

**Prescription status (10 games fully solved, 12 partial, 3 unsolved):** a "prescription" is a fixed action sequence derived by analysing each game (for example a BFS solver written from the game's source), not something a learning system found. Prescriptions are reference solutions, not scoreable results under the rules ([`docs/journal/archive_steps_1_1022.md`](docs/journal/archive_steps_1_1022.md), Step 1018e).

| Status | Games | Action range |
|--------|-------|-------------|
| Complete (all levels) | ft09(75), ls20(311), vc33(176), lp85(79), tr87(123), sb26(124), sp80(107), cd82(140), cn04(107), tu93(185) | 75-311 |
| Partial (L1+) | re86(210), r11l(65), s5i5(39), m0r0(15), su15(25), ar25(26), dc22(20), sc25(17), g50t(17), wa30(77), bp35(29), lf52(8) | 8-210 |
| Unsolved | ka59, sk48, tn36 | — |

## Repository structure

- `docs/` — All research documentation
  - `CONSTITUTION.md` — R0-R6 constitutional constraints
  - `RESEARCH_STATE.md` — Dated research record
  - `MAP.md` — Constraint map + underexplored directions
  - `COMPONENT_CATALOG.md` — Validated component library
  - `PAPER.md` — Draft paper
  - `propositions/` — 27 propositions + 3 theorems
  - `kills/` — Killed family documentation
  - `bans/` — Ban policy (both lifted)
- `experiments/` — All experiment code, infrastructure, and results
  - `steps/` — Step scripts (the box above says which ones run and what they need)
  - `compositions/` — Results of Steps 1393-1394
  - `results/` — Result files cited in the findings
  - `run.py` — Universal PRISM harness
  - `environments/` — Game environment wrappers
