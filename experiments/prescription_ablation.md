# Prescription Ablation Analysis

*Jun directive 2026-03-24. For each game, strip the prescription to its minimum viable form. What's essential vs removable scaffolding? The minimum prescription = what the substrate must autonomously discover.*

## Ablation Hierarchy (most → least purgable)

1. **Pixel coordinates** — exact (x,y) positions. Replaceable by CC zone discovery.
2. **Pre-computed paths** — BFS/shortest-path. Replaceable by interaction-based exploration.
3. **Zone identity** — knowing which regions are clickable/important. Discoverable by random probing + frame diff.
4. **Action ordering** — which zone/action WHEN. The temporal credit problem. Likely essential.
5. **Mechanics understanding** — how game elements interact (colors cycle, water levels change, shapes match). Likely essential.

---

## FT09 — Full prescription: 75 clicks, 6 levels (ABLATION COMPLETE)

**Prescription components:**
- C1: Exact pixel coordinates per click (2-pixel precision required)
- C2: Which level you're on (level detection)
- C3: Per-level click SET (L1: 4 walls, L2: 7 walls, L3: 14 walls, L4: 16 clicks on 11 walls, L5: 21 Lights-Out, L6: 13 Lights-Out)
- C4: ~~Total ordering within each level~~ **ORDER IRRELEVANT (Step 1019b)**
- C5: Game mechanics: L1-L4 = single/multi-click wall toggling, L5/L6 = Lights-Out (+spread / up-spread, solved via GF(2) Gaussian elimination)

**Ablation results:**
| Step | Ablation | Result | Finding |
|------|----------|--------|---------|
| 1019a-A | Exact coords (baseline) | 6/6 PASS | 75 clicks (4+7+14+16+21+13) |
| 1019a-B | 8px zone snap | 1/6 (L1 only) | Walls at odd grid, snap at even → off-by-1 |
| 1019a-C | 6-cluster k-means | 0/6 | Clusters collapse position info |
| 1019b-A | Original order | 6/6 PASS | Baseline |
| 1019b-B | Reversed order | 6/6 PASS | Order irrelevant |
| 1019b-C | Random permutation (20×) | 6/6 PASS all | Order irrelevant for ALL levels including Lights-Out |

**CONFIRMED minimum prescription:** {Unordered set of exact (x,y) per level}. 75 clicks total, all necessary, 2-pixel precision. No ordering required. No redundancy possible.

**What the substrate must discover autonomously:**
1. Exact wall positions (not zones — 2px precision)
2. How many times to click each wall (L4: some walls need 2 for 3-color cycle)
3. For L5/L6: which subset of walls to toggle (Lights-Out solution = linear algebra over GF(2))
4. Does NOT need to learn click ordering

---

## VC33 — Full prescription: 176 analytical BFS clicks, 7 levels

**Prescription components:**
- C1: Zone coordinates (canal gates, water sources, targets)
- C2: Canal mechanics model (water flows downhill, gates block/allow)
- C3: BFS path planner (optimal gate sequence)
- C4: Camera offset correction
- C5: Per-level state estimation (water level tracking)

**Ablation experiments:**
| Step | Remove | Replace with | Expected | Tests |
|------|--------|-------------|----------|-------|
| B1 | C1 (zone coords) | CC-discovered zones | Probably works | Same as FT09 A1 |
| B2 | C1+C3 (coords+BFS) | CC zones + greedy (click zone that changes most) | Partial — may work for easy levels | Greedy vs planned |
| B3 | C1+C2+C3 (coords+mechanics+BFS) | CC zones + random | 0/7 (proved by Step 1017) | Confirms minimum |
| B4 | C4 (camera offset) | Auto-detect from frame analysis | Probably works | Minor scaffolding |

**Predicted minimum prescription:** Zone identity + mechanics model (C2) + some planning (C3, possibly simplified). Camera offset auto-detectable.

---

## LS20 — Full prescription: 311 actions (Step 1018e confirmed), 7 levels [13,45,41,43,44,72,53]

**Prescription components (from source analysis):**
- C1: Wall layout per level (ihdgageizm positions)
- C2: Goal positions per level (rjlbuycveu positions)
- C3: Shape matching requirements (GoalColor, GoalRotation per level)
- C4: Collectible positions (npxgalaybz — extend step budget)
- C5: BFS path planning (start → shape match → goal)
- C6: Shape cycling mechanics (color/rotation/shape cycling via tagged sprites)
- C7: Step budget management (42 steps, 3 lives)

**Ablation experiments (after Step 1018):**
| Step | Remove | Replace with | Expected | Tests |
|------|--------|-------------|----------|-------|
| C1 | Walls (C1) | Collision detection by trying to move | Works (discovery by interaction) | Wall knowledge purgable |
| C2 | Goal position (C2) | Explore until frame matches goal indicator | Likely works if goal visible | Goal detection purgable? |
| C3 | Shape matching (C3) | Trial-and-error cycling at goal | Works if budget allows ~12 tries (4 shapes × 4 rotations × 4 colors... too many?) | Shape matching essential? |
| C4 | Collectibles (C4) | Ignore collectibles | Fails on tight-budget levels | Budget-dependent |
| C5 | Path planning (C5) | Random walk within budget | Fails (21 moves too few for random, proved by Step 1015) | Planning essential |

**Predicted minimum prescription:** Goal detection (simplified C2) + shape matching (C3, possibly trial-and-error) + some path planning (C5, even simple). Wall knowledge purgable. Collectible knowledge depends on budget tightness.

---

## Cross-Game Minimum

**What ALL games share that's essential:**
1. **Zone/target discovery** — knowing where to interact (CC discovery handles this)
2. **Effect observation** — seeing what each action does (frame diff handles this)
3. **Ordering/sequencing** — knowing which action WHEN (the universal gap)
4. **Mechanics inference** — learning game rules from interaction (unsolved)

**What's purgable across all games:**
- Exact coordinates (CC discovery)
- Pre-computed paths (interaction-based exploration)
- Level detection (frame change monitoring)
- Camera correction (auto-detectable)

**The substrate must discover:** action-effect mappings + ordering + game mechanics. From interaction, not from source analysis.

---

## RHAE Scoring Constraint (ARC-AGI-3)

Score = (human_baseline / agent_actions)^2 per level. The substrate must not just SOLVE levels but solve them EFFICIENTLY.

| Game | Level | Human Baseline | BFS Solution | Max Budget (25%) | Max Budget (1%) |
|------|-------|---------------|-------------|-----------------|----------------|
| FT09 | L1 | 17 | 7 | 34 | 170 |
| FT09 | L2 | 19 | 7 | 38 | 190 |
| FT09 | L3 | 15 | 7 | 30 | 150 |
| FT09 | L4 | 21 | 7 | 42 | 210 |
| FT09 | L5 | 65 | 40 | 130 | 650 |
| FT09 | L6 | 26 | 7 | 52 | 260 |
| LS20 | L1 | 21 | 13 | 42 | 210 |
| LS20 | L2 | 123 | 45 | 246 | 1230 |
| LS20 | L3 | 39 | 41 | 78 | 390 |
| LS20 | L4 | 92 | 43 | 184 | 920 |
| LS20 | L5 | 54 | 44 | 108 | 540 |
| LS20 | L6 | 108 | 72 | 216 | 1080 |
| LS20 | L7 | 109 | 53 | 218 | 1090 |

**Implication for ablation:** A mechanism that discovers the prescription in 10K steps but EXECUTES it in 20 actions scores well. A mechanism that needs 10K steps to stumble on the solution and has no way to replay it scores ~0%. The substrate needs LEARNING (discover once) + EXECUTION (replay efficiently), not just exploration.
