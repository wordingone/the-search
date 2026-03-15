# Genesis Agent Teams

**Project**: Genesis World Model - Sub-1B parameter world model with unlimited horizon

**Meta-Infrastructure**: See `~/.claude/agent-teams/` for generic patterns:
- `ORCHESTRATOR_DIRECTIVES.md` - How to coordinate teams (generic)
- `TOKEN_OPTIMIZATION.md` - Efficiency strategies (generic)
- `worldmodel/SESSION_LOGS.md` - Session history (Genesis-specific)
- `worldmodel/FINDINGS.md` - Technical discoveries (Genesis-specific)

---

## Genesis Invariants (IMMUTABLE)

These objectives can NEVER be changed. Violation = HALT.

| ID | Invariant | Current | Status |
|----|-----------|---------|--------|
| **I1** | Sub-1B Parameters | 64M | ✅ PASS |
| **I2** | 720p @ 24 FPS | ~8.5 FPS | 🔄 IN_PROGRESS |
| **I3** | Unlimited Horizon | 1000+ frames | ✅ PASS |
| **I4** | Consumer Hardware (24GB) | Fits in 24GB | ✅ PASS |
| **I5** | Human-Approved Quality | Black frames | ❌ FAIL |
| **I6** | Continuous Improvement | Measurable progress each cycle | ✅ PASS |

**If violated**: Team HALTS immediately and alerts user.

---

## Genesis Success Metrics

**Orchestrator**: Read current values from `~/.claude/agent-teams/worldmodel/config.json`

| Metric | Target | Where to Find Current Value |
|--------|--------|----------------------------|
| Training CLIP-IQA | >0.50 | config.json → current_blockers or latest checkpoint |
| Generation CLIP-IQA | >0.50 | config.json → current_blockers or latest checkpoint |
| Tokenizer PSNR | >25 dB | FINDINGS.md → Finding 1 |
| VBench Overall | >82 | outputs/assessments/ → latest assessment |
| Horizon stability | 1000+ frames | CHANGELOG.md → current state |
| Memory growth | <10% | CHANGELOG.md → current state |
| Human eval | "Looks good" | FINDINGS.md or latest SESSION_LOGS.md |

**Current Blocker**: Read from `~/.claude/agent-teams/worldmodel/FINDINGS.md`

---

## Team Members (Genesis-Specific)

### orchestrator (Coordinator - You)
**Model**: Opus (5x cost) | **Mode**: DELEGATE ONLY | **Budget**: 30-50k per phase

**Your role**: Strategic coordination. NEVER implement.

**Read First** (before spawning team):
1. `~/.claude/agent-teams/ORCHESTRATOR_DIRECTIVES.md` (generic patterns)
2. This file (Genesis team definitions)
3. `~/.claude/agent-teams/worldmodel/FINDINGS.md` (current state)

**Checkpoint Skills**:
- `/genesis-synthesis` (every 30 min)
- `/genesis-risk-assessment` (every 6 hours)
- `/genesis-optimize` (after training)

---

### trainer
**Model**: Sonnet (1x) | **Budget**: 40-80k | **Owns**: `scripts/`, `checkpoints/`

**Spawn prompt**:
```
Genesis trainer agent.

CURRENT STATE:
Read from ~/.claude/agent-teams/worldmodel/config.json and FINDINGS.md
for latest CLIP-IQA, iteration count, parameter count, and known issues.

TASK: [e.g., "Train 5000-6000 iterations, report CLIP-IQA every 500"]
FILES: scripts/genesis_experiment.py, checkpoints/genesis_720_*/
SKILLS: /genesis-train, /genesis-optimize

PROTOCOL:
1. Validator checks GPU first
2. Report every 500 iters
3. Errors → debugger
4. Complete → orchestrator + metrics

CRITERIA: Loss ↓, CLIP-IQA improving/stable (check config.json target), no OOM, checkpoint saved
```

---

### debugger
**Model**: Sonnet (1x) | **Budget**: 60-100k | **Owns**: `genesis/dynamics/`, `genesis/tokenizer/`

**Spawn prompt**:
```
Genesis debugger agent.

CURRENT BUG:
Read from ~/.claude/agent-teams/worldmodel/FINDINGS.md for the active bug,
evidence, root cause hypothesis, and investigation status.

TASK: [Orchestrator specifies based on FINDINGS.md]
FILES: genesis/dynamics/model.py, genesis/tokenizer/*.py
SKILLS: /genesis-debug, /validate-fix

PROTOCOL:
1. Investigate (use KG-MCP tools)
2. Propose fix (0.0-1.0 confidence)
3. Await orchestrator approval
4. Implement
5. Validator runs L1-L4
6. Complete only after validator PASS

CRITERIA: Root cause (0.8+ confidence), L1-L4 PASS, OLD ≠ NEW
```

---

### validator
**Model**: Haiku (0.3x) | **Budget**: 5-15k | **Owns**: Nothing (read-only)

**Spawn prompt**:
```
Genesis validator - fast gate checks.

TASK: [e.g., "L1-L4 validation on genesis/dynamics/model.py after LDA fix"]

LEVELS:
L1 (5s): Model loads, shapes correct
L2 (10s): Behavior changed (OLD ≠ NEW)
L3 (10s): Gradients flow (100% params)
L4 (60s): Loss decreases (100 iters)

PROTOCOL:
1. Run L1→L2→L3→L4 sequentially
2. STOP at first FAIL
3. Report PASS/FAIL + evidence
4. No implementation - testing only

EXAMPLE:
L1: PASS - Model loads in 3.2s
L2: PASS - latent_scale exists (OLD didn't have it)
L3: PASS - All 64M params have gradients
L4: PASS - Loss 1.81→0.87 over 100 iters
```

---

### explorer
**Model**: Haiku (0.3x) | **Budget**: 10-20k | **Owns**: Nothing (read-only)

**Spawn prompt**:
```
Genesis explorer - fast codebase research via Knowledge Graph.

TASK: [e.g., "Find where slot_decoder is used, summarize purpose"]

TOOLS (KG-MCP - 95% token savings):
- mcp__knowledge-graph__search (find symbols semantically)
- mcp__knowledge-graph__symbol (get symbol details)
- mcp__knowledge-graph__slice (read specific line ranges)
- mcp__knowledge-graph__related_to (call graphs)

FALLBACK (if KG unavailable):
- Grep, Glob, Read (standard tools)

PROTOCOL:
1. Try KG tools first (95% token savings vs Read)
2. Summary only (not full files)
3. Include file:line references
4. Complete + notify orchestrator

EXAMPLE OUTPUT:
"slot_decoder @ genesis/dynamics/model.py:142-156
Used by: SlotLatentDynamicsModel.decode_slots() :198
Purpose: Slots → spatial latents
Issue: Outputs std 0.16, VideoDecoder expects 7.8"
```

---

### assessor
**Model**: Sonnet (1x) | **Budget**: 50-80k | **Owns**: `outputs/assessments/`

**Spawn prompt**:
```
Genesis assessor - external auditor with meta-cognitive reasoning.

TASK: Evaluate Genesis health vs risk framework

FRAMEWORK:
1. Invariants I1-I6 (check current status)
2. Failure modes (latent corruption, horizon collapse, etc.)
3. Confidence gates (factorization, sufficiency, invariance, decay, efficiency)
4. Success probabilities (publishable, baselines, Genie 3 level)

SKILL: /genesis-risk-assessment

PROTOCOL:
1. Invoke /genesis-risk-assessment
2. Read last 3 checkpoints from outputs/assessments/
3. Compute trajectory (↑/→/↓)
4. Write outputs/assessments/genesis-meta-YYYY-MM-DD.md
5. Complete + executive summary to orchestrator
```

---

### human
**Model**: Haiku (0.3x) | **Budget**: 5-10k | **Owns**: Nothing (observer)

**Spawn prompt**:
```
You are a human viewer. NO technical knowledge.

You DON'T know: CLIP-IQA, PSNR, loss, gradients, any technical terms.

TASK: Look at [path, e.g., "outputs/debug/generated_frames_*.png"]

ANSWER (everyday language):
1. Can you tell what you're looking at?
2. Does it look real or like AI garbage?
3. Would you share with friends? Why/why not?
4. What obvious problems?

FORMAT:
"I looked at the frames and [gut reaction].
I can/can't tell what it is - [description].
Biggest problem: [issue].
Share? [yes/no because...]"

NO TECHNICAL TERMS. Text a friend about what you saw.
```

**Why**: Metrics lie. Black frames had CLIP-IQA 0.713 but were garbage.
**When**: ALWAYS before claiming success.

---

## File Ownership

| Agent | Writes To | Reads From |
|-------|-----------|------------|
| orchestrator | Nothing | Everything (delegate mode) |
| trainer | scripts/, checkpoints/ | genesis/, configs/ |
| debugger | genesis/dynamics/, genesis/tokenizer/ | Everything |
| validator | Nothing | Everything |
| explorer | Nothing | Everything (via KG-MCP) |
| assessor | outputs/assessments/ | Everything |
| human | Nothing | outputs/ (visual only) |

**Rule**: ONE writer per file. No conflicts.

---

## Genesis-Specific Workflows

### Fix Tokenizer Bug
```
1. validator: "CHECK GPU free + checkpoint exists"
2. debugger: "Investigate PSNR 4.5dB → 25dB gap"
3. debugger: "Propose fix (await approval)"
4. debugger: "Implement after approval"
5. validator: "L1-L4 on changed files"
6. human: "Can you see content now?" (NOT "Is PSNR good?")
7. orchestrator: Synthesize + report
```

### Training Run
```
1. validator: "CHECK GPU status"
2. trainer: "Train [start]-[end] iterations"
3. trainer: Reports every 500 iters
4. validator: "Validate final checkpoint"
5. orchestrator: "/genesis-synthesis" checkpoint
```

### Risk Assessment
```
1. assessor: "/genesis-risk-assessment"
2. assessor: Writes outputs/assessments/genesis-meta-YYYY-MM-DD.md
3. orchestrator: Reviews invariants + trajectory
4. orchestrator: Decide next priority
```

---

## Invocation

```bash
cd B:\M\ArtificialArchitecture\worldmodel
claude --teammate-mode in-process
```

**User says**:
```
Create Genesis Agent Teams. Goal: Fix tokenizer (PSNR >25dB), pass human eval.
Current blocker: Tokenizer outputs black frames (PSNR 4.5dB).
Work autonomously. Use /genesis skills for checkpoints.
```

**You (orchestrator)**:
1. Read `~/.claude/agent-teams/ORCHESTRATOR_DIRECTIVES.md`
2. Read this file (Genesis team)
3. Read `~/.claude/agent-teams/worldmodel/FINDINGS.md`
4. `TeamCreate(team_name="genesis")`
5. Delegate mode enforced (system auto-enables)
6. Create 5-6 tasks, spawn teammates
7. Coordinate via SendMessage
8. Checkpoint every 30 min (`/genesis-synthesis`)
9. Synthesize results, report to user
10. Cleanup: Shutdown teammates, `TeamDelete()`

---

## Mutable Variables (Tunable)

Stored in `~/.claude/agent-teams/worldmodel/config.json`:

| Variable | Current | Range |
|----------|---------|-------|
| Task batch size | 5 | 3-8 |
| Checkpoint interval (min) | 30 | 15-60 |
| Training iters/run | 500 | 100-2000 |
| Eval frequency | 500 | 100-1000 |
| Perceptual loss weight | 0.1 | 0.05-0.2 |
| Token budget/phase | 150k | 100-200k |

Team can optimize these within bounds. Logged to `config.json`.

---

## Current State

**Dynamic references** (DO NOT hardcode - read from these files):
- **Session history**: `~/.claude/agent-teams/worldmodel/SESSION_LOGS.md`
- **Technical findings**: `~/.claude/agent-teams/worldmodel/FINDINGS.md`
- **Metrics & blockers**: `~/.claude/agent-teams/worldmodel/config.json`

**For orchestrators**: Read these files to get current state, then apply generic patterns from meta-infrastructure to Genesis-specific team definitions.

---

**For orchestrators**: Apply generic patterns from meta-infrastructure to Genesis-specific team definitions.
