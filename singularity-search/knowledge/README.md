# Project Knowledge System

A lightweight, structured memory layer that any AI agent can use to instantly understand a project's history, constraints, and current state.

## What This Is

Think of it like a biological sensory organ. The retina doesn't send 130 million raw photoreceptor readings to the brain — it sends edges, motion, contrast. This system does the same: it takes raw project knowledge (experiment results, decisions, discoveries) and compiles them into a **pre-processed state artifact** that an agent reads ONCE at session start.

No tool calls. No queries. No context bloat.

## File Structure

```
.knowledge/
├── state.md              # THE output artifact. Compressed. Agent reads this ONCE.
├── index.json            # Machine-readable structured data behind state.md
├── entries/              # Raw knowledge records (append-only log)
│   ├── 001.json
│   ├── 002.json
│   └── ...
├── constraints.json      # Derived active constraints (compiled from entries)
├── compile.py            # entries/ → state.md + index.json + constraints.json
├── ingest.py             # CLI helper to create entries
└── README.md             # This file
```

## Quick Start

### Reading Knowledge (Agent)

```python
# At session start, read the compiled state
with open('.knowledge/state.md') as f:
    state = f.read()

# That's it. No queries. No DB. Just read it once.
```

### Adding Knowledge (Human or Agent)

```bash
# Interactive entry creation
python .knowledge/ingest.py

# From a JSON file
python .knowledge/ingest.py --from experiment_results.json

# From stdin (pipe experiment output)
echo '{"type":"experiment","title":"..."}' | python .knowledge/ingest.py --stdin

# Batch create from JSON array
python .knowledge/ingest.py --batch entries.json
```

### Compiling Knowledge

```bash
# Regenerate state.md, index.json, constraints.json from entries/
python .knowledge/compile.py
```

This is idempotent — run it 10 times, same output.

## Entry Schema

Each entry in `entries/` is a JSON file:

```json
{
  "id": "006",
  "timestamp": "2026-02-24T...",
  "type": "experiment",
  "title": "Self-referential eta adaptation",
  "tags": ["stage3", "eta", "meta-learning"],
  "status": "failed",
  "content": {
    // type-specific fields
  },
  "constraints_implied": [
    "Multiplicative self-referential meta-rates are unstable"
  ],
  "references": ["003", "004"],
  "session": 6
}
```

### Entry Types

The type system is open. The `content` field is type-specific but the outer envelope is universal.

**Current types:**
- `experiment`: `{hypothesis, parameters, method, result, metrics, root_cause}`
- `decision`: `{choice, alternatives, rationale, reversible}`
- `discovery`: `{finding, evidence, implications}`
- `constraint`: `{rule, source_entries, active, scope}`
- `architecture`: `{component, description, role, dependencies}`

New types can be added as needed. This lets the system work for any project — a web app might use `decision`, `architecture`, `discovery` but never `experiment`.

## state.md Structure

Organized by RELEVANCE, not chronology:

1. **Active Constraints** — what NOT to do and why (most important for agents)
2. **Current State** — where the project stands right now
3. **Key Decisions** — architectural choices still in effect
4. **Recent Findings** — discoveries from the last N sessions
5. **Historical Summary** — older entries compressed into one-liners

Rules:
- Target: under 100 lines (this goes into agent context, every line costs tokens)
- Constraints are FIRST because they prevent repeating mistakes
- Recent entries get more detail than old ones (temporal decay)
- Failed experiments are MORE important than successes (they define the search boundary)
- Terse, telegraphic style. No prose. No "we discovered that..." — just "X: failed. Cause: Y. Constraint: Z."

## Design Principles

1. **Files, not databases** — JSON files are the database. No server, no SQL, no vector DB.
2. **Stdlib only** — Python stdlib ONLY. No pip install. No dependencies.
3. **Append-only** — Never modify existing entries, only add new ones.
4. **Idempotent compilation** — Run `compile.py` 10 times, same output.
5. **Deterministic** — No LLM calls in compilation. Pure Python logic.
6. **Works on Windows** — Paths, encoding, line endings all handled.

## Current Status

- 18 entries seeded from Sessions 1-6 of the_singularity_search project
- 9 active constraints extracted
- state.md: 100 lines (exactly at target)
- Covers full experimental history: Stage 1 (Living Seed), Stage 2 (beta/gamma adaptation failures), Stage 3 (eta meta-learning)

## Future Hooks (Not Yet Implemented)

- Claude Code `post_session` hook that runs `compile.py` automatically
- CLAUDE.md could include: "Read .knowledge/state.md for project history"
- Different "lenses" (compile profiles) for different agent roles
- Integration with B:\M\memory for deeper storage
- MCP server interface for programmatic access

## Why This Matters

Traditional project memory is either:
- **Too verbose**: Full CHANGELOG is hundreds of lines, drowns agents in detail
- **Too sparse**: Project README lacks operational constraints and failure history
- **Too scattered**: Decisions in one file, experiments in another, constraints nowhere

This system compresses 6 sessions of experimental work into 100 lines of agent-readable state that includes:
- What doesn't work (9 constraints)
- Why it doesn't work (root causes)
- What to try next (current state)
- What decisions are still in effect (2 decisions)

An agent reading state.md knows more about the project than an agent reading 1000 lines of CHANGELOG.

## Examples

### Failed Experiment → Constraint

Entry 016 (Stage 3 Exp A) failed with root cause "Multiplicative self-reference creates exponential growth/decay."

This generates constraint c008: "Multiplicative self-referential meta-rates cause bang-bang oscillation (inherently unstable)"

Future agents see this in state.md line 8 and never try self-referential meta-rates again.

### Discovery → Implication

Entry 012 (Strong Thesis Verdict) finds: "Adaptation holds for per-cell params (alpha), FAILS for global shared params (beta/gamma)"

Implications: "Frozen frame floor may be >0 for global coupling parameters" + "Architecture change may be required"

This guides strategic decisions: either accept frozen frame floor or change architecture.

## Contributing

1. Add entries via `ingest.py` (manual) or by appending to `entries/` (programmatic)
2. Run `compile.py` to regenerate outputs
3. Verify `state.md` is still under 100 lines (or adjust compilation logic if needed)
4. Commit `entries/*.json` and compiled outputs together

## License

Part of the_singularity_search project. Same license applies.
