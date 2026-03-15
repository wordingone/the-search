#!/usr/bin/env python3
"""
compile.py — Knowledge base compiler.
Reads all entries and produces:
  - state.md: compressed human/agent-readable summary
  - index.json: machine-readable structured index
  - constraints.json: active constraints extracted from entries
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import textwrap

KNOWLEDGE_DIR = Path(__file__).parent
ENTRIES_DIR = KNOWLEDGE_DIR / "entries"
STATE_FILE = KNOWLEDGE_DIR / "state.md"
INDEX_FILE = KNOWLEDGE_DIR / "index.json"
CONSTRAINTS_FILE = KNOWLEDGE_DIR / "constraints.json"
PROGRESS_FILE = KNOWLEDGE_DIR / "meta" / "progress.json"

def load_entries():
    """Load all entries from entries/ directory."""
    entries = []
    for filepath in sorted(ENTRIES_DIR.glob("*.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            entry = json.load(f)
            entries.append(entry)
    return entries

def extract_constraints(entries):
    """Extract active constraints from all entries."""
    constraints = []
    constraint_id = 1

    for entry in entries:
        # Explicit constraint entries
        if entry.get("type") == "constraint":
            constraints.append({
                "id": f"c{constraint_id:03d}",
                "rule": entry.get("content", {}).get("rule", ""),
                "source_entries": [entry["id"]],
                "tags": entry.get("tags", []),
                "active": entry.get("content", {}).get("active", True)
            })
            constraint_id += 1

        # Implied constraints from other entries
        for constraint_text in entry.get("constraints_implied", []):
            constraints.append({
                "id": f"c{constraint_id:03d}",
                "rule": constraint_text,
                "source_entries": [entry["id"]],
                "tags": entry.get("tags", []),
                "active": True
            })
            constraint_id += 1

    return constraints

def build_index(entries, constraints):
    """Build machine-readable index."""
    entries_by_type = defaultdict(list)
    entries_by_tag = defaultdict(list)
    entries_by_status = defaultdict(list)

    for entry in entries:
        entry_id = entry["id"]
        entries_by_type[entry["type"]].append(entry_id)

        for tag in entry.get("tags", []):
            entries_by_tag[tag].append(entry_id)

        status = entry.get("status")
        if status:
            entries_by_status[status].append(entry_id)

    return {
        "project": "the_singularity_search",
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "active_constraints": len([c for c in constraints if c.get("active", True)]),
        "entries_by_type": dict(entries_by_type),
        "entries_by_tag": dict(entries_by_tag),
        "entries_by_status": dict(entries_by_status)
    }

def format_experiment(entry):
    """Format experiment entry for state.md."""
    content = entry.get("content", {})
    status = entry.get("status", "").upper()
    lines = [f"**{entry['title']}** [{status}]"]

    if "hypothesis" in content:
        lines.append(f"  Hypothesis: {content['hypothesis']}")

    if "result" in content:
        lines.append(f"  Result: {content['result']}")

    if "metrics" in content:
        lines.append(f"  Metrics: {content['metrics']}")

    if status == "FAILED" and "root_cause" in content:
        lines.append(f"  Root cause: {content['root_cause']}")

    return "\n".join(lines)

def format_decision(entry):
    """Format decision entry for state.md."""
    content = entry.get("content", {})
    lines = [f"**{entry['title']}**"]

    # Support 'choice', 'decision', and 'finding' keys
    choice_text = content.get("choice") or content.get("decision") or content.get("finding", "")
    lines.append(f"  Choice: {choice_text}")

    if content.get("reversible") is not None:
        rev = "reversible" if content["reversible"] else "irreversible"
        lines.append(f"  Status: {rev}")

    if "rationale" in content:
        lines.append(f"  Rationale: {content['rationale']}")

    return "\n".join(lines)

def format_discovery(entry):
    """Format discovery entry for state.md."""
    content = entry.get("content", {})
    lines = [f"**{entry['title']}**"]
    lines.append(f"  Finding: {content.get('finding', '')}")

    if "evidence" in content:
        lines.append(f"  Evidence: {content['evidence']}")

    return "\n".join(lines)

def format_architecture(entry):
    """Format architecture entry for state.md."""
    content = entry.get("content", {})
    status = entry.get("status", "")

    # Determine label suffix
    if status == "active-substrate":
        label = " [ACTIVE SUBSTRATE]"
    elif status == "archived":
        frozen_note = content.get("frozen_frame_note", "")
        label = f" [ARCHIVED — {frozen_note}]" if frozen_note else " [ARCHIVED]"
    else:
        label = ""

    lines = [f"**{entry['title']}**{label}"]
    desc = content.get("description") or content.get("change", "")
    lines.append(f"  {desc}")

    # Canonical params for active substrate
    if "canonical_params" in content:
        params = content["canonical_params"]
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"  Canonical params: {param_str}")

    # Stage status
    if "stage_status" in content:
        lines.append(f"  {content['stage_status']}")

    if "role" in content and status != "active-substrate":
        lines.append(f"  {content['role']}")

    return "\n".join(lines)

def load_progress():
    """Load progress.json if it exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def generate_review_section(progress, current_session):
    """Generate meta-cognitive review section for state.md."""
    lines = ["## META-COGNITIVE REVIEW DUE", ""]

    lines.append(f"**Current session:** {current_session}")
    lines.append(f"**Last review:** Session {progress['last_review_session']}")
    lines.append(f"**Sessions since last review:** {current_session - progress['last_review_session']}")
    lines.append("")

    # Frozen frame inventory
    lines.append("### Frozen Frame Inventory")
    lines.append("")
    lines.append(f"**Frozen elements ({progress['frozen_frame']['count']}):**")
    for elem in progress['frozen_frame']['elements']:
        lines.append(f"  - {elem}")
    lines.append("")

    lines.append("**Adaptive elements:**")
    for elem in progress['frozen_frame']['adaptive']:
        lines.append(f"  - {elem}")
    lines.append("")

    # Stage status
    lines.append("### Stage Progress")
    lines.append("")
    lines.append("| Stage | Name | Status | Notes |")
    lines.append("|-------|------|--------|-------|")
    for stage in progress['stages']:
        status = stage['status']
        notes = stage.get('notes', '')[:50] + '...' if len(stage.get('notes', '')) > 50 else stage.get('notes', '')
        lines.append(f"| {stage['stage']} | {stage['name'][:20]} | {status} | {notes} |")
    lines.append("")

    # Review prompt
    lines.append("### Review Prompt")
    lines.append("")
    lines.append("Perform adversarial progress assessment:")
    lines.append("1. Distinguish progress ON the path (frozen frame reduction) from progress BESIDE the path (infrastructure, understanding).")
    lines.append("2. What changed since last review? What experiments ran? What failed? What worked?")
    lines.append("3. Honest assessment: Are we closer to Stage 3? Or optimizing within Stage 2?")
    lines.append("4. Update progress.json: increment last_review_session, append review summary to reviews array.")
    lines.append("")

    return "\n".join(lines)

def generate_state_md(entries, constraints):
    """Generate compressed state.md file."""
    lines = ["# Project Knowledge State", ""]

    # 1. ACTIVE CONSTRAINTS (most important)
    active_constraints = [c for c in constraints if c.get("active", True)]
    if active_constraints:
        lines.append("## Active Constraints")
        lines.append("What NOT to do and why:")
        lines.append("")
        for c in active_constraints:
            lines.append(f"- **[{c['id']}]** {c['rule']}")
        lines.append("")

    # 2. CURRENT STATE
    latest_entries = sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True)[:3]
    lines.append("## Current State")
    lines.append("")

    # Find the most recent session number
    sessions = [e.get("session") for e in entries if e.get("session")]
    current_session = max(sessions) if sessions else "unknown"
    lines.append(f"**Session:** {current_session}")
    lines.append("")

    # Find the most recent stage-related entry (by session, then timestamp)
    stage_entries = [e for e in entries if any(tag.startswith("stage") for tag in e.get("tags", []))]
    if stage_entries:
        stage_entries_sorted = sorted(stage_entries, key=lambda e: (e.get("session") or 0, e.get("timestamp", "")))
        latest_stage = stage_entries_sorted[-1]
        stage_tags = [t for t in latest_stage.get("tags", []) if t.startswith("stage")]
        if stage_tags:
            lines.append(f"**Stage:** {stage_tags[0]}")
            lines.append(f"**Status:** {latest_stage.get('status', 'unknown')} — {latest_stage['title']}")
            lines.append("")

    # 3. KEY DECISIONS
    decisions = [e for e in entries if e["type"] == "decision"]
    if decisions:
        lines.append("## Key Decisions")
        lines.append("")
        for decision in decisions:
            lines.append(format_decision(decision))
            lines.append("")

    # 4. RECENT FINDINGS (last 2 sessions)
    recent_sessions = sorted(set(sessions))[-2:] if len(set(sessions)) >= 2 else sessions
    recent_experiments = [e for e in entries if e.get("session") in recent_sessions and e["type"] == "experiment"]
    recent_discoveries = [e for e in entries if e.get("session") in recent_sessions and e["type"] == "discovery"]

    if recent_experiments:
        lines.append("## Recent Experiments")
        lines.append("")
        for exp in recent_experiments:
            lines.append(format_experiment(exp))
            lines.append("")

    if recent_discoveries:
        lines.append("## Recent Discoveries")
        lines.append("")
        for disc in recent_discoveries:
            lines.append(format_discovery(disc))
            lines.append("")

    # 5. HISTORICAL SUMMARY (older experiments compressed)
    old_sessions = sorted(set(sessions))[:-2] if len(set(sessions)) > 2 else []
    if old_sessions:
        lines.append("## Historical Summary")
        lines.append("")
        # Include experiments and analysis entries for history
        old_work = [e for e in entries if e.get("session") in old_sessions
                    and e["type"] in ("experiment", "analysis")]
        # Group by session
        by_session = defaultdict(list)
        for entry in old_work:
            by_session[entry.get("session")].append(entry)

        for session in sorted(by_session.keys()):
            exps = by_session[session]
            failed = [e for e in exps if e.get("status") == "failed"]
            partial = [e for e in exps if e.get("status") == "partial"]
            passed = [e for e in exps if e.get("status") in ("passed", "succeeded")]
            active = [e for e in exps if e.get("status") == "active"]

            summary_parts = []
            if failed:
                summary_parts.append(f"{len(failed)} failed")
            if partial:
                summary_parts.append(f"{len(partial)} partial")
            if passed:
                summary_parts.append(f"{len(passed)} passed")
            if active:
                summary_parts.append(f"{len(active)} active")

            lines.append(f"**Session {session}:** {', '.join(summary_parts)}")

            # List failed experiments (most important)
            for exp in failed:
                lines.append(f"  - {exp['title']}: FAIL. {exp.get('content', {}).get('root_cause', '')}")

        lines.append("")

    # 6. ARCHITECTURE
    arch_entries = [e for e in entries if e["type"] == "architecture"]
    if arch_entries:
        lines.append("## Architecture")
        lines.append("")
        # Active substrate first, archived next, other entries last
        active = [e for e in arch_entries if e.get("status") == "active-substrate"]
        archived = [e for e in arch_entries if e.get("status") == "archived"]
        other = [e for e in arch_entries if e.get("status") not in ("active-substrate", "archived")]
        for arch in active + archived + other:
            lines.append(format_architecture(arch))
            lines.append("")

    # 7. META-COGNITIVE REVIEW (if due)
    progress = load_progress()
    if progress:
        last_review = progress.get('last_review_session', 0)
        review_interval = progress.get('review_interval', 3)
        sessions_since_review = current_session - last_review if isinstance(current_session, int) else 0

        if sessions_since_review >= review_interval:
            # Review is due
            lines.append(generate_review_section(progress, current_session))
        else:
            # Not yet due
            next_review_session = last_review + review_interval
            lines.append(f"*Next meta-cognitive review due: Session {next_review_session}*")
            lines.append("")

    return "\n".join(lines)

def main():
    print("Compiling knowledge base...")

    # Load all entries
    entries = load_entries()
    print(f"Loaded {len(entries)} entries")

    # Extract constraints
    constraints = extract_constraints(entries)
    print(f"Extracted {len(constraints)} constraints")

    # Build index
    index = build_index(entries, constraints)

    # Generate state.md
    state_md = generate_state_md(entries, constraints)

    # Write outputs
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        f.write(state_md)
    print(f"Written {STATE_FILE}")

    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Written {INDEX_FILE}")

    with open(CONSTRAINTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(constraints, f, indent=2, ensure_ascii=False)
    print(f"Written {CONSTRAINTS_FILE}")

    # Print stats
    line_count = state_md.count('\n') + 1
    print(f"\nstate.md: {line_count} lines")
    if line_count > 100:
        print(f"  WARNING: Exceeds 100-line target by {line_count - 100} lines")

    print("\nCompilation complete.")

if __name__ == "__main__":
    main()
