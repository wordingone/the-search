#!/usr/bin/env python3
"""
test_system.py — Verify the knowledge system works correctly.
"""

import json
import sys
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent
ENTRIES_DIR = KNOWLEDGE_DIR / "entries"
STATE_FILE = KNOWLEDGE_DIR / "state.md"
INDEX_FILE = KNOWLEDGE_DIR / "index.json"
CONSTRAINTS_FILE = KNOWLEDGE_DIR / "constraints.json"

def test_files_exist():
    """Verify all expected files exist."""
    print("Testing file structure...")
    assert ENTRIES_DIR.exists(), "entries/ directory missing"
    assert STATE_FILE.exists(), "state.md missing"
    assert INDEX_FILE.exists(), "index.json missing"
    assert CONSTRAINTS_FILE.exists(), "constraints.json missing"
    print("  [OK] All files exist")

def test_entry_count():
    """Verify entry count matches expectations."""
    print("Testing entry count...")
    entries = list(ENTRIES_DIR.glob("*.json"))
    assert len(entries) == 18, f"Expected 18 entries, found {len(entries)}"
    print(f"  [OK] Found {len(entries)} entries")

def test_entry_validity():
    """Verify all entries are valid JSON with required fields."""
    print("Testing entry validity...")
    required_fields = ["id", "timestamp", "type", "title", "content"]

    for filepath in sorted(ENTRIES_DIR.glob("*.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            entry = json.load(f)

        for field in required_fields:
            assert field in entry, f"Entry {filepath.name} missing field: {field}"

        # Verify ID matches filename
        expected_id = filepath.stem
        assert entry["id"] == expected_id, f"ID mismatch in {filepath.name}: {entry['id']} != {expected_id}"

    print(f"  [OK] All entries valid")

def test_constraint_extraction():
    """Verify constraints were extracted correctly."""
    print("Testing constraint extraction...")

    with open(CONSTRAINTS_FILE, 'r', encoding='utf-8') as f:
        constraints = json.load(f)

    assert len(constraints) == 9, f"Expected 9 constraints, found {len(constraints)}"

    # Verify all constraints have required fields
    for c in constraints:
        assert "id" in c and "rule" in c and "source_entries" in c, f"Constraint {c.get('id', '?')} missing fields"
        assert c["id"].startswith("c"), f"Constraint ID should start with 'c': {c['id']}"

    print(f"  [OK] {len(constraints)} constraints extracted correctly")

def test_index_structure():
    """Verify index.json has correct structure."""
    print("Testing index structure...")

    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        index = json.load(f)

    required_keys = ["project", "compiled_at", "entry_count", "active_constraints",
                     "entries_by_type", "entries_by_tag", "entries_by_status"]

    for key in required_keys:
        assert key in index, f"Index missing key: {key}"

    assert index["entry_count"] == 18, f"Entry count mismatch: {index['entry_count']} != 18"
    assert index["active_constraints"] == 9, f"Constraint count mismatch: {index['active_constraints']} != 9"

    # Verify entry type categories
    assert "experiment" in index["entries_by_type"], "Missing experiment entries"
    assert "discovery" in index["entries_by_type"], "Missing discovery entries"
    assert "decision" in index["entries_by_type"], "Missing decision entries"

    print("  [OK] Index structure valid")

def test_state_md_format():
    """Verify state.md has correct structure."""
    print("Testing state.md format...")

    with open(STATE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Verify required sections
    required_sections = [
        "# Project Knowledge State",
        "## Active Constraints",
        "## Current State",
        "## Key Decisions",
        "## Recent Experiments",
        "## Recent Discoveries",
        "## Architecture"
    ]

    for section in required_sections:
        assert section in content, f"Missing section: {section}"

    # Verify line count (should be around 100)
    line_count = content.count('\n') + 1
    assert line_count <= 110, f"state.md too long: {line_count} lines (target: 100)"

    # Verify constraints are listed
    assert "[c001]" in content, "Constraint c001 not found in state.md"
    assert "[c009]" in content, "Constraint c009 not found in state.md"

    print(f"  [OK] state.md format valid ({line_count} lines)")

def test_constraint_traceability():
    """Verify constraints trace back to source entries."""
    print("Testing constraint traceability...")

    with open(CONSTRAINTS_FILE, 'r', encoding='utf-8') as f:
        constraints = json.load(f)

    for c in constraints:
        for source_id in c["source_entries"]:
            source_file = ENTRIES_DIR / f"{source_id}.json"
            assert source_file.exists(), f"Constraint {c['id']} references missing entry {source_id}"

    print("  [OK] All constraints traceable to source entries")

def test_entry_types():
    """Verify all expected entry types are present."""
    print("Testing entry type coverage...")

    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        index = json.load(f)

    expected_types = {"experiment", "discovery", "decision", "architecture"}
    actual_types = set(index["entries_by_type"].keys())

    assert expected_types.issubset(actual_types), f"Missing entry types: {expected_types - actual_types}"

    print(f"  [OK] All expected entry types present: {', '.join(sorted(actual_types))}")

def test_session_coverage():
    """Verify entries cover Sessions 1-6."""
    print("Testing session coverage...")

    sessions = set()
    for filepath in ENTRIES_DIR.glob("*.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            entry = json.load(f)
            if "session" in entry:
                sessions.add(entry["session"])

    expected_sessions = {1, 2, 3, 4, 5, 6}
    assert expected_sessions == sessions, f"Session coverage mismatch: {sessions} != {expected_sessions}"

    print(f"  [OK] Sessions 1-6 covered")

def main():
    """Run all tests."""
    tests = [
        test_files_exist,
        test_entry_count,
        test_entry_validity,
        test_constraint_extraction,
        test_index_structure,
        test_state_md_format,
        test_constraint_traceability,
        test_entry_types,
        test_session_coverage,
    ]

    print("=" * 60)
    print("Knowledge System Test Suite")
    print("=" * 60)
    print()

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  [FAIL] FAILED: {e}")
            failed.append((test.__name__, str(e)))
        except Exception as e:
            print(f"  [FAIL] ERROR: {e}")
            failed.append((test.__name__, str(e)))

    print()
    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests failed")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed")
        print()
        print("The knowledge system is fully functional:")
        print("  - 18 entries from Sessions 1-6")
        print("  - 9 active constraints")
        print("  - 100-line state.md")
        print("  - Complete index and constraint traceability")
        sys.exit(0)

if __name__ == "__main__":
    main()
