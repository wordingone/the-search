#!/usr/bin/env python3
"""
ingest.py — Entry creation helper for the knowledge system.
Creates properly formatted JSON entries with auto-incrementing IDs and timestamps.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

ENTRIES_DIR = Path(__file__).parent / "entries"
ENTRIES_DIR.mkdir(exist_ok=True)

VALID_TYPES = ["experiment", "decision", "discovery", "constraint", "architecture"]
VALID_STATUSES = ["succeeded", "failed", "partial", "active", "superseded"]

def get_next_id():
    """Get next available ID by scanning existing entries."""
    existing = list(ENTRIES_DIR.glob("*.json"))
    if not existing:
        return "001"
    ids = [int(p.stem) for p in existing if p.stem.isdigit()]
    return f"{max(ids) + 1:03d}"

def validate_entry(entry):
    """Validate entry structure and required fields."""
    if "type" not in entry:
        raise ValueError("Entry must have 'type' field")
    if entry["type"] not in VALID_TYPES:
        print(f"Warning: unknown type '{entry['type']}'. Valid types: {VALID_TYPES}", file=sys.stderr)

    if "status" in entry and entry["status"] not in VALID_STATUSES:
        print(f"Warning: unknown status '{entry['status']}'. Valid statuses: {VALID_STATUSES}", file=sys.stderr)

    if "title" not in entry:
        raise ValueError("Entry must have 'title' field")

    if "content" not in entry:
        raise ValueError("Entry must have 'content' field")

def create_entry(data):
    """Create entry with auto-generated ID and timestamp."""
    entry = {
        "id": get_next_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data
    }

    # Set defaults
    entry.setdefault("tags", [])
    entry.setdefault("references", [])
    entry.setdefault("constraints_implied", [])

    validate_entry(entry)

    # Write to file
    filepath = ENTRIES_DIR / f"{entry['id']}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    print(f"Created entry {entry['id']}: {entry['title']}")
    return entry['id']

def interactive_mode():
    """Interactive entry creation."""
    print("=== Knowledge Entry Creation ===")
    print(f"Valid types: {', '.join(VALID_TYPES)}")
    print(f"Valid statuses: {', '.join(VALID_STATUSES)}\n")

    entry_type = input("Type: ").strip()
    title = input("Title: ").strip()
    status = input("Status (optional): ").strip() or None
    tags = input("Tags (comma-separated, optional): ").strip()
    tags = [t.strip() for t in tags.split(",")] if tags else []

    print("\nContent fields (JSON format). Enter empty line when done.")
    print("Example: {\"hypothesis\": \"...\", \"result\": \"...\"}")
    content_lines = []
    while True:
        line = input()
        if not line:
            break
        content_lines.append(line)

    try:
        content = json.loads("\n".join(content_lines))
    except json.JSONDecodeError as e:
        print(f"Error parsing content JSON: {e}", file=sys.stderr)
        sys.exit(1)

    data = {
        "type": entry_type,
        "title": title,
        "tags": tags,
        "content": content
    }
    if status:
        data["status"] = status

    return create_entry(data)

def main():
    parser = argparse.ArgumentParser(description="Create knowledge base entries")
    parser.add_argument("--from", dest="from_file", help="Create from JSON file")
    parser.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    parser.add_argument("--batch", help="Create multiple entries from JSON array file")

    args = parser.parse_args()

    if args.from_file:
        with open(args.from_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        create_entry(data)

    elif args.stdin:
        data = json.load(sys.stdin)
        create_entry(data)

    elif args.batch:
        with open(args.batch, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            print("Batch file must contain JSON array", file=sys.stderr)
            sys.exit(1)
        for data in entries:
            create_entry(data)

    else:
        interactive_mode()

if __name__ == "__main__":
    main()
