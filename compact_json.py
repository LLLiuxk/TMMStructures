"""
Compact JSON Reformatter for dataset_schema.json
=================================================
Reads an existing dataset_schema.json (with one-number-per-line formatting)
and rewrites it using a compact encoder that keeps numeric arrays on single lines.

Usage:
    python compact_json.py <path_to_dataset_schema.json>

Example:
    python compact_json.py Output/dataset/batch_1/dataset_schema.json
"""

import json
import sys
import os
import shutil


class CompactJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that keeps short lists (e.g. matrix rows) on a single line."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indent_level = 0

    def encode(self, o):
        return self._encode(o, self._indent_level)

    def _encode(self, o, indent_level):
        indent_str = "  " * indent_level
        child_indent = "  " * (indent_level + 1)

        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            for k, v in o.items():
                val = self._encode(v, indent_level + 1)
                items.append(f"{child_indent}{json.dumps(k)}: {val}")
            return "{\n" + ",\n".join(items) + "\n" + indent_str + "}"

        if isinstance(o, list):
            if not o:
                return "[]"
            # If all elements are simple scalars (numbers, strings, bools, None), put on one line
            if all(isinstance(x, (int, float, bool, type(None), str)) for x in o):
                return "[" + ", ".join(json.dumps(x) for x in o) + "]"
            # If this is a list-of-lists where inner lists are all simple (e.g. matrix), one row per line
            if all(isinstance(x, list) and all(isinstance(y, (int, float, bool, type(None))) for y in x) for x in o):
                rows = ["[" + ", ".join(json.dumps(y) for y in row) + "]" for row in o]
                return "[\n" + ",\n".join(child_indent + r for r in rows) + "\n" + indent_str + "]"
            # Otherwise use multi-line format
            items = []
            for item in o:
                items.append(child_indent + self._encode(item, indent_level + 1))
            return "[\n" + ",\n".join(items) + "\n" + indent_str + "]"

        return json.dumps(o)


def reformat_json(filepath):
    """Reformat a dataset_schema.json file using compact encoding."""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    # Get file size before
    size_before = os.path.getsize(filepath) / (1024 * 1024)  # MB

    print(f"Loading {filepath}...")
    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"  Records: {len(data)}")

    # Create backup
    backup_path = filepath + ".bak"
    shutil.copy2(filepath, backup_path)
    print(f"  Backup created: {backup_path}")

    # Write compacted version
    print(f"  Writing compact format...")
    with open(filepath, "w") as f:
        f.write(CompactJSONEncoder().encode(data))

    size_after = os.path.getsize(filepath) / (1024 * 1024)  # MB
    ratio = (1 - size_after / size_before) * 100 if size_before > 0 else 0

    print(f"  Size: {size_before:.1f} MB → {size_after:.1f} MB  ({ratio:.1f}% reduction)")
    print(f"  Done!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compact_json.py <path_to_dataset_schema.json>")
        print("Example: python compact_json.py Output/dataset/batch_1/dataset_schema.json")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\n{'=' * 50}")
        reformat_json(path)
    print(f"\n{'=' * 50}")
    print("All files processed.")
