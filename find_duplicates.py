"""
Find Duplicate Files
====================
Scans a folder (recursively) for duplicate files using MD5 hashing,
generates a CSV report, and optionally deletes the duplicates.

Usage:
    python find_duplicates.py [folder_path]

If no folder path is given, the current directory is used.
"""

import os
import sys
import csv
import hashlib
from collections import defaultdict


def hash_file(filepath, max_bytes=None, chunk_size=65536):
    """Compute SHA-256 hash of a file (or just the first max_bytes)."""
    hasher = hashlib.sha256()
    bytes_read = 0
    try:
        with open(filepath, "rb") as f:
            while True:
                to_read = chunk_size
                if max_bytes is not None:
                    to_read = min(chunk_size, max_bytes - bytes_read)
                    if to_read <= 0:
                        break
                chunk = f.read(to_read)
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)
    except (PermissionError, OSError) as e:
        print(f"  [SKIP] Cannot read: {filepath}  ({e})")
        return None
    return hasher.hexdigest()


PARTIAL_HASH_SIZE = 10 * 64 * 1024  # 640 KB — first 10 blocks, fast but thorough


def format_size(size_bytes):
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def find_duplicates(root_folder):
    """Walk the folder tree using 3-phase detection (size → partial hash → full hash)."""
    print(f"\nScanning: {root_folder}\n")

    # --- Phase 1: group files by size (instant) ---
    size_map = defaultdict(list)
    file_count = 0
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                continue
            size_map[fsize].append(fpath)
            file_count += 1

    print(f"  Found {file_count:,} files total.")

    candidates = {sz: paths for sz, paths in size_map.items() if len(paths) > 1}
    candidate_count = sum(len(p) for p in candidates.values())
    print(f"  {candidate_count:,} files share a size with at least one other file.")

    # --- Phase 2: partial hash (first 640 KB only — very fast) ---
    print(f"  Phase 2: partial hashing (first 640 KB) on {candidate_count:,} files...")
    partial_map = defaultdict(list)
    hashed = 0
    for sz, paths in candidates.items():
        for fpath in paths:
            h = hash_file(fpath, max_bytes=PARTIAL_HASH_SIZE)
            if h is not None:
                partial_map[(sz, h)].append(fpath)
            hashed += 1
            if hashed % 500 == 0:
                print(f"    ... {hashed:,} / {candidate_count:,} partial-hashed", end="\r")
    print(f"    ... {hashed:,} / {candidate_count:,} partial-hashed — done.")

    partial_dups = {k: v for k, v in partial_map.items() if len(v) > 1}
    partial_count = sum(len(v) for v in partial_dups.values())
    print(f"  {partial_count:,} files still match after partial hash.")

    # --- Phase 3: full hash (only files that survived partial hash) ---
    print(f"  Phase 3: full hashing {partial_count:,} remaining candidates...")
    full_map = defaultdict(list)
    hashed = 0
    for (sz, _phash), paths in partial_dups.items():
        for fpath in paths:
            h = hash_file(fpath)  # full file hash
            if h is not None:
                full_map[(sz, h)].append(fpath)
            hashed += 1
            if hashed % 50 == 0:
                print(f"    ... {hashed:,} / {partial_count:,} full-hashed", end="\r")
    print(f"    ... {hashed:,} / {partial_count:,} full-hashed — done.")

    duplicate_groups = {k: v for k, v in full_map.items() if len(v) > 1}
    return duplicate_groups


def write_csv_report(duplicate_groups, csv_path):
    """Write a CSV listing every duplicate group with file sizes."""
    rows = []
    group_id = 0
    for (size_bytes, _hash), paths in sorted(duplicate_groups.items(), key=lambda x: -x[0][0]):
        group_id += 1
        # First file in the group is treated as the "original"
        original = paths[0]
        for dup in paths[1:]:
            rows.append({
                "Group": group_id,
                "Original File": original,
                "Duplicate File": dup,
                "File Size (bytes)": size_bytes,
                "File Size": format_size(size_bytes),
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Group", "Original File", "Duplicate File",
            "File Size (bytes)", "File Size",
        ])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def prompt_delete(duplicate_groups):
    """Interactively ask the user which duplicate groups to delete."""
    total_duplicates = sum(len(v) - 1 for v in duplicate_groups.values())
    total_saved = sum((len(v) - 1) * k[0] for k, v in duplicate_groups.items())

    print(f"\n{'=' * 60}")
    print(f"  Found {len(duplicate_groups)} duplicate group(s)")
    print(f"  {total_duplicates} duplicate file(s) consuming {format_size(total_saved)}")
    print(f"{'=' * 60}\n")

    if total_duplicates == 0:
        print("  No duplicates to delete. All clean!")
        return

    # Show summary per group
    group_id = 0
    groups_list = []
    for (size_bytes, _hash), paths in sorted(duplicate_groups.items(), key=lambda x: -x[0][0]):
        group_id += 1
        groups_list.append((group_id, size_bytes, paths))
        print(f"  Group {group_id}  |  {format_size(size_bytes)} each  |  {len(paths)} copies")
        print(f"    KEEP : {paths[0]}")
        for dup in paths[1:]:
            print(f"    DELETE: {dup}")
        print()

    # Ask user
    print("Options:")
    print("  [A] Delete ALL duplicates (keep first copy of each)")
    print("  [S] Step through each group one by one")
    print("  [N] Do NOT delete anything")
    choice = input("\nYour choice (A/S/N): ").strip().upper()

    if choice == "N":
        print("\n  No files deleted.")
        return

    deleted_count = 0
    freed_bytes = 0

    if choice == "A":
        for gid, size_bytes, paths in groups_list:
            for dup in paths[1:]:
                try:
                    os.remove(dup)
                    deleted_count += 1
                    freed_bytes += size_bytes
                    print(f"  Deleted: {dup}")
                except OSError as e:
                    print(f"  [ERROR] Could not delete {dup}: {e}")

    elif choice == "S":
        for gid, size_bytes, paths in groups_list:
            print(f"\n--- Group {gid} ({format_size(size_bytes)} each) ---")
            print(f"  KEEP : {paths[0]}")
            for dup in paths[1:]:
                print(f"  DELETE: {dup}")
            ans = input("  Delete duplicates in this group? (Y/N): ").strip().upper()
            if ans == "Y":
                for dup in paths[1:]:
                    try:
                        os.remove(dup)
                        deleted_count += 1
                        freed_bytes += size_bytes
                        print(f"    Deleted: {dup}")
                    except OSError as e:
                        print(f"    [ERROR] Could not delete {dup}: {e}")
            else:
                print("    Skipped.")
    else:
        print("  Invalid choice. No files deleted.")
        return

    print(f"\n  Done! Deleted {deleted_count} file(s), freed {format_size(freed_bytes)}.")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    # Find duplicates
    duplicate_groups = find_duplicates(folder)

    if not duplicate_groups:
        print("  No duplicate files found!")
        sys.exit(0)

    # Write CSV report
    csv_path = os.path.join(os.getcwd(), "duplicate_files.csv")
    rows = write_csv_report(duplicate_groups, csv_path)
    print(f"  CSV report saved to: {csv_path}")
    print(f"  ({len(rows)} duplicate entries across {len(duplicate_groups)} groups)")

    # Ask user about deletion
    prompt_delete(duplicate_groups)


if __name__ == "__main__":
    main()
