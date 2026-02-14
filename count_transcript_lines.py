"""
Count total, repeating, and non-repeating lines in transcript files.

Usage:
    python count_transcript_lines.py <folder>
    python count_transcript_lines.py  (defaults to LongAudioApp\\Transcripts)

Outputs a CSV file (transcript_line_counts.csv) with columns:
    Non-Repeating Lines, Total Lines, Repeating Lines, Full Path
"""

import os
import sys
import csv
import re
from collections import Counter

# Default transcript folder
DEFAULT_FOLDER = os.path.join(
    os.environ.get("APPDATA", ""), "LongAudioApp", "Transcripts"
)

# Regex to strip timestamp prefixes like [0.00 - 2.00]
TIMESTAMP_RE = re.compile(r"^\[\d+\.\d+\s*-\s*\d+\.\d+\]\s*")


def analyse_transcript(filepath):
    """Return (total_lines, repeating_lines, non_repeating_lines) for a file.

    Only content lines are counted (header/metadata lines at the top are skipped).
    A 'repeating line' is any line whose text content appears more than once
    in the file. The text is compared after stripping the timestamp prefix.
    """
    content_lines = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        in_content = False
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            # Content starts at the first timestamped line
            if not in_content:
                if TIMESTAMP_RE.match(line):
                    in_content = True
                else:
                    continue
            if line == "":
                continue
            # Strip timestamp prefix to get the spoken text
            text = TIMESTAMP_RE.sub("", line).strip()
            if text:
                content_lines.append(text)

    total = len(content_lines)
    counts = Counter(content_lines)
    non_repeating = sum(1 for t, c in counts.items() if c == 1)
    repeating = total - non_repeating  # all occurrences of repeated text

    return total, repeating, non_repeating


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER

    if not os.path.isdir(folder):
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    # Collect transcript .txt files
    txt_files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    )

    if not txt_files:
        print(f"No .txt files found in {folder}")
        sys.exit(1)

    print(f"Analysing {len(txt_files)} transcripts in: {folder}\n")

    results = []
    for filepath in txt_files:
        total, repeating, non_repeating = analyse_transcript(filepath)
        results.append((non_repeating, total, repeating, filepath))

    # Sort by non-repeating count descending (most unique content first)
    results.sort(key=lambda r: r[0], reverse=True)

    # Write CSV
    out_csv = os.path.join(os.getcwd(), "transcript_line_counts.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Non-Repeating Lines", "Total Lines", "Repeating Lines", "Full Path"])
        writer.writerows(results)

    # Print summary
    print(f"{'Non-Repeat':>10}  {'Total':>6}  {'Repeat':>6}  File")
    print("-" * 80)
    for non_rep, total, rep, path in results[:20]:
        print(f"{non_rep:>10}  {total:>6}  {rep:>6}  {os.path.basename(path)}")
    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more (see CSV)")

    print(f"\nCSV saved to: {out_csv}")
    print(f"Total files analysed: {len(results)}")


if __name__ == "__main__":
    main()
