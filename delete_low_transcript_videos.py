"""
Delete video files whose transcripts have few non-repeating lines.

Reads transcript_line_counts.csv and deletes the corresponding .mp4 video
for every transcript with fewer than --min-lines non-repeating lines (default 10).

Video filename convention (from timestamp_engine.py):
    YYMMDD-DoW-HHMM-HHMM-ampm-camera_location.mp4
    e.g. 260201-SUN-0725-0759-am-fence1104.mp4

Transcript filename convention:
    YYYY-MM-DD-DoW-HH-MM-SSampm-HH-MM-SSampm-camera_transcript_base.en.txt
    e.g. 2026-02-01-SUN-07-25-49am-07-59-58am-fence1104_transcript_base.en.txt

Usage:
    python delete_low_transcript_videos.py                          # dry-run, threshold 10
    python delete_low_transcript_videos.py --min-lines 5            # threshold 5
    python delete_low_transcript_videos.py --delete                 # actually delete
    python delete_low_transcript_videos.py --video-dir "E:\\"       # custom video root
"""

import argparse
import csv
import os
import re
import sys


# Suffix appended by the transcription pipeline
TRANSCRIPT_SUFFIX = "_transcript_base.en.txt"

# Regex to parse the transcript filename stem
# e.g.  2026-02-01-SUN-07-25-49am-07-59-58am-fence1104
TRANSCRIPT_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})-(\w{3})-"       # YYYY-MM-DD-DoW
    r"(\d{2})-(\d{2})-\d{2}(am|pm)-"           # HH-MM-SS(am|pm)  start
    r"(\d{2})-(\d{2})-\d{2}(am|pm)-"           # HH-MM-SS(am|pm)  end
    r"(.+)"                                     # camera_location
)


def build_video_index(video_dir):
    """Walk video_dir and return {lowercase_basename: full_path} for all .mp4 files."""
    index = {}
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                index[f.lower()] = os.path.join(root, f)
    return index


def transcript_to_video_name(transcript_path):
    """Convert a transcript full-path to the expected video basename.

    Transcripts share the same naming pattern as videos, so we just swap the
    suffix.  Falls back to regex conversion for old long-format names.

    New format:  260201-SUN-0725-0759-am-fence1104_transcript_base.en.txt
              -> 260201-SUN-0725-0759-am-fence1104.mp4

    Old format:  2026-02-01-SUN-07-25-49am-07-59-58am-fence1104_transcript_base.en.txt
              -> 260201-SUN-0725-0759-am-fence1104.mp4
    """
    basename = os.path.basename(transcript_path)

    # Strip transcript suffix
    if basename.endswith(TRANSCRIPT_SUFFIX):
        stem = basename[: -len(TRANSCRIPT_SUFFIX)]
    else:
        stem = os.path.splitext(basename)[0]

    # Try old long-format conversion first (YYYY-MM-DD-DoW-HH-MM-SSampm-...)
    m = TRANSCRIPT_RE.match(stem)
    if m:
        yyyy, mm, dd, dow = m.group(1), m.group(2), m.group(3), m.group(4)
        s_hh, s_mm, s_ampm = m.group(5), m.group(6), m.group(7)
        e_hh, e_mm, e_ampm = m.group(8), m.group(9), m.group(10)
        camera = m.group(11)
        yy = yyyy[2:]
        return f"{yy}{mm}{dd}-{dow}-{s_hh}{s_mm}-{e_hh}{e_mm}-{s_ampm}-{camera}.mp4"

    # New format: stem already matches the video name, just add .mp4
    return stem + ".mp4"


def main():
    parser = argparse.ArgumentParser(
        description="Delete videos whose transcripts have few non-repeating lines."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "transcript_line_counts.csv"),
        help="Path to transcript_line_counts.csv (default: same dir as script)",
    )
    parser.add_argument(
        "--video-dir",
        default="E:\\",
        help="Root directory containing video files (default: E:\\)",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=10,
        help="Delete videos with fewer than this many non-repeating lines (default: 10)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files. Without this flag the script only does a dry run.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    # Build index of videos on the target drive
    print(f"Scanning videos in {args.video_dir} ...")
    video_index = build_video_index(args.video_dir)
    print(f"Found {len(video_index)} .mp4 files.\n")

    # Read the CSV
    candidates = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            non_rep = int(row["Non-Repeating Lines"])
            transcript = row["Full Path"]
            if non_rep < args.min_lines:
                video_name = transcript_to_video_name(transcript)
                if video_name:
                    video_path = video_index.get(video_name.lower())
                    candidates.append((non_rep, transcript, video_name, video_path))
                else:
                    candidates.append((non_rep, transcript, "PARSE_FAILED", None))

    if not candidates:
        print(f"No transcripts found with fewer than {args.min_lines} non-repeating lines.")
        return

    # Report
    mode = "DELETE" if args.delete else "DRY RUN"
    print(f"[{mode}]  Threshold: non-repeating lines < {args.min_lines}")
    print(f"{'Non-Rep':>8}  {'Status':>12}  Video")
    print("-" * 100)

    deleted = 0
    would_delete = 0
    not_found = 0
    errors = 0

    for non_rep, transcript, video_name, video_path in candidates:
        if video_path is None:
            status = "NOT FOUND"
            not_found += 1
            display = video_name
        elif args.delete:
            try:
                os.remove(video_path)
                status = "DELETED"
                deleted += 1
                display = video_path
            except OSError as e:
                status = f"ERROR"
                errors += 1
                display = f"{video_path}  ({e})"
        else:
            status = "WOULD DEL"
            would_delete += 1
            display = video_path

        print(f"{non_rep:>8}  {status:>12}  {display}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total candidates:    {len(candidates)}")
    if args.delete:
        print(f"Deleted:             {deleted}")
        print(f"Errors:              {errors}")
    else:
        print(f"Would delete:        {would_delete}")
    print(f"Video not found:     {not_found}")

    if not args.delete and would_delete > 0:
        print(f"\nRe-run with --delete to actually remove the files.")


if __name__ == "__main__":
    main()
