import argparse
import json
import os
import re
import csv
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Whisper model performance based on meeting detection results.",
        epilog="Example: python compare_models.py --report detection_report.json"
    )
    parser.add_argument(
        "--report",
        default="detection_report.json",
        help="Path to the detection report JSON file (default: detection_report.json)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to model_comparison.csv",
    )
    return parser.parse_args()

def load_report(filepath):
    """Load the detection report JSON."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_filename(filepath):
    """
    Extracts (audio_id, model_name) from a transcript filepath.
    Assumes format: .../YYYYMMDD_HHMMSS-HHMMSS_location_transcript_{model}.txt
    """
    basename = os.path.basename(filepath)
    # Regex to find the part before "_transcript_" and the part after
    match = re.match(r"(.*)_transcript_(.*)\.txt$", basename)
    if match:
        audio_id = match.group(1)
        model_part = match.group(2)
        return audio_id, model_part
    return None, None

def analyze_models(data, export_csv=False):
    """
    Analyzes the detection data to compare models.
    """
    # 1. Organize data
    # audio_id -> { model_name -> result_entry }
    audio_map = defaultdict(dict)
    all_models = set()

    for entry in data:
        audio_id, model = parse_filename(entry["file"])
        if audio_id and model:
            audio_map[audio_id][model] = entry
            all_models.add(model)

    if not audio_map:
        print("No valid transcript files found in report.")
        return

    # 2. Establish "Optimistic Ground Truth"
    # If ANY model detects a meeting with High Confidence (>80%), assume the file contains a meeting.
    confirmed_meeting_files = set()
    
    for audio_id, models_data in audio_map.items():
        for model, entry in models_data.items():
            if entry.get("has_meeting") and entry.get("confidence", 0) >= 80:
                confirmed_meeting_files.add(audio_id)
                break
    
    print(f"\n{'='*60}")
    print(f"TRANSCRIPT QUALITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total Unique Audio Segments: {len(audio_map)}")
    print(f"Segments with Confirmed Meetings: {len(confirmed_meeting_files)}")
    print(f"  (Determined by consensus/high-confidence detection)")
    print("-" * 60)

    # 3. Calculate Metrics per Model
    model_stats = defaultdict(lambda: {"total": 0, "meetings_found": 0})
    
    stats_list = []

    for model in all_models:
        stats = {"model": model, "total": 0, "true_positives": 0, "false_negatives": 0, "hallucinations": 0}
        
        # Only examine files where this model actually produced a transcript
        model_audio_ids = [aid for aid in audio_map if model in audio_map[aid]]
        stats["total"] = len(model_audio_ids)
        
        if stats["total"] == 0:
            continue

        confirmed_runs = 0
        found_in_confirmed = 0
        
        for audio_id in model_audio_ids:
            entry = audio_map[audio_id][model]
            has_meeting = entry.get("has_meeting")
            
            # Metric 1: Discovery Rate (Recall on Hidden Meetings)
            if audio_id in confirmed_meeting_files:
                confirmed_runs += 1
                if has_meeting:
                    found_in_confirmed += 1
                else:
                    stats["false_negatives"] += 1
            
            # Metric 2: Hallucination Rate (Raw)
            if not has_meeting:
                stats["hallucinations"] += 1

        # Discovery Rate: % of "Real Meetings" this model found
        discovery_rate = (found_in_confirmed / confirmed_runs * 100) if confirmed_runs > 0 else 0.0
        
        # Hallucination Rate: % of TOTAL outputs that were non-meetings (hallucinations/silence)
        # Low hallucination rate might just mean it finds "everything" as a meeting (over-sensitive).
        # So Discovery Rate is the critical metric for "Hidden Meetings".
        hallucination_rate = (stats["hallucinations"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        
        # Accuracy Score (Weighted)
        # For this user, finding hidden meetings is key. 
        # Score = Discovery Rate
        accuracy_score = discovery_rate

        stats_list.append({
            "model": model,
            "total": stats["total"],
            "meetings_found": found_in_confirmed, # Actual True Positives
            "discovery_rate": discovery_rate,
            "hallucination_rate": hallucination_rate,
            "score": accuracy_score
        })

    # Sort by Score (Desc)
    stats_list.sort(key=lambda x: x["score"], reverse=True)

    # 4. Format Output
    print(f"{'Model':<20} | {'Files':<6} | {'Recall %':<10} | {'Halluc. %':<10} | {'Quality Score':<12}")
    print(f"{'-'*20}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    
    for s in stats_list:
        score_str = f"{s['score']:.1f}/100"
        print(f"{s['model']:<20} | {s['total']:<6} | {s['discovery_rate']:<10.1f} | {s['hallucination_rate']:<10.1f} | {score_str:<12}")

    print("-" * 60)
    
    # 5. Recommendation
    if stats_list:
        best_model = stats_list[0]
        print(f"\n🏆 RECOMMENDED MODEL: {best_model['model']}")
        print(f"   Reason: Highest Discovery Rate ({best_model['discovery_rate']:.1f}%) on confirmed meetings.")
        print(f"   This model is best at finding content that other models might miss.")

    # 6. Export CSV
    if export_csv:
        csv_file = "model_comparison.csv"
        try:
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=stats_list[0].keys())
                writer.writeheader()
                writer.writerows(stats_list)
            print(f"\n[INFO] Detailed stats exported to {csv_file}")
        except Exception as e:
            print(f"[ERROR] Failed to export CSV: {e}")

if __name__ == "__main__":
    args = parse_args()
    data = load_report(args.report)
    if data:
        analyze_models(data, export_csv=args.csv)
