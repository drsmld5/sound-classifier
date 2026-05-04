# main.py
import sys
import json
from dataclasses import asdict
from pipeline.scanner import scan_folder
from pipeline.aggregator import classify_track, classify_batch


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path> [--analyze] [--json]")
        sys.exit(1)

    folder = sys.argv[1]
    do_analyze = '--analyze' in sys.argv or '--analyse' in sys.argv
    do_json = '--json' in sys.argv

    print(f"\nScanning: {folder}\n")

    try:
        tracks = scan_folder(folder)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not tracks:
        print("No audio files found.")
        sys.exit(0)

    if not do_analyze:
        # Scanner only — fast metadata read
        for track in tracks:
            print(f"  {track.relative_path}  [{track.size_mb} MB]")
            if track.duration_seconds:
                mins = int(track.duration_seconds // 60)
                secs = int(track.duration_seconds % 60)
                print(f"    Duration : {mins}:{secs:02d}")
            if track.artist:
                print(f"    Artist   : {track.artist}")
            if track.title:
                print(f"    Title    : {track.title}")
            print()
        print(f"Found {len(tracks)} audio file(s)")
        return

    # Full pipeline — analyze all tracks
    print(f"Analyzing {len(tracks)} track(s)...\n")

    def on_progress(current, total, result):
        status = "✓" if not result.has_errors else "⚠"
        print(f"  [{current}/{total}] {status}  {result.summary()}")
        if result.has_errors:
            for err in result.errors:
                print(f"         ✗ {err.module}: {err.error}")

    results = classify_batch(tracks, on_progress=on_progress)

    # Summary
    complete = sum(1 for r in results if r.is_complete)
    errors = sum(1 for r in results if r.has_errors)
    total_time = sum(r.processing_time_seconds for r in results)

    print(f"\n{'─' * 60}")
    print(f"  Tracks analyzed : {len(results)}")
    print(f"  Fully complete  : {complete}")
    print(f"  With errors     : {errors}")
    print(f"  Total time      : {total_time:.1f}s")
    print(f"  Avg per track   : {total_time / len(results):.1f}s")
    print(f"{'─' * 60}\n")

    if do_json:
        print(json.dumps(
            [asdict(r) for r in results],
            indent=2,
            ensure_ascii=False
        ))


if __name__ == '__main__':
    main()