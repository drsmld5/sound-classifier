# main.py
# Phase 1 CLI — run directly to validate the scanner pipeline
# Usage: python main.py /path/to/your/music/folder

import sys
import json
from dataclasses import asdict

from pip._internal.resolution.resolvelib import factory

from pipeline.scanner import scan_folder


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    print(f"\nScanning: {folder}\n")

    try:
        tracks = scan_folder(folder)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not tracks:
        print("No audio files found.")
        sys.exit(0)

    for track in tracks:
        print(f"  {track.relative_path}")
        if track.duration_seconds:
            mins = int(track.duration_seconds // 60)
            secs = int(track.duration_seconds % 60)
            print(f"    Duration : {mins}:{secs:02d}")
        if track.embedded_bpm:
            print(f"    BPM      : {track.embedded_bpm}")
        if track.embedded_key:
            print(f"    Key      : {track.embedded_key}")
        if track.embedded_genre:
            print(f"    Genre    : {track.embedded_genre}")
        if track.artist:
            print(f"    Artist   : {track.artist}")
        print()

    print(f"Found {len(tracks)} audio file(s)")

    # Optionally dump full JSON
    if '--json' in sys.argv:
        print(json.dumps([asdict(t) for t in tracks], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()