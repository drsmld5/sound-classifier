# sound-classifier/main.py
# Phase 1: CLI validation — run this directly to test the pipeline
# before any Tauri or FastAPI work

import sys
from pathlib import Path

def scan_folder(folder_path: str) -> list[dict]:
    audio_extensions = {'.mp3', '.wav', '.flac', '.aiff', '.m4a'}
    folder = Path(folder_path)
    tracks = []

    for file in folder.rglob('*'):
        if file.suffix.lower() in audio_extensions:
            tracks.append({
                'name': file.name,
                'path': str(file),
                'size_mb': round(file.stat().st_size / 1_000_000, 2),
                'extension': file.suffix.lower()
            })

    return tracks


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    tracks = scan_folder(folder)
    for t in tracks:
        print(t)
    print(f'\nFound {len(tracks)} audio files')