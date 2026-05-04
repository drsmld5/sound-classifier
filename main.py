# main.py
import sys
import json
from dataclasses import asdict
from pipeline.scanner import scan_folder
from pipeline.bpm import detect_bpm
from pipeline.key import detect_key
from pipeline.genre import detect_genre


def main():
    print(">>> MAIN.PY LOADED - VERSION CHECK 001")
    if len(sys.argv) < 2:
        print("Usage: python main.py <folder_path> [--json] [--analyze]")
        sys.exit(1)

    folder = sys.argv[1]
    do_analyze = '--analyze' in sys.argv
    do_json = '--json' in sys.argv

    print(f"\nScanning: {folder}\n")
    print(f">>> sys.argv = {sys.argv}")
    print(f">>> do_analyze = {do_analyze}")

    try:
        tracks = scan_folder(folder)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not tracks:
        print("No audio files found.")
        sys.exit(0)

    for track in tracks:
        print(f"  {track.relative_path}  [{track.size_mb} MB]")

        if do_analyze:
            print(f"\n  Analysing: {track.filename}")

            bpm_result = detect_bpm(track.path, embedded_bpm=track.embedded_bpm)
            print(f"  BPM result: {bpm_result}")
            track.detected_bpm = bpm_result.corrected_bpm
            track.confidence['bpm'] = bpm_result.confidence

            key_result = detect_key(track.path, embedded_key=track.embedded_key)
            print(f"  Key result: {key_result}")
            track.detected_key = key_result.label
            track.confidence['key'] = key_result.confidence

            # Genre
            try:
                genre_result = detect_genre(
                    track.path,
                    bpm=track.detected_bpm,
                    embedded_genre=track.embedded_genre
                )
                track.detected_genre = genre_result.genre
                track.confidence['genre'] = genre_result.confidence
                print(f"    Genre    : {genre_result.genre} "
                      f"(source: {genre_result.source}, "
                      f"confidence: {genre_result.confidence})")
            except RuntimeError as e:
                print(f"    Genre    : ERROR — {e}")

            # Verify assignment happened
            print(f"    → track.detected_bpm = {track.detected_bpm}")
            print(f"    → track.detected_key = {track.detected_key}")

        elif track.embedded_bpm or track.embedded_key:
            print(f"    BPM tag  : {track.embedded_bpm or '—'}")
            print(f"    Key tag  : {track.embedded_key or '—'}")

        print()

    print(f"Found {len(tracks)} audio file(s)")

    if do_json:
        print(json.dumps([asdict(t) for t in tracks], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()