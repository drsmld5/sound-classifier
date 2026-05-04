# pipeline/scanner.py

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import mutagen
from mutagen.id3 import ID3NoHeaderError


AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aiff', '.aif', '.m4a', '.ogg', '.opus'}


@dataclass
class AudioFile:
    id: str                          # relative path used as stable identifier
    name: str                        # filename without extension
    filename: str                    # full filename with extension
    path: str                        # absolute path
    relative_path: str               # path relative to scanned root
    extension: str
    size_mb: float
    duration_seconds: Optional[float] = None

    # Embedded metadata (from ID3/FLAC/etc tags)
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    year: Optional[str] = None
    embedded_genre: Optional[str] = None
    embedded_bpm: Optional[float] = None
    embedded_key: Optional[str] = None

    # AI classification results (populated later by pipeline)
    detected_genre: Optional[str] = None
    detected_bpm: Optional[float] = None
    detected_key: Optional[str] = None
    detected_vibe: Optional[str] = None
    confidence: dict = field(default_factory=dict)

import unicodedata

def _clean_tag(value: str) -> str:
    """
    Normalize Unicode tag strings.
    - NFC normalisation fixes composed vs decomposed characters
    - Replace fullwidth punctuation with standard ASCII equivalents
    - Strip leading/trailing whitespace
    """
    value = unicodedata.normalize('NFC', value)
    # Fullwidth to ASCII punctuation
    replacements = {
        '\uff02': '"',   # fullwidth quotation mark
        '\uff01': '!',   # fullwidth exclamation
        '\uff08': '(',   # fullwidth left paren
        '\uff09': ')',   # fullwidth right paren
        '\u2019': "'",   # right single quotation mark
        '\u2018': "'",   # left single quotation mark
        '\u201c': '"',   # left double quotation mark
        '\u201d': '"',   # right double quotation mark
        '\u2013': '-',   # en dash
        '\u2014': '-',   # em dash
    }
    for char, replacement in replacements.items():
        value = value.replace(char, replacement)
    return value.strip()

def _read_metadata(path: Path) -> dict:
    """Read embedded tags from audio file using mutagen."""
    meta = {}
    try:
        audio = mutagen.File(path, easy=True)
        if audio is None:
            return meta

        # Duration
        if hasattr(audio, 'info') and hasattr(audio.info, 'length'):
            meta['duration_seconds'] = round(audio.info.length, 2)

        # Common tags (easy=True normalizes tag names across formats)
        tag_map = {
            'title': 'title',
            'artist': 'artist',
            'album': 'album',
            'date': 'year',
            'genre': 'embedded_genre',
            'bpm': 'embedded_bpm',
            'initialkey': 'embedded_key',
        }

        for tag, field_name in tag_map.items():
            value = audio.get(tag)
            if value:
                raw = value[0] if isinstance(value, list) else value
                if field_name == 'embedded_bpm':
                    try:
                        meta[field_name] = float(raw)
                    except (ValueError, TypeError):
                        pass
                else:
                    meta[field_name] = _clean_tag(str(raw))

    except (ID3NoHeaderError, Exception):
        # Unreadable tags — file is still valid, just no metadata
        pass

    return meta


def scan_folder(folder_path: str) -> list[AudioFile]:
    """
    Recursively scan a folder for audio files.
    Returns a list of AudioFile dataclasses with metadata populated.
    """
    root = Path(folder_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder_path}")

    tracks = []

    for file_path in sorted(root.rglob('*')):
        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if file_path.name.startswith('.'):
            continue  # skip macOS hidden files like ._track.mp3

        relative = file_path.relative_to(root)
        size_mb = round(file_path.stat().st_size / 1_000_000, 2)
        metadata = _read_metadata(file_path)

        track = AudioFile(
            id=str(relative),
            name=file_path.stem,
            filename=file_path.name,
            path=str(file_path),
            relative_path=str(relative),
            extension=file_path.suffix.lower(),
            size_mb=size_mb,
            **metadata
        )

        tracks.append(track)

    return tracks


def scan_files(file_paths: list[str]) -> list[AudioFile]:
    """
    Scan a specific list of file paths instead of a whole folder.
    Used when the user selects individual files rather than a folder.
    """
    tracks = []
    for path_str in file_paths:
        file_path = Path(path_str).resolve()
        if not file_path.exists():
            continue
        if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        size_mb = round(file_path.stat().st_size / 1_000_000, 2)
        metadata = _read_metadata(file_path)

        track = AudioFile(
            id=str(file_path),
            name=file_path.stem,
            filename=file_path.name,
            path=str(file_path),
            relative_path=file_path.name,
            extension=file_path.suffix.lower(),
            size_mb=size_mb,
            **metadata
        )
        tracks.append(track)

    return tracks