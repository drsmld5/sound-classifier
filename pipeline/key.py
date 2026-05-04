# pipeline/key.py
# Key and tonality detection using librosa (Krumhansl-Schmuckler algorithm)
# essentia gives better results and will be added in Phase 2,
# but librosa covers the MVP without the TF dependency complexity.

import librosa
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KeyResult:
    key: str                  # e.g. "A"
    mode: str                 # "major" or "minor"
    label: str                # combined e.g. "A minor"
    camelot: str              # Camelot wheel notation e.g. "8A"
    confidence: float         # 0.0 - 1.0
    source: str               # "detected" or "embedded"


# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
               'F#', 'G', 'G#', 'A', 'A#', 'B']

# Camelot wheel mapping: (root_index, mode) -> camelot
_CAMELOT = {
    ('C', 'major'): '8B',  ('C', 'minor'): '5A',
    ('C#', 'major'): '3B', ('C#', 'minor'): '12A',
    ('D', 'major'): '10B', ('D', 'minor'): '7A',
    ('D#', 'major'): '5B', ('D#', 'minor'): '2A',
    ('E', 'major'): '12B', ('E', 'minor'): '9A',
    ('F', 'major'): '7B',  ('F', 'minor'): '4A',
    ('F#', 'major'): '2B', ('F#', 'minor'): '11A',
    ('G', 'major'): '9B',  ('G', 'minor'): '6A',
    ('G#', 'major'): '4B', ('G#', 'minor'): '1A',
    ('A', 'major'): '11B', ('A', 'minor'): '8A',
    ('A#', 'major'): '6B', ('A#', 'minor'): '3A',
    ('B', 'major'): '1B',  ('B', 'minor'): '10A',
}


def _correlate_key(chroma_mean: np.ndarray) -> tuple[int, str, float]:
    """
    Match chroma vector against major and minor key profiles
    for all 12 roots using Krumhansl-Schmuckler correlation.
    Returns (root_index, mode, confidence).
    """
    best_score = -np.inf
    best_root = 0
    best_mode = 'major'

    for root in range(12):
        # Rotate profiles to match root
        major_rotated = np.roll(_MAJOR_PROFILE, root)
        minor_rotated = np.roll(_MINOR_PROFILE, root)

        major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]

        if major_corr > best_score:
            best_score = major_corr
            best_root = root
            best_mode = 'major'

        if minor_corr > best_score:
            best_score = minor_corr
            best_root = root
            best_mode = 'minor'

    # normalize correlation score to 0-1 confidence
    confidence = round(float((best_score + 1) / 2), 3)
    return best_root, best_mode, confidence


def detect_key(
    file_path: str,
    embedded_key: Optional[str] = None,
    use_embedded_if_available: bool = True,
    duration: float = 60.0
) -> KeyResult:
    """
    Detect musical key and mode of an audio file.

    Args:
        file_path:                  absolute path to the audio file
        embedded_key:               key from embedded tags (mutagen), if any
        use_embedded_if_available:  trust embedded tag if present
        duration:                   seconds to analyze

    Returns:
        KeyResult dataclass
    """

    # Use embedded key if available and looks valid
    if use_embedded_if_available and embedded_key:
        cleaned = embedded_key.strip()
        # normalize common formats: "Am", "A minor", "Amin" -> key + mode
        if len(cleaned) >= 1:
            return _parse_embedded_key(cleaned)

    # Load audio
    try:
        y, sr = librosa.load(
            file_path,
            mono=True,
            duration=duration,
            res_type='soxr_hq'
        )
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {file_path}\n{e}")

    # Compute chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    root_idx, mode, confidence = _correlate_key(chroma_mean)
    note = _NOTE_NAMES[root_idx]
    camelot = _CAMELOT.get((note, mode), '?')

    return KeyResult(
        key=note,
        mode=mode,
        label=f"{note} {mode}",
        camelot=camelot,
        confidence=confidence,
        source='detected'
    )


def _parse_embedded_key(raw: str) -> KeyResult:
    """Parse an embedded key string into a KeyResult."""
    raw = raw.strip()
    mode = 'minor' if raw.lower().endswith(('m', 'min', 'minor')) else 'major'

    # Extract root note
    root = raw[0].upper()
    if len(raw) > 1 and raw[1] in ('#', 'b'):
        root += '#' if raw[1] == '#' else 'b'

    # normalize flats to sharps
    flat_to_sharp = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
    root = flat_to_sharp.get(root, root)

    camelot = _CAMELOT.get((root, mode), '?')
    return KeyResult(
        key=root,
        mode=mode,
        label=f"{root} {mode}",
        camelot=camelot,
        confidence=0.95,
        source='embedded'
    )