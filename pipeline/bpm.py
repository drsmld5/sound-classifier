# pipeline/bpm.py
# BPM detection using librosa
# Uses beat tracking on the audio signal — works on any file
# regardless of whether embedded BPM tags exist.

import librosa
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BPMResult:
    bpm: float                    # detected tempo, rounded to 1 decimal
    confidence: float             # 0.0 - 1.0
    is_half_time: bool            # true if likely half-time feel detected
    corrected_bpm: float          # bpm after half/double time correction
    source: str                   # "detected" or "embedded" (tag was used)


# Typical BPM ranges per genre — used for half/double time correction
# Most electronic music sits between 85-175 BPM
_EXPECTED_RANGE = (85.0, 175.0)


def _correct_tempo(bpm: float) -> tuple[float, bool]:
    """
    librosa sometimes detects half or double the actual tempo.
    Nudge it into the expected dance music range.
    """
    low, high = _EXPECTED_RANGE
    is_half_time = False

    # Too slow — likely half time, double it
    if bpm < low:
        doubled = bpm * 2
        if low <= doubled <= high:
            return round(doubled, 1), True

    # Too fast — likely double time, halve it
    if bpm > high:
        halved = bpm / 2
        if low <= halved <= high:
            return round(halved, 1), False

    return round(bpm, 1), is_half_time


def _estimate_confidence(onset_env: np.ndarray) -> float:
    """
    Estimate BPM detection confidence from onset strength envelope.
    A strong, regular beat produces high onset peaks → high confidence.
    A weak or irregular signal produces low peaks → low confidence.
    """
    if onset_env is None or len(onset_env) == 0:
        return 0.5

    # Ratio of mean to std — high ratio means regular, predictable beats
    mean = float(np.mean(onset_env))
    std = float(np.std(onset_env))

    if std < 1e-6:
        return 0.5

    ratio = mean / (std + 1e-6)

    # Typical values: ratio ~0.5 (weak) to ~2.5 (strong)
    # Normalize to 0.4 - 0.95 range
    confidence = 0.4 + min(ratio / 2.5, 1.0) * 0.55
    return round(float(confidence), 3)

def detect_bpm(
    file_path: str,
    embedded_bpm: Optional[float] = None,
    use_embedded_if_available: bool = True,
    duration: float = 60.0          # analyze first N seconds — faster than full file
) -> BPMResult:
    """
    Detect BPM of an audio file.

    Strategy:
    1. If a reliable embedded BPM tag exists and use_embedded_if_available
       is True, trust it but still validate it looks reasonable.
    2. Otherwise run librosa beat tracking on the first `duration` seconds.

    Args:
        file_path:                  absolute path to the audio file
        embedded_bpm:               BPM from embedded tags (mutagen), if any
        use_embedded_if_available:  trust embedded tag if it looks valid
        duration:                   seconds of audio to analyze (default 60s)

    Returns:
        BPMResult dataclass
    """

    # Use embedded BPM if it looks valid (between 60 and 220)
    if (
        use_embedded_if_available
        and embedded_bpm is not None
        and 60.0 <= embedded_bpm <= 220.0
    ):
        corrected, is_half = _correct_tempo(embedded_bpm)
        return BPMResult(
            bpm=round(embedded_bpm, 1),
            confidence=0.95,        # high confidence — human-set tag
            is_half_time=is_half,
            corrected_bpm=corrected,
            source='embedded'
        )

    # Load audio — mono, native sample rate, limited duration for speed
    try:
        y, sr = librosa.load(
            file_path,
            mono=True,
            duration=duration,
            res_type='soxr_hq'
        )
    except Exception as e:
        raise RuntimeError(f"Could not load audio file: {file_path}\n{e}")

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')

    # librosa >=0.10 returns tempo as a 1-element array
    raw_bpm = float(np.atleast_1d(tempo)[0])

    # Estimate confidence from beat strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    confidence = _estimate_confidence(onset_env)

    corrected, is_half = _correct_tempo(raw_bpm)

    return BPMResult(
        bpm=round(raw_bpm, 1),
        confidence=confidence,
        is_half_time=is_half,
        corrected_bpm=corrected,
        source='detected'
    )