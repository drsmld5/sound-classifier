# pipeline/genre.py
# Genre classification using audio features extracted with librosa.
#
# Architecture:
#   Stage 1 (MVP): Feature-based heuristic classifier
#     - Extracts spectral, rhythmic and timbral features
#     - Maps feature clusters to genre labels via rule-based scoring
#     - No model weights needed, works offline immediately
#
#   Stage 2 (later): Drop-in replacement using essentia pretrained models
#     - Same GenreResult return type
#     - Just swap _classify_features() with _classify_model()

import librosa
import numpy as np
from dataclasses import dataclass, field


# Supported genre labels
GENRES = [
    'Techno',
    'House',
    'Trance',
    'Drum and Bass',
    'Ambient',
    'Hip-Hop',
    'Breaks',
    'Downtempo',
    'Indie Dance',
    'Unknown',
]


@dataclass
class GenreResult:
    genre: str                          # top predicted genre label
    confidence: float                   # 0.0 - 1.0
    scores: dict = field(default_factory=dict)   # score per genre
    source: str = 'heuristic'          # 'heuristic' or 'model'
    embedded_genre: str = None         # original tag if present


@dataclass
class AudioFeatures:
    """Intermediate feature set used for classification."""
    bpm: float
    spectral_centroid_mean: float      # brightness
    spectral_rolloff_mean: float       # high frequency energy
    spectral_contrast_mean: float      # difference between peaks/valleys
    zero_crossing_rate_mean: float     # noisiness / transient density
    mfcc_mean: np.ndarray              # timbre fingerprint (13 coefficients)
    rms_mean: float                    # overall energy level
    onset_rate: float                  # onsets per second (rhythmic density)


def _extract_features(file_path: str, duration: float = 60.0) -> AudioFeatures:
    """
    Extract timbral and spectral features from audio.
    These form the basis of the heuristic genre classifier.
    """
    y, sr = librosa.load(
        file_path,
        mono=True,
        duration=duration,
        res_type='soxr_hq'
    )

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    # MFCCs — timbre representation
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # RMS energy
    rms = librosa.feature.rms(y=y)

    # Onset rate — rhythmic density (onsets per second)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = len(onsets) / (len(y) / sr)

    return AudioFeatures(
        bpm=bpm,
        spectral_centroid_mean=float(np.mean(spectral_centroid)),
        spectral_rolloff_mean=float(np.mean(spectral_rolloff)),
        spectral_contrast_mean=float(np.mean(spectral_contrast)),
        zero_crossing_rate_mean=float(np.mean(zcr)),
        mfcc_mean=np.mean(mfcc, axis=1),
        rms_mean=float(np.mean(rms)),
        onset_rate=onset_rate,
    )


def _score_genres(f: AudioFeatures) -> dict[str, float]:
    """
    Rule-based genre scoring from audio features.
    Each genre accumulates points based on how well the features match
    its typical characteristics. Returns normalized scores (0.0 - 1.0).

    Feature reference ranges (approximate, based on electronic music):
      BPM:
        Ambient/Downtempo:  60-100
        Hip-Hop:            80-110
        House:             118-130
        Trance:            128-145
        Breaks:            130-160
        Techno:            130-160
        Drum and Bass:     160-180

      Spectral centroid (brightness Hz):
        Low  < 1500  → warm, bass-heavy
        Mid  1500-3000 → balanced
        High > 3000  → bright, high energy

      ZCR (zero crossing rate):
        Low  < 0.05  → tonal, smooth
        High > 0.10  → noisy, percussive
    """

    scores = {g: 0.0 for g in GENRES}
    bpm = f.bpm
    sc = f.spectral_centroid_mean
    zcr = f.zero_crossing_rate_mean
    rms = f.rms_mean
    onset = f.onset_rate
    contrast = f.spectral_contrast_mean

    # ── Techno ──────────────────────────────────────────────────
    if 128 <= bpm <= 160:
        scores['Techno'] += 3.0
    if sc < 2500:
        scores['Techno'] += 1.0         # darker, less bright
    if zcr > 0.07:
        scores['Techno'] += 1.0         # percussive transients
    if onset > 3.0:
        scores['Techno'] += 1.0         # dense rhythmic events
    if rms > 0.08:
        scores['Techno'] += 0.5         # high energy

    # ── House ────────────────────────────────────────────────────
    if 118 <= bpm <= 132:
        scores['House'] += 3.0
    if 1500 <= sc <= 3000:
        scores['House'] += 1.5          # warm mid-range brightness
    if 0.04 <= zcr <= 0.09:
        scores['House'] += 1.0          # moderate transient density
    if 2.0 <= onset <= 4.0:
        scores['House'] += 1.0

    # ── Trance ───────────────────────────────────────────────────
    if 128 <= bpm <= 145:
        scores['Trance'] += 2.5
    if sc > 2500:
        scores['Trance'] += 2.0         # bright, high synths
    if f.mfcc_mean[1] > 0:
        scores['Trance'] += 0.5         # tonal content
    if rms > 0.07:
        scores['Trance'] += 0.5

    # ── Drum and Bass ────────────────────────────────────────────
    if 160 <= bpm <= 185:
        scores['Drum and Bass'] += 4.0
    if onset > 4.0:
        scores['Drum and Bass'] += 1.5  # very dense rhythmic events
    if sc > 2000:
        scores['Drum and Bass'] += 1.0

    # ── Ambient ──────────────────────────────────────────────────
    if bpm < 100:
        scores['Ambient'] += 2.0
    if sc < 1500:
        scores['Ambient'] += 1.5        # warm, low frequency dominant
    if zcr < 0.04:
        scores['Ambient'] += 2.0        # smooth, few transients
    if onset < 1.5:
        scores['Ambient'] += 2.0        # sparse events
    if rms < 0.05:
        scores['Ambient'] += 1.0        # quiet

    # ── Hip-Hop ──────────────────────────────────────────────────
    if 80 <= bpm <= 115:
        scores['Hip-Hop'] += 3.0
    if sc < 2000:
        scores['Hip-Hop'] += 1.0        # bass-heavy
    if 0.05 <= zcr <= 0.10:
        scores['Hip-Hop'] += 1.0
    if contrast > 20:
        scores['Hip-Hop'] += 1.0        # strong contrast between bass and treble

    # ── Breaks ───────────────────────────────────────────────────
    if 130 <= bpm <= 160:
        scores['Breaks'] += 2.0
    if onset > 3.5:
        scores['Breaks'] += 2.0         # syncopated, dense
    if zcr > 0.08:
        scores['Breaks'] += 1.0

    # ── Downtempo ────────────────────────────────────────────────
    if 80 <= bpm <= 105:
        scores['Downtempo'] += 2.5
    if sc < 2000:
        scores['Downtempo'] += 1.0
    if zcr < 0.06:
        scores['Downtempo'] += 1.0
    if 1.0 <= onset <= 3.0:
        scores['Downtempo'] += 1.0

    # ── Indie Dance ──────────────────────────────────────────────
    if 115 <= bpm <= 128:
        scores['Indie Dance'] += 2.5
    if 1800 <= sc <= 3500:
        scores['Indie Dance'] += 1.5    # bright but not harsh
    if 0.05 <= zcr <= 0.09:
        scores['Indie Dance'] += 1.0

    # Normalize scores to 0.0 - 1.0
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {g: round(v / max_score, 3) for g, v in scores.items()}

    return scores


def detect_genre(
    file_path: str,
    bpm: float = None,
    embedded_genre: str = None,
    use_embedded_if_available: bool = False,   # genres tags are often wrong
    duration: float = 60.0
) -> GenreResult:
    """
    Detect genre of an audio file.

    Note: use_embedded_if_available defaults to False for genre because
    embedded genre tags are frequently inaccurate or overly broad
    (e.g. 'Electronic' for everything). The heuristic is often more
    precise for electronic music subgenres.

    Args:
        file_path:                  absolute path to the audio file
        bpm:                        already-detected BPM (avoids recomputing)
        embedded_genre:             genre from embedded tags, if any
        use_embedded_if_available:  trust embedded genre tag
        duration:                   seconds of audio to analyze

    Returns:
        GenreResult dataclass
    """

    # Optionally trust embedded tag
    if use_embedded_if_available and embedded_genre:
        cleaned = embedded_genre.strip()
        if cleaned:
            return GenreResult(
                genre=cleaned,
                confidence=0.7,     # lower confidence — tags often imprecise
                scores={cleaned: 1.0},
                source='embedded',
                embedded_genre=cleaned
            )

    try:
        features = _extract_features(file_path, duration=duration)
    except Exception as e:
        raise RuntimeError(f"Could not extract features from: {file_path}\n{e}")

    # Override BPM if we already have a better detection from bpm.py
    if bpm is not None:
        features.bpm = bpm

    scores = _score_genres(features)

    # Pick top genre
    top_genre = max(scores, key=scores.get)
    top_score = scores[top_genre]

    # If top score is too low, we are not confident enough
    if top_score < 0.3:
        top_genre = 'Unknown'

    # Scale top score to a meaningful confidence value
    # A normalized score of 1.0 maps to ~0.80 confidence max
    # (heuristics are never 100% certain)
    confidence = round(min(top_score * 0.80, 0.80), 3)

    return GenreResult(
        genre=top_genre,
        confidence=confidence,
        scores=scores,
        source='heuristic',
        embedded_genre=embedded_genre
    )