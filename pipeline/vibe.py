# pipeline/vibe.py
# Vibe and mood tagging from audio features.
#
# Produces human-readable descriptors a non-technical user understands:
# "dark / driving", "warm / late-night", "energetic / peak-time" etc.
#
# Architecture mirrors genre.py:
#   Stage 1 (MVP): Feature-based heuristic
#   Stage 2 (later): essentia mood models (happy, aggressive, relaxed, sad)
#   Same VibeResult return type throughout.

import librosa
import numpy as np
from dataclasses import dataclass, field


# Energy level descriptors
ENERGY_LEVELS = ['low', 'medium', 'high', 'peak']

# Mood descriptors
MOODS = ['dark', 'warm', 'euphoric', 'melancholic', 'aggressive', 'hypnotic', 'dreamy']

# Atmosphere descriptors
ATMOSPHERES = ['late-night', 'driving', 'chilled', 'peak-time', 'deep', 'trippy', 'uplifting']


@dataclass
class VibeResult:
    energy: str                             # low / medium / high / peak
    mood: str                               # dark / warm / euphoric etc.
    atmosphere: str                         # late-night / driving / peak-time etc.
    label: str                              # combined human label e.g. "dark / driving"
    tags: list[str] = field(default_factory=list)   # all applicable tags
    confidence: float = 0.0
    source: str = 'heuristic'


def _compute_energy(rms: float, bpm: float, onset_rate: float) -> tuple[str, float]:
    """
    Map RMS energy, tempo and onset density to an energy level.
    Returns (label, normalized_score 0-1).
    """
    # Normalize each component to 0-1
    rms_score = min(rms / 0.15, 1.0)           # 0.15 RMS ≈ very loud
    bpm_score = min((bpm - 60) / 130.0, 1.0)   # 60 BPM = 0, 190 BPM = 1
    onset_score = min(onset_rate / 6.0, 1.0)   # 6 onsets/s ≈ very dense

    combined = (rms_score * 0.4) + (bpm_score * 0.4) + (onset_score * 0.2)

    if combined < 0.25:
        return 'low', combined
    if combined < 0.50:
        return 'medium', combined
    if combined < 0.75:
        return 'high', combined
    return 'peak', combined


def _score_moods(
    bpm: float,
    spectral_centroid: float,
    zcr: float,
    contrast: float,
    mfcc_mean: np.ndarray,
    rms: float
) -> dict[str, float]:
    """
    Score mood descriptors from spectral and timbral features.
    """
    scores = {m: 0.0 for m in MOODS}

    # Dark — low brightness, low BPM relative to genre, high contrast
    if spectral_centroid < 1800:
        scores['dark'] += 2.0
    if contrast > 25:
        scores['dark'] += 1.5
    if bpm >= 128 and spectral_centroid < 2000:
        scores['dark'] += 1.0      # dark techno profile

    # Warm — mid brightness, moderate energy, smooth
    if 1500 <= spectral_centroid <= 2800:
        scores['warm'] += 2.0
    if zcr < 0.07:
        scores['warm'] += 1.5      # smooth, tonal
    if 110 <= bpm <= 130:
        scores['warm'] += 1.0

    # Euphoric — high brightness, high energy, high BPM
    if spectral_centroid > 2800:
        scores['euphoric'] += 2.0
    if bpm >= 130:
        scores['euphoric'] += 1.5
    if rms > 0.09:
        scores['euphoric'] += 1.0

    # Melancholic — low energy, low BPM, tonal (low ZCR)
    if bpm < 100:
        scores['melancholic'] += 2.0
    if zcr < 0.05:
        scores['melancholic'] += 1.5
    if rms < 0.05:
        scores['melancholic'] += 1.5
    if spectral_centroid < 2000:
        scores['melancholic'] += 1.0

    # Aggressive — high ZCR, high contrast, high BPM
    if zcr > 0.10:
        scores['aggressive'] += 2.0
    if contrast > 30:
        scores['aggressive'] += 1.5
    if bpm > 145:
        scores['aggressive'] += 1.5
    if rms > 0.10:
        scores['aggressive'] += 1.0

    # Hypnotic — repetitive structure (flat MFCCs), mid BPM, moderate brightness
    mfcc_variance = float(np.var(mfcc_mean))
    if mfcc_variance < 200:
        scores['hypnotic'] += 2.5   # low MFCC variance = repetitive texture
    if 125 <= bpm <= 145:
        scores['hypnotic'] += 1.5
    if spectral_centroid < 2500:
        scores['hypnotic'] += 1.0

    # Dreamy — low ZCR, low energy, bright but gentle
    if zcr < 0.04:
        scores['dreamy'] += 2.0
    if rms < 0.06:
        scores['dreamy'] += 1.5
    if spectral_centroid > 2000:
        scores['dreamy'] += 1.0
    if bpm < 110:
        scores['dreamy'] += 1.0

    # Normalize
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {m: round(v / max_score, 3) for m, v in scores.items()}

    return scores


def _score_atmospheres(
    bpm: float,
    energy_label: str,
    mood: str,
    onset_rate: float,
    spectral_centroid: float
) -> dict[str, float]:
    """
    Score atmosphere descriptors from energy + mood combination.
    """
    scores = {a: 0.0 for a in ATMOSPHERES}

    # Late-night — moderate energy, dark or warm mood, mid BPM
    if mood in ('dark', 'warm', 'hypnotic'):
        scores['late-night'] += 2.0
    if 110 <= bpm <= 135:
        scores['late-night'] += 1.5
    if energy_label in ('medium', 'high'):
        scores['late-night'] += 1.0

    # Driving — consistent high energy, mid-high BPM
    if energy_label in ('high', 'peak'):
        scores['driving'] += 2.0
    if 125 <= bpm <= 155:
        scores['driving'] += 2.0
    if onset_rate > 3.0:
        scores['driving'] += 1.0

    # Chilled — low energy, slow BPM
    if energy_label == 'low':
        scores['chilled'] += 3.0
    if bpm < 105:
        scores['chilled'] += 2.0
    if mood in ('dreamy', 'melancholic', 'warm'):
        scores['chilled'] += 1.0

    # Peak-time — maximum energy, fast BPM
    if energy_label == 'peak':
        scores['peak-time'] += 3.0
    if bpm >= 135:
        scores['peak-time'] += 2.0
    if mood in ('euphoric', 'aggressive'):
        scores['peak-time'] += 1.5

    # Deep — low brightness, repetitive, moderate BPM
    if spectral_centroid < 1800:
        scores['deep'] += 2.0
    if mood == 'hypnotic':
        scores['deep'] += 2.0
    if 120 <= bpm <= 135:
        scores['deep'] += 1.0

    # Trippy — dreamy + moderate energy
    if mood in ('dreamy', 'hypnotic'):
        scores['trippy'] += 2.0
    if energy_label == 'medium':
        scores['trippy'] += 1.0
    if 90 <= bpm <= 125:
        scores['trippy'] += 1.0

    # Uplifting — euphoric mood, high brightness, peak energy
    if mood == 'euphoric':
        scores['uplifting'] += 3.0
    if spectral_centroid > 2800:
        scores['uplifting'] += 1.5
    if energy_label in ('high', 'peak'):
        scores['uplifting'] += 1.0

    # Normalize
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {a: round(v / max_score, 3) for a, v in scores.items()}

    return scores


def detect_vibe(
    file_path: str,
    bpm: float = None,
    duration: float = 60.0
) -> VibeResult:
    """
    Detect vibe, energy, mood and atmosphere tags for an audio file.

    Args:
        file_path:  absolute path to the audio file
        bpm:        pre-detected BPM (avoids recomputing tempo)
        duration:   seconds of audio to analyze

    Returns:
        VibeResult dataclass
    """
    try:
        y, sr = librosa.load(
            file_path,
            mono=True,
            duration=duration,
            res_type='soxr_hq'
        )
    except Exception as e:
        raise RuntimeError(f"Could not load audio: {file_path}\n{e}")

    # BPM — reuse if already detected
    if bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])

    # Features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = len(onsets) / (len(y) / sr)

    # Energy
    energy_label, energy_score = _compute_energy(rms, bpm, onset_rate)

    # Mood
    mood_scores = _score_moods(
        bpm, spectral_centroid, zcr, spectral_contrast, mfcc_mean, rms
    )
    top_mood = max(mood_scores, key=mood_scores.get)

    # Atmosphere
    atm_scores = _score_atmospheres(
        bpm, energy_label, top_mood, onset_rate, spectral_centroid
    )
    top_atmosphere = max(atm_scores, key=atm_scores.get)

    # Build tag list — include all descriptors scoring above 0.5
    tags = [energy_label]
    tags += [m for m, s in mood_scores.items() if s >= 0.5]
    tags += [a for a, s in atm_scores.items() if s >= 0.5]
    # Deduplicate while preserving order
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    label = f"{top_mood} / {top_atmosphere}"

    confidence = round(
        (mood_scores[top_mood] + atm_scores[top_atmosphere]) / 2 * 0.85,
        3
    )

    return VibeResult(
        energy=energy_label,
        mood=top_mood,
        atmosphere=top_atmosphere,
        label=label,
        tags=tags,
        confidence=confidence,
        source='heuristic'
    )