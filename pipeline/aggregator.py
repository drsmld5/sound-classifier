# pipeline/aggregator.py
# Orchestrates the full classification pipeline for a single track.
#
# Takes an AudioFile from the scanner and runs all four modules:
#   bpm.py → key.py → genre.py → vibe.py
#
# Returns a ClassificationResult — the single normalized output object
# used by the DB layer, the FastAPI server, and the Tauri UI.
#
# Design principles:
#   - Each module is independent — a failure in one does not block others
#   - BPM is detected first and shared with genre + vibe to avoid
#     reloading audio and recomputing tempo three times
#   - All results are stored on the AudioFile dataclass in-place
#     AND returned as a ClassificationResult for API consumers

import time
from dataclasses import dataclass, field
from typing import Optional

from pipeline.scanner import AudioFile
from pipeline.bpm import detect_bpm, BPMResult
from pipeline.key import detect_key, KeyResult
from pipeline.genre import detect_genre, GenreResult
from pipeline.vibe import detect_vibe, VibeResult


@dataclass
class ModuleError:
    module: str
    error: str


@dataclass
class ClassificationResult:
    # Identity
    track_id: str
    filename: str
    path: str
    duration_seconds: Optional[float]

    # Metadata from tags
    title: Optional[str]
    artist: Optional[str]
    album: Optional[str]
    year: Optional[str]

    # Classification outputs
    bpm: Optional[float] = None
    bpm_raw: Optional[float] = None
    bpm_source: str = 'unknown'

    key: Optional[str] = None
    key_camelot: Optional[str] = None
    key_source: str = 'unknown'

    genre: Optional[str] = None
    genre_source: str = 'unknown'
    genre_scores: dict = field(default_factory=dict)

    vibe_label: Optional[str] = None
    vibe_energy: Optional[str] = None
    vibe_mood: Optional[str] = None
    vibe_atmosphere: Optional[str] = None
    vibe_tags: list[str] = field(default_factory=list)

    # Confidence per module
    confidence: dict = field(default_factory=dict)

    # Pipeline metadata
    errors: list[ModuleError] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    pipeline_version: str = '1.0.0'

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def is_complete(self) -> bool:
        """True if all four modules produced a result."""
        return all([
            self.bpm is not None,
            self.key is not None,
            self.genre is not None,
            self.vibe_label is not None,
        ])

    def summary(self) -> str:
        """Single-line human readable summary."""
        parts = [
            f"{self.filename}",
            f"BPM: {self.bpm or '?'}",
            f"Key: {self.key or '?'}",
            f"Genre: {self.genre or '?'}",
            f"Vibe: {self.vibe_label or '?'}",
        ]
        if self.has_errors:
            parts.append(f"[{len(self.errors)} error(s)]")
        return "  |  ".join(parts)


def classify_track(
    track: AudioFile,
    duration: float = 60.0,
    use_embedded_bpm: bool = True,
    use_embedded_key: bool = True,
    use_embedded_genre: bool = False,
) -> ClassificationResult:
    """
    Run the full classification pipeline on a single AudioFile.

    Args:
        track:                  AudioFile from scanner
        duration:               seconds of audio to analyze per module
        use_embedded_bpm:       trust embedded BPM tag if present
        use_embedded_key:       trust embedded key tag if present
        use_embedded_genre:     trust embedded genre tag (default False
                                because genre tags are often too broad)

    Returns:
        ClassificationResult with all available outputs populated
    """
    start = time.perf_counter()

    result = ClassificationResult(
        track_id=track.id,
        filename=track.filename,
        path=track.path,
        duration_seconds=track.duration_seconds,
        title=track.title,
        artist=track.artist,
        album=track.album,
        year=track.year,
    )

    # ── BPM ──────────────────────────────────────────────────────────────
    bpm_value = None
    try:
        bpm_result: BPMResult = detect_bpm(
            track.path,
            embedded_bpm=track.embedded_bpm,
            use_embedded_if_available=use_embedded_bpm,
            duration=duration
        )
        result.bpm = bpm_result.corrected_bpm
        result.bpm_raw = bpm_result.bpm
        result.bpm_source = bpm_result.source
        result.confidence['bpm'] = bpm_result.confidence
        bpm_value = bpm_result.corrected_bpm

        # Write back to AudioFile for downstream modules
        track.detected_bpm = bpm_result.corrected_bpm

    except Exception as e:
        result.errors.append(ModuleError(module='bpm', error=str(e)))

    # ── Key ───────────────────────────────────────────────────────────────
    try:
        key_result: KeyResult = detect_key(
            track.path,
            embedded_key=track.embedded_key,
            use_embedded_if_available=use_embedded_key,
            duration=duration
        )
        result.key = key_result.label
        result.key_camelot = key_result.camelot
        result.key_source = key_result.source
        result.confidence['key'] = key_result.confidence

        track.detected_key = key_result.label

    except Exception as e:
        result.errors.append(ModuleError(module='key', error=str(e)))

    # ── Genre ─────────────────────────────────────────────────────────────
    try:
        genre_result: GenreResult = detect_genre(
            track.path,
            bpm=bpm_value,                          # reuse BPM — no recompute
            embedded_genre=track.embedded_genre,
            use_embedded_if_available=use_embedded_genre,
            duration=duration
        )
        result.genre = genre_result.genre
        result.genre_source = genre_result.source
        result.genre_scores = genre_result.scores
        result.confidence['genre'] = genre_result.confidence

        track.detected_genre = genre_result.genre

    except Exception as e:
        result.errors.append(ModuleError(module='genre', error=str(e)))

    # ── Vibe ──────────────────────────────────────────────────────────────
    try:
        vibe_result: VibeResult = detect_vibe(
            track.path,
            bpm=bpm_value,                          # reuse BPM — no recompute
            duration=duration
        )
        result.vibe_label = vibe_result.label
        result.vibe_energy = vibe_result.energy
        result.vibe_mood = vibe_result.mood
        result.vibe_atmosphere = vibe_result.atmosphere
        result.vibe_tags = vibe_result.tags
        result.confidence['vibe'] = vibe_result.confidence

        track.detected_vibe = vibe_result.label

    except Exception as e:
        result.errors.append(ModuleError(module='vibe', error=str(e)))

    result.processing_time_seconds = round(time.perf_counter() - start, 3)
    return result


def classify_batch(
    tracks: list[AudioFile],
    duration: float = 60.0,
    on_progress=None,
) -> list[ClassificationResult]:
    """
    Run classify_track on a list of AudioFiles.

    Args:
        tracks:       list of AudioFile from scanner.scan_folder()
        duration:     seconds of audio to analyze per track
        on_progress:  optional callback(current: int, total: int, result: ClassificationResult)
                      called after each track completes — used later by FastAPI
                      to stream progress to the UI

    Returns:
        list of ClassificationResult in the same order as input tracks
    """
    results = []
    total = len(tracks)

    for i, track in enumerate(tracks):
        result = classify_track(track, duration=duration)
        results.append(result)

        if on_progress:
            on_progress(current=i + 1, total=total, result=result)

    return results