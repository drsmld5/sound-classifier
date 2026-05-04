# api/routes/classify.py
# Classification endpoints.
# Runs the full BPM / key / genre / vibe pipeline on one or more tracks.

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import asdict
from typing import Optional
import json

from pipeline.scanner import scan_folder, scan_files, AudioFile
from pipeline.aggregator import classify_track, classify_batch

router = APIRouter()


class ClassifyFolderRequest(BaseModel):
    folder_path: str
    duration: float = 60.0
    use_embedded_bpm: bool = True
    use_embedded_key: bool = True
    use_embedded_genre: bool = False


class ClassifyFilesRequest(BaseModel):
    file_paths: list[str]
    duration: float = 60.0
    use_embedded_bpm: bool = True
    use_embedded_key: bool = True
    use_embedded_genre: bool = False


class ClassifyTrackRequest(BaseModel):
    file_path: str
    duration: float = 60.0
    use_embedded_bpm: bool = True
    use_embedded_key: bool = True
    use_embedded_genre: bool = False


@router.post("/track")
def classify_single_track(request: ClassifyTrackRequest):
    """
    Classify a single audio file.
    Runs the full pipeline: BPM → key → genre → vibe.
    Blocking — returns when classification is complete.
    """
    tracks = scan_files([request.file_path])
    if not tracks:
        raise HTTPException(
            status_code=404,
            detail=f"File not found or unsupported format: {request.file_path}"
        )

    try:
        result = classify_track(
            tracks[0],
            duration=request.duration,
            use_embedded_bpm=request.use_embedded_bpm,
            use_embedded_key=request.use_embedded_key,
            use_embedded_genre=request.use_embedded_genre,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return asdict(result)


@router.post("/folder")
def classify_folder(request: ClassifyFolderRequest):
    """
    Classify all audio files in a folder.
    Blocking — returns when all tracks are complete.
    For large folders, prefer /folder/stream.
    """
    try:
        tracks = scan_folder(request.folder_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not tracks:
        return {"count": 0, "results": []}

    try:
        results = classify_batch(tracks, duration=request.duration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "count": len(results),
        "complete": sum(1 for r in results if r.is_complete),
        "errors": sum(1 for r in results if r.has_errors),
        "results": [asdict(r) for r in results],
    }


@router.post("/folder/stream")
def classify_folder_stream(request: ClassifyFolderRequest):
    """
    Classify all audio files in a folder with Server-Sent Events streaming.
    Each track result is emitted as it completes — the UI can update
    progressively rather than waiting for the full batch.

    Event format:
      data: {"type": "progress", "current": 1, "total": 10, "result": {...}}
      data: {"type": "complete", "total": 10, "errors": 0}
      data: {"type": "error", "detail": "..."}
    """
    try:
        tracks = scan_folder(request.folder_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not tracks:
        raise HTTPException(status_code=404, detail="No audio files found in folder")

    def event_stream():
        results = []
        total = len(tracks)
        error_count = 0

        # Emit total count first so UI can set up a progress bar immediately
        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        for i, track in enumerate(tracks):
            try:
                result = classify_track(
                    track,
                    duration=request.duration,
                    use_embedded_bpm=request.use_embedded_bpm,
                    use_embedded_key=request.use_embedded_key,
                    use_embedded_genre=request.use_embedded_genre,
                )
                results.append(result)
                if result.has_errors:
                    error_count += 1

                payload = {
                    "type": "progress",
                    "current": i + 1,
                    "total": total,
                    "result": asdict(result),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            except Exception as e:
                error_count += 1
                payload = {
                    "type": "track_error",
                    "current": i + 1,
                    "total": total,
                    "filename": track.filename,
                    "detail": str(e),
                }
                yield f"data: {json.dumps(payload)}\n\n"

        # Final completion event
        yield f"data: {json.dumps({'type': 'complete', 'total': total, 'errors': error_count})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering if proxied
        }
    )


@router.post("/files")
def classify_files(request: ClassifyFilesRequest):
    """
    Classify a specific list of audio files.
    """
    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided")

    try:
        tracks = scan_files(request.file_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not tracks:
        return {"count": 0, "results": []}

    try:
        results = classify_batch(tracks, duration=request.duration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "count": len(results),
        "complete": sum(1 for r in results if r.is_complete),
        "errors": sum(1 for r in results if r.has_errors),
        "results": [asdict(r) for r in results],
    }