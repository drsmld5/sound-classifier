# api/routes/scan.py
# Folder and file scanning endpoints.
# Returns track metadata without running the classification pipeline.

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dataclasses import asdict

from pipeline.scanner import scan_folder, scan_files

router = APIRouter()


class ScanFolderRequest(BaseModel):
    folder_path: str


class ScanFilesRequest(BaseModel):
    file_paths: list[str]


@router.post("/folder")
def scan_folder_endpoint(request: ScanFolderRequest):
    """
    Scan a folder recursively for audio files.
    Returns metadata from embedded tags only — no audio analysis.
    Fast — suitable for immediate UI feedback after folder selection.
    """
    try:
        tracks = scan_folder(request.folder_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "folder": request.folder_path,
        "count": len(tracks),
        "tracks": [asdict(t) for t in tracks],
    }


@router.post("/files")
def scan_files_endpoint(request: ScanFilesRequest):
    """
    Scan a specific list of file paths.
    Used when the user selects individual files rather than a folder.
    """
    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No file paths provided")

    try:
        tracks = scan_files(request.file_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "count": len(tracks),
        "tracks": [asdict(t) for t in tracks],
    }