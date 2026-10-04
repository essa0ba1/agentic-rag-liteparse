# manifest.py
import hashlib
import json
from pathlib import Path

MANIFEST_PATH = Path("data/ingest_manifest.json")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk_size):
            h.update(data)
    return h.hexdigest()


def is_unchanged(path: Path, cached: dict | None) -> bool:
    """Cheap check first (mtime+size); only hash the file if those look changed."""
    if cached is None:
        return False
    fp = file_fingerprint(path)
    if fp["mtime"] == cached.get("mtime") and fp["size"] == cached.get("size"):
        return True
    # mtime/size changed (or file was touched) — confirm with a real hash
    return file_hash(path) == cached.get("hash")