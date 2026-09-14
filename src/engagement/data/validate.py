"""Validate ImageFolder datasets without importing PyTorch."""

from pathlib import Path
from typing import Dict, Iterable, Sequence, Union

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_imagefolder(root: Union[str, Path], expected_classes: Sequence[str]) -> Dict[str, object]:
    root = Path(root)
    report: Dict[str, object] = {"root": str(root), "counts": {}, "corrupt": [], "missing": []}
    hashes = {}
    duplicates = []
    import hashlib

    for split in ("train", "val", "test"):
        for label in expected_classes:
            directory = root / split / label
            key = "%s/%s" % (split, label)
            if not directory.is_dir():
                report["missing"].append(key)  # type: ignore[union-attr]
                continue
            files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            report["counts"][key] = len(files)  # type: ignore[index]
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest in hashes:
                        duplicates.append((str(hashes[digest]), str(path)))
                    else:
                        hashes[digest] = path
                except Exception as exc:
                    report["corrupt"].append({"path": str(path), "error": str(exc)})  # type: ignore[union-attr]
    report["duplicates"] = duplicates
    report["valid"] = not report["missing"] and not report["corrupt"] and not duplicates
    return report
