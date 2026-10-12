"""Download selected official UCI Hydraulic Systems files; no bundled dataset."""
from pathlib import Path
import shutil
import tempfile
import ssl
import hashlib
import json
from datetime import datetime, timezone
import certifi
from urllib.request import Request, urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw"
REQUIRED = {"profile.txt", "PS1.txt", "PS2.txt", "FS1.txt", "TS1.txt", "VS1.txt"}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if all((DEST / name).exists() for name in REQUIRED):
        print("Required UCI files already downloaded.")
        return
    print("Downloading ~73 MB official UCI Hydraulic Systems archive...")
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "hydraulic.zip"
        request = Request(URL, headers={"User-Agent": "FactoryMindAI/0.1 (educational hackathon project)"})
        with urlopen(request, timeout=120, context=ssl.create_default_context(cafile=certifi.where())) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with ZipFile(archive) as zipped:
            found = set()
            for item in zipped.infolist():
                base = Path(item.filename).name
                if base in REQUIRED and not item.is_dir():
                    with zipped.open(item) as src, (DEST / base).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    found.add(base)
        missing = REQUIRED - found
        if missing:
            raise RuntimeError(f"Dataset archive missing: {sorted(missing)}")
        manifest = {"source_url": URL, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "archive_sha256": hashlib.file_digest(archive.open("rb"), "sha256").hexdigest(),
                    "files": {name: hashlib.sha256((DEST / name).read_bytes()).hexdigest() for name in sorted(REQUIRED)}}
        (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Extracted {len(found)} verified filenames to {DEST}")

if __name__ == "__main__":
    main()
