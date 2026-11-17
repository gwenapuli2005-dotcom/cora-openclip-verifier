"""Download exported ONNX artifacts from a *public* GitHub Release at build time.

No Hugging Face dependency on Render. The export_onnx.py script uses OpenCLIP
only on the developer's laptop; inference on Render uses ONNX Runtime only.
"""

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

base_url = os.environ.get("MODEL_RELEASE_URL", "").strip().rstrip("/")
if not base_url.startswith("https://github.com/") or "/releases/download/" not in base_url:
    raise SystemExit(
        "Set MODEL_RELEASE_URL to your public GitHub Release URL: "
        "https://github.com/OWNER/REPOSITORY/releases/download/onnx-v1"
    )

out = Path(__file__).resolve().parent / "models"
out.mkdir(parents=True, exist_ok=True)

for name in ("openclip_vision_int8.onnx", "text_embeddings.json"):
    url = base_url + "/" + urllib.parse.quote(name)
    dest = out / name
    temp = out / (name + ".part")
    digest = hashlib.sha256()
    max_bytes = (450 * 1024 * 1024 if name.endswith(".onnx") else 2 * 1024 * 1024)
    print(f"Downloading GitHub Release asset {name}...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CORA-OpenCLIP-ONNX/3.0"})
        with urllib.request.urlopen(req, timeout=240) as response, temp.open("wb") as file:
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > max_bytes:
                    raise RuntimeError(f"File too big: {name}")
                digest.update(chunk)
                file.write(chunk)
        if received < (1024 * 1024 if name.endswith(".onnx") else 100):
            raise RuntimeError(f"Unexpectedly small download: {name} ({received} bytes)")
        if name.endswith(".json"):
            with temp.open(encoding="utf-8") as handle:
                parsed = json.load(handle)
            if not all(k in parsed for k in ("categories", "unrelated")):
                raise RuntimeError("Malformed embeddings JSON")
        expected = os.getenv("ONNX_SHA256" if name.endswith(".onnx") else "EMBEDDINGS_SHA256", "").lower().strip()
        if expected and digest.hexdigest() != expected:
            raise RuntimeError(f"SHA-256 checksum mismatch for {name}")
        temp.replace(dest)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Unable to download {name} from {url}. "
            "Confirm the release is published and public, and the asset filename is exact."
        ) from exc
    print(f"Ready: {name} ({received / 1024**2:.1f} MiB; SHA256={digest.hexdigest()})", flush=True)
