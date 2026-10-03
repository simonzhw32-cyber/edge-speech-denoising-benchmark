"""Pinned DNS asset integrity and atomic download; Python standard library only."""

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

MODEL_ID = "Zhaoheng/tfgridnet_dns_ins20_epoch33"
REVISION = "667c73df547006f47513f2b31c1709bdac63f79c"
CHECKPOINT_SHA256 = "613db9fb4dafa7860d9e1390ec8f6bb61bba62cc0a67482bebcfbd0221d865c3"
CHECKPOINT_BYTES = 10332558
CHECKPOINT_URL = (
    f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
    "exp/enh_train_enh_tfgrid_raw/33epoch.pth"
)
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints/tfgridnet_dns_epoch33.pth"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_file(path):
    path = Path(path)
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("DNS checkpoint size mismatch; fetch the pinned asset.")
    digest = file_sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("DNS checkpoint SHA-256 mismatch; fetch the pinned asset.")
    return digest


def fetch_checkpoint(path=DEFAULT_CHECKPOINT):
    """Reuse verified files; never overwrite an existing mismatched checkpoint."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Checkpoint destination must not be a symlink.")
    if path.exists():
        verify_checkpoint_file(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tfgridnet_dns_", delete=False) as handle:
            temporary = Path(handle.name)
            request = urllib.request.Request(CHECKPOINT_URL, headers={"User-Agent": "edge-speech-denoising-benchmark/phase29"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if urllib.parse.urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Checkpoint download must remain HTTPS.")
                downloaded = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    downloaded += len(block)
                    if downloaded > CHECKPOINT_BYTES:
                        raise ValueError("Downloaded checkpoint exceeds expected size.")
                    handle.write(block)
        verify_checkpoint_file(temporary)
        # A destination created during download must also pass verification.
        if path.exists():
            verify_checkpoint_file(path)
        else:
            os.replace(temporary, path)
        return path
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
