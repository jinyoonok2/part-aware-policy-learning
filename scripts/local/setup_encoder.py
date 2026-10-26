"""Download a pinned DINOv2 encoder and verify frozen patch features."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import torch

REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = torch.hub.load(
        f"facebookresearch/dinov2:{REVISION}",
        "dinov2_vits14",
        trust_repo=True,
    ).eval().requires_grad_(False).to(device)
    with torch.inference_mode():
        features = encoder.forward_features(torch.zeros(1, 3, 224, 224, device=device))
    patches = features["x_norm_patchtokens"]
    assert tuple(patches.shape) == (1, 256, 384), patches.shape
    assert torch.isfinite(patches).all(), "Non-finite patch features"
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "model": "dinov2_vits14",
        "source_revision": REVISION,
        "checkpoint_sha256": digest,
        "device": device,
        "patch_feature_shape": list(patches.shape),
        "torch_version": torch.__version__,
    }
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "encoder_setup.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("PASS: frozen DINOv2 encoder forward pass", flush=True)


if __name__ == "__main__":
    main()
