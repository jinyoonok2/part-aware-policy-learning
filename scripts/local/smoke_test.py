"""Verify package imports, CPU physics, and optionally camera rendering."""
import argparse
import importlib.metadata
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MS_ASSET_DIR", str(ROOT / "data" / "assets"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--drawer", action="store_true")
    parser.add_argument("--render-backend", default="cpu" if sys.platform == "win32" else "gpu")
    args = parser.parse_args()
    for package in ("mani_skill", "sapien", "torch", "torchvision", "numpy"):
        print(f"{package}: {importlib.metadata.version(package)}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        value = (torch.ones(4, device="cuda") * 2).sum().item()
        assert value == 8, "CUDA tensor computation failed"

    task = "OpenCabinetDrawer-v1" if args.drawer else "PickCube-v1"
    env = gym.make(
        task,
        num_envs=1,
        sim_backend="physx_cpu",
        render_backend=args.render_backend,
        obs_mode="rgb+segmentation" if args.render else "state",
        render_mode="rgb_array" if args.render else None,
    )
    try:
        obs, _ = env.reset(seed=0)
        env.action_space.seed(0)
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if np.asarray(terminated).any() or np.asarray(truncated).any():
                break
        if args.render:
            import imageio.v3 as iio
            output = ROOT / "outputs"
            output.mkdir(exist_ok=True)
            for camera, data in obs["sensor_data"].items():
                rgb = data["rgb"].detach().cpu().numpy()
                segmentation = data["segmentation"].detach().cpu().numpy()
                assert rgb.shape[-1] == 3 and rgb.size > 0
                assert segmentation.size > 0
                if rgb.ndim == 4:
                    rgb = rgb[0]
                iio.imwrite(output / f"{task}_{camera}.png", rgb)
                print(f"Camera {camera}: RGB {rgb.shape}; segmentation IDs {np.unique(segmentation)}", flush=True)
            print("Camera images saved under outputs/", flush=True)
        print(f"PASS: {task} reset and CPU-physics stepping", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
