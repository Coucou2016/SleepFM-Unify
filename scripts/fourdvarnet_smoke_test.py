"""End-to-end smoke test: data -> train -> forward pass."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(root)}
    steps = [
        [py, str(root / "scripts" / "fourdvarnet_generate_data.py"), "--demo"],
        [py, str(root / "scripts" / "fourdvarnet_train.py"), "--demo"],
    ]
    for cmd in steps:
        print("RUN:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=root, env=env)

    import torch

    from fourdvarnet.models.fourdvarnet import FourDVarNet

    m = FourDVarNet(n_timesteps=7)
    b, t3, h, w = 2, 21, 16, 16
    y = torch.randn(b, t3, h, w)
    mask = torch.zeros_like(y)
    mask[:, :7] = 1.0
    z = torch.randn(b, 1, h, w)
    out = m(y, mask, z)
    assert out["u_last"].shape == (b, 1, h, w)
    print("Smoke test OK:", {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})


if __name__ == "__main__":
    main()
