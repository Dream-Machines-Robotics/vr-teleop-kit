"""Mujoco model construction for the TRLC-DK1 arm.

Loads the URDF and decorates the resulting `MjSpec` with two named sites
that the IK pipeline needs:

  tool0      — the end-effector target. Re-attached as a site on link6-7
               because the URDF importer collapses the fixed-joint
               gripper_tool0 child into its parent.
  j4_anchor  — the position-task anchor used by the decoupled IK. Placed
               on link3-4 (upstream of joint 4 → wrist-invariant) and
               offset 10 cm past joint 4's pivot along the link3→link4
               direction to keep the anchor off the joint-1 axis at
               folded poses.

The URDF itself is deliberately NOT vendored. Clone the DK1 repository
(https://github.com/robot-learning-co/trlc-dk1) and point at
`urdf/follower/TRLC-DK1-Follower.urdf`, either explicitly or via the
`DK1_URDF` environment variable (see `resolve_urdf_path`).

The IK math itself lives in decoupled_ik.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

# Environment variable consulted when no explicit URDF path is given.
DK1_URDF_ENV = "DK1_URDF"

# URDF gripper_tool0 fixed-joint geometry, transcribed from the URDF.
TOOL0_OFFSET_XYZ = np.array([0.158, 0.0, 0.0])
TOOL0_OFFSET_RPY = np.array([-np.pi / 2, 0.0, -np.pi / 2])

DEFAULT_Q_REST = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])


def rpy_to_wxyz(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def resolve_urdf_path(explicit: str | Path | None = None) -> Path:
    """Resolve the DK1 URDF path: an explicit argument wins, otherwise the
    `DK1_URDF` environment variable. Raises with setup instructions when
    neither is configured or the file is missing."""
    raw = explicit or os.environ.get(DK1_URDF_ENV)
    if not raw:
        raise FileNotFoundError(
            "No DK1 URDF configured. Pass urdf_path=... or set the DK1_URDF "
            "environment variable to .../trlc-dk1/urdf/follower/"
            "TRLC-DK1-Follower.urdf (clone "
            "https://github.com/robot-learning-co/trlc-dk1)."
        )
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"DK1 URDF not found at {path}")
    return path


def build_model_with_tool0_site(
    urdf_path: str | Path | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_file(str(resolve_urdf_path(urdf_path)))
    link67 = spec.body("link6-7")
    if link67 is None:
        raise RuntimeError("link6-7 body not found in URDF spec")
    link67.add_site(
        name="tool0",
        pos=TOOL0_OFFSET_XYZ.tolist(),
        quat=rpy_to_wxyz(TOOL0_OFFSET_RPY).tolist(),
    )
    # Position-task anchor. On link3-4 (upstream of joint 4 → fully
    # wrist-invariant), 10 cm past joint 4's pivot along the link3→link4
    # direction. Joint 4 URDF origin in link3-4 frame: (0.244, 0, 0.060),
    # magnitude 0.251 m. Unit direction (0.972, 0, 0.239); 10 cm offset
    # → (0.0972, 0, 0.0239). Anchor position: (0.3412, 0, 0.0839).
    link34 = spec.body("link3-4")
    if link34 is None:
        raise RuntimeError("link3-4 body not found in URDF spec")
    link34.add_site(
        name="j4_anchor",
        pos=[0.3412, 0.0, 0.0839],
        size=[0.015, 0.0, 0.0],     # 1.5 cm sphere
        rgba=[1.0, 0.5, 0.0, 1.0],  # orange
    )
    model = spec.compile()
    return model, mujoco.MjData(model)
