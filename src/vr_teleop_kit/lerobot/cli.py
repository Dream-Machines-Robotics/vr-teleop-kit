"""Shared helpers for the VR-teleop entry points (`examples/pure_sim.py`,
`examples/teleop_bi_dk1.py`). Each script defines its own loop and CLI; this module
holds the bits that would otherwise be copy-pasted.
"""

from __future__ import annotations

import argparse
import os
from typing import Any


# IK / mapping knobs that both entry points expose on the CLI. Defaults
# are intentionally None — argparse won't overwrite the dataclass default
# unless the operator passes the flag. This is the single source of truth
# for which IK/mapping knobs (damping, reach limits, the wrist park
# gate, scales, pose filter) get plumbed through, and the dataclass
# `BiQuestTeleoperatorConfig` is the single source of truth for their
# values.
_IK_CLI_FIELDS: tuple[tuple[str, str, str], ...] = (
    # (CLI flag, argparse dest, help text)
    ("--lam",                "lam",                "Position-solve base damping."),
    ("--lam0",               "lam0",               "Adaptive-damping ramp amplitude near singular."),
    ("--w0",                 "w0",                 "Manipulability threshold where the ramp starts."),
    ("--mu",                 "mu",                 "Tikhonov stiffness toward rest pose."),
    ("--lam-rot",            "lam_rot",            "Orientation-solve base damping."),
    ("--lam0-rot",           "lam0_rot",           "Extra wrist damping ramp amplitude near gimbal."),
    ("--w0-rot",             "w0_rot",             "Wrist-manipulability threshold where the gimbal ramp starts."),
    ("--rot-reach",          "rot_reach_limit",    "Max angle (rad) the orientation target may run ahead of the arm; 0 disables."),
    ("--pos-reach",          "pos_reach_limit",    "Max distance (m) the position target may run ahead of the arm; 0 disables."),
    ("--rot-err-hold",       "rot_err_hold",       "Park the wrist above this orientation error (rad); > 3.15 disables (demo use)."),
    ("--scale-translation",  "scale_translation",  "controller→EE translation gain."),
    ("--scale-rotation",     "scale_rotation",     "controller→EE rotation gain."),
    ("--pose-filter-alpha",  "pose_filter_alpha",  "EMA on the controller pose (1.0 = none, lower = smoother + laggier)."),
)


def add_ik_cli_args(ap: argparse.ArgumentParser) -> None:
    """Register the shared IK / mapping CLI flags on `ap`. All default to
    None so they only override the dataclass when the operator passes the
    flag explicitly."""
    for flag, _, help_text in _IK_CLI_FIELDS:
        ap.add_argument(flag, type=float, default=None, help=help_text + " (see BiQuestTeleoperatorConfig for default)")


def ik_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Pluck the IK fields the operator set on the CLI; omit unset (None)
    so the dataclass default applies."""
    return {dest: getattr(args, dest) for _, dest, _ in _IK_CLI_FIELDS
            if getattr(args, dest, None) is not None}


def parse_rest_pose_env(env_var: str, fallback: list[float]) -> list[float]:
    """Parse LEFT_REST_POSE / RIGHT_REST_POSE in the same 7-comma format
    `make rest` consumes (joints 1..6 + gripper). The trailing gripper
    value is dropped because the teleop sources gripper position from
    the Quest trigger, not from rest config. Unset or empty → fallback.
    """
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return list(fallback)
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != 7:
        raise ValueError(
            f"{env_var}={raw!r} must contain 7 comma-separated values "
            f"(joint_1..joint_6, gripper); got {len(parts)}"
        )
    return parts[:6]
