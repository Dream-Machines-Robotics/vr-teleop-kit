"""VR bimanual teleop — live hardware example.

Runs the BiQuestTeleoperator against a real bimanual DK1, alongside the
relay server (`vr-teleop-relay`) and the WebXR client on the Quest.

At startup, the teleop's internal qpos is initialised to the rest pose
(LEFT_REST_POSE / RIGHT_REST_POSE env vars, seven comma-separated values
with the trailing gripper dropped; elbow-up default if unset), and
`ramp_to_rest()` drives the physical arms there in linearly-interpolated
steps (--rest-duration-s / --rest-steps). The operator can then squeeze
grip to engage and start teleoperating from the same anchor pose.

Pre-requisites:
  - Relay server + transport up (`vr-teleop-relay`; see the README for the
    USB `adb reverse` and LAN HTTPS transports)
  - Quest browser at http://localhost:8443/ (USB) or
    https://<workstation-lan-ip>:8443/ (LAN), in passthrough VR
  - The DK1 driver installed:
    pip install git+https://github.com/robot-learning-co/trlc-dk1
  - DK1_URDF pointing at urdf/follower/TRLC-DK1-Follower.urdf from that
    repository (or pass --urdf-path)

Run:
    python examples/teleop_bi_dk1.py \\
        --left-port /dev/serial/by-path/<left> \\
        --right-port /dev/serial/by-path/<right>
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from vr_teleop_kit.lerobot.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)
from vr_teleop_kit.lerobot.cli import (
    add_ik_cli_args,
    ik_kwargs_from_args,
    parse_rest_pose_env,
)

from lerobot.utils.utils import init_logging
from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig
from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig

def ramp_to_rest(
    follower,
    target: dict[str, float],
    duration_s: float,
    steps: int,
    logger: logging.Logger,
) -> None:
    """Smoothly drive the follower from its current pose to `target` over
    `duration_s` seconds in `steps` linearly-interpolated increments.
    Caller builds `target` matching the follower's action schema (prefixed
    `left_/right_` for `BiDK1Follower`, unprefixed for `DK1Follower`)."""
    obs = follower.get_observation()
    start = {k: float(obs.get(k, target[k])) for k in target}

    logger.info("ramping to rest pose over %.1fs in %d steps", duration_s, steps)
    dt = duration_s / max(1, steps)
    for i in range(1, steps + 1):
        alpha = i / steps
        command = {k: start[k] + alpha * (target[k] - start[k]) for k in target}
        follower.send_action(command)
        time.sleep(dt)
    logger.info("rest pose reached")


def _build_target(rest_joints: list[float], prefix: str) -> dict[str, float]:
    """Build a follower-action target dict from a 6-joint rest pose, with
    gripper open at 0.0. `prefix` is "" (single-arm) or "left_"/"right_"
    (bimanual)."""
    keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    return {f"{prefix}{k}": float(v) for k, v in zip(keys, list(rest_joints) + [0.0])}


def _strip_prefix(action: dict[str, float], prefix: str) -> dict[str, float]:
    """Pluck only the keys matching `prefix` (e.g. 'left_') and remove it.
    Used to feed a single-arm `DK1Follower` from a bimanual action dict."""
    return {k.removeprefix(prefix): v for k, v in action.items() if k.startswith(prefix)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-port", required=True,
                    help="left follower serial device (prefer stable /dev/serial/by-path/... names; "
                         "/dev/ttyACMx reorders on every plug cycle)")
    ap.add_argument("--right-port", required=True,
                    help="right follower serial device (see --left-port)")
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws", help="relay server WS URL")
    ap.add_argument("--freq", type=int, default=200, help="teleop loop rate (Hz)")
    ap.add_argument("--joint-velocity-scaling", type=float, default=0.2,
                    help="passes through to the follower config (Bi/DK1FollowerConfig)")
    ap.add_argument("--rest-duration-s", type=float, default=3.0,
                    help="seconds to ramp arms to rest pose at startup")
    ap.add_argument("--rest-steps", type=int, default=90,
                    help="number of interpolation steps in the startup ramp")
    ap.add_argument("--arm", choices=("both", "left", "right"), default="both",
                    help="run bimanual ('both') or just one arm. With 'left' or "
                         "'right' the BiQuestTeleoperator still tracks both, but "
                         "only the chosen arm's actions go to a single DK1Follower; "
                         "the other controller's IK is computed and ignored.")
    add_ik_cli_args(ap)
    args = ap.parse_args()

    init_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fallback = [0.0, float(np.pi / 2), float(np.pi / 2), 0.0, 0.0, 0.0]
    rest_left = parse_rest_pose_env("LEFT_REST_POSE", fallback)
    rest_right = parse_rest_pose_env("RIGHT_REST_POSE", fallback)
    ik_overrides = ik_kwargs_from_args(args)
    logger.info("======== vr-teleop config ========")
    logger.info("arm          : %s", args.arm)
    logger.info("ports        : left=%s  right=%s", args.left_port, args.right_port)
    logger.info("rest         : left=%s  right=%s",
                [round(x, 3) for x in rest_left],
                [round(x, 3) for x in rest_right])
    if ik_overrides:
        logger.info("CLI IK overrides: %s", ik_overrides)
    logger.info("==================================")

    if args.arm == "both":
        follower = BiDK1Follower(BiDK1FollowerConfig(
            left_arm_port=args.left_port,
            right_arm_port=args.right_port,
            joint_velocity_scaling=args.joint_velocity_scaling,
        ))
    else:
        port = args.left_port if args.arm == "left" else args.right_port
        follower = DK1Follower(DK1FollowerConfig(
            port=port,
            joint_velocity_scaling=args.joint_velocity_scaling,
        ))

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id="vr-teleop",
        ws_url=args.ws_url,
        rest_qpos_left=rest_left,
        rest_qpos_right=rest_right,
        **ik_overrides,
    ))

    teleop.connect()
    follower.connect()

    # Build the target dict for the active follower(s) and ramp.
    if args.arm == "both":
        target = {**_build_target(rest_left, "left_"),
                  **_build_target(rest_right, "right_")}
    else:
        rest = rest_left if args.arm == "left" else rest_right
        target = _build_target(rest, "")
    ramp_to_rest(follower, target, args.rest_duration_s, args.rest_steps, logger)

    period = 1.0 / args.freq
    logger.info("vr-teleop running at %d Hz (arm=%s); Ctrl-C to stop", args.freq, args.arm)
    arm_prefix = "" if args.arm == "both" else f"{args.arm}_"
    get_torques = getattr(follower, "get_joint_torques", None)
    try:
        next_tick = time.perf_counter()
        while True:
            action = teleop.get_action()
            if args.arm != "both":
                action = _strip_prefix(action, arm_prefix)
            follower.send_action(action)
            # Per-tick force haptic: read gripper torque and forward to the
            # teleop. Best-effort; silent if the follower doesn't expose it.
            if get_torques is not None:
                try:
                    torques = get_torques()
                except Exception:
                    torques = None
                if torques:
                    teleop.send_feedback({"torques": torques})
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        try:
            teleop.disconnect()
        finally:
            follower.disconnect()


if __name__ == "__main__":
    main()
