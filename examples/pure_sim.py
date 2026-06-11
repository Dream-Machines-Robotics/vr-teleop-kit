"""Pure-simulation VR teleop — no hardware follower.

Runs `BiQuestTeleoperator` against the Quest pose stream and publishes
`ik_state` back to the relay so `tools/viewer_client.py` can render the
resulting qpos in mujoco. Same IK pipeline as `teleop_bi_dk1.py` minus
the BiDK1Follower / port handling — useful for testing IK behavior
without needing the physical robot powered up.

Quick test workflow (no robot needed; DK1_URDF must point at the URDF):
  1. Relay server up:        vr-teleop-relay
     (USB via `adb reverse` or LAN HTTPS — see the README)
  2. Mujoco viewer up:       python tools/viewer_client.py
  3. This pure-sim loop:     python examples/pure_sim.py
  4. Quest browser → http://localhost:8443/ (USB) or
     https://<workstation-lan-ip>:8443/ (LAN) → Start Teleop →
     squeeze a grip, watch the arm move in the mujoco viewer.

Both arms are broadcast in every `ik_state` (`left_qpos` / `right_qpos`);
the viewer renders one arm per instance — default right, pass
`--arm left` (or run a second instance) for the left.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8443/ws")
    ap.add_argument("--freq", type=int, default=200, help="IK loop rate (Hz)")
    ap.add_argument("--id", default="vr-teleop-sim",
                    help="teleop id stamped into ik_state broadcasts; pair with "
                         "viewer_client --from-id to run several sims side by side")
    add_ik_cli_args(ap)
    args = ap.parse_args()

    init_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fallback = [0.0, float(np.pi / 2), float(np.pi / 2), 0.0, 0.0, 0.0]
    rest_left = parse_rest_pose_env("LEFT_REST_POSE", fallback)
    rest_right = parse_rest_pose_env("RIGHT_REST_POSE", fallback)

    ik_overrides = ik_kwargs_from_args(args)
    logger.info("======== pure-sim vr-teleop config ========")
    logger.info("ws_url       : %s", args.ws_url)
    logger.info("freq         : %d Hz", args.freq)
    logger.info("rest         : left=%s  right=%s",
                [round(x, 3) for x in rest_left],
                [round(x, 3) for x in rest_right])
    if ik_overrides:
        logger.info("CLI IK overrides: %s", ik_overrides)
    logger.info("==========================================")

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(
        id=args.id,
        ws_url=args.ws_url,
        rest_qpos_left=rest_left,
        rest_qpos_right=rest_right,
        **ik_overrides,
    ))

    teleop.connect()
    logger.info("pure-sim loop running at %d Hz — Ctrl-C to stop", args.freq)

    period = 1.0 / args.freq
    next_tick = time.perf_counter()
    try:
        while True:
            # get_action() drives one IK tick and (with publish_ik_state=True)
            # broadcasts the resulting qpos back to the relay for the viewer.
            teleop.get_action()
            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    main()
