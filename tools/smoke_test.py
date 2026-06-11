"""BiQuestTeleoperator smoke test — fake Quest, no hardware.

Spins up a fake Quest as a WS client that sends synthetic xr_frame
messages through the relay server, while the BiQuestTeleoperator
subscribes from the other side. Verifies:

  - teleop.connect() blocks until the WS is open
  - get_action() returns the correct dict schema with home pose first
  - clutch ENGAGE on right grip captures origin, moving controller +X
    drives the right_joint_* values
  - trigger pull drives right_gripper.pos
  - left arm stays at home (untouched in this test)

Prereqs: relay running locally (vr-teleop-relay), lerobot installed,
DK1_URDF pointing at the URDF.

Run:
    python tools/smoke_test.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
import time

import websockets

from vr_teleop_kit.lerobot.bi_quest_teleop import (
    BiQuestTeleoperator,
    BiQuestTeleoperatorConfig,
)

# Optional override: python tools/smoke_test.py ws://127.0.0.1:8443/ws
WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8443/ws"


def _frame(right_pos, right_grip, right_trigger, left_pos=None, viewer_yaw_deg=0.0):
    def buttons(grip, trig):
        b = [{"p": False, "v": 0.0}] * 7
        b[0] = {"p": trig > 0.5, "v": float(trig)}
        b[1] = {"p": bool(grip), "v": 1.0 if grip else 0.0}
        return b

    if left_pos is None:
        left_pos = [-0.2, 1.4, -0.3]

    yaw = math.radians(viewer_yaw_deg)
    return {
        "type": "xr_frame",
        "t_client": time.time(),
        "controllers": {
            "right": {
                "position": list(right_pos),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "buttons": buttons(right_grip, right_trigger),
                "axes": [],
            },
            "left": {
                "position": list(left_pos),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "buttons": buttons(False, 0.0),
                "axes": [],
            },
        },
        "viewer": {
            "position": [0.0, 1.5, 0.0],
            "orientation": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)],
        },
    }


async def fake_quest_send(scenario):
    """Walk a list of (delay_seconds, frame_dict) through a fresh WS."""
    async with websockets.connect(WS_URL) as ws:
        for delay, frame in scenario:
            await asyncio.sleep(delay)
            await ws.send(json.dumps(frame))


def run_scenario_in_thread(scenario):
    def worker():
        asyncio.run(fake_quest_send(scenario))
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def fmt_action(a, hand):
    return [round(a[f"{hand}_joint_{i+1}.pos"], 3) for i in range(6)]


def main() -> None:
    print(f"connecting BiQuestTeleoperator to {WS_URL}")
    teleop = BiQuestTeleoperator(
        BiQuestTeleoperatorConfig(id="smoke", ws_url=WS_URL, publish_ik_state=False)
    )
    teleop.connect()
    assert teleop.is_connected, "teleop failed to connect"
    print("connected ✓")

    # Snapshot home action before any xr_frame arrives.
    a0 = teleop.get_action()
    print(f"\n[before xr_frame] action keys: {sorted(a0.keys())[:3]} ... ({len(a0)} total)")
    print(f"[before xr_frame] right arm: {fmt_action(a0, 'right')}  gripper={a0['right_gripper.pos']:.3f}")
    print(f"[before xr_frame] left  arm: {fmt_action(a0, 'left')}  gripper={a0['left_gripper.pos']:.3f}")

    # Scenario:
    #  1. controller idle, no clutch -> action stays at home
    #  2. clutch engage at (0, 1.4, -0.3)
    #  3. controller moves +X by 5 cm in two steps (still engaged)
    #  4. trigger 0.5 then 1.0 (gripper closes proportionally)
    #  5. release clutch
    scenario = [
        (0.30, _frame(right_pos=[0.0, 1.4, -0.3], right_grip=False, right_trigger=0.0)),
        (0.30, _frame(right_pos=[0.0, 1.4, -0.3], right_grip=True,  right_trigger=0.0)),  # engage
        (0.30, _frame(right_pos=[0.025, 1.4, -0.3], right_grip=True, right_trigger=0.0)),
        (0.30, _frame(right_pos=[0.05, 1.4, -0.3], right_grip=True, right_trigger=0.5)),  # half-close
        (0.30, _frame(right_pos=[0.05, 1.4, -0.3], right_grip=True, right_trigger=1.0)),  # full-close
        (0.30, _frame(right_pos=[0.05, 1.4, -0.3], right_grip=False, right_trigger=0.0)), # release
    ]
    run_scenario_in_thread(scenario)

    # Sample get_action while the scenario plays out.
    print("\nsampling get_action() at 5 Hz for 2.5 s while fake Quest plays:")
    samples = []
    t0 = time.time()
    while time.time() - t0 < 2.5:
        a = teleop.get_action()
        samples.append((time.time() - t0, a))
        time.sleep(0.2)

    # Print a compressed timeline focusing on right-arm state changes.
    print()
    print(f"{'t(s)':>6}  right joints[1..6]                       gripper   left j2  hint")
    last_right = None
    for t, a in samples:
        right = fmt_action(a, "right")
        gripper = a["right_gripper.pos"]
        left_j2 = round(a["left_joint_2.pos"], 3)
        hint = ""
        if right != last_right:
            hint = "← right qpos changed"
        last_right = right
        print(f"{t:6.2f}  {right}  {gripper:6.3f}  {left_j2:6.3f}  {hint}")

    # Sanity assertions.
    final = samples[-1][1]
    print()
    assert any(abs(final[f"right_joint_{j}.pos"] - a0[f"right_joint_{j}.pos"]) > 1e-3
               for j in (1, 2, 3, 4, 5, 6)), "right arm did not move from home — IK pipeline broken"
    print("✓ right arm joints moved from home pose during engagement")
    # Left arm should still be at home (we never sent left clutch).
    for j in range(1, 7):
        diff = abs(final[f"left_joint_{j}.pos"] - a0[f"left_joint_{j}.pos"])
        assert diff < 1e-6, f"left arm drifted at j{j}: {diff}"
    print("✓ left arm stayed at home pose (no drift)")
    # Gripper should respond to trigger.
    grippers = [a["right_gripper.pos"] for _, a in samples]
    assert max(grippers) > 0.4, f"right gripper did not close (max={max(grippers):.3f})"
    print(f"✓ right gripper responded to trigger (max value: {max(grippers):.3f})")

    teleop.disconnect()
    print("\ndone, teleop disconnected.")


if __name__ == "__main__":
    main()
