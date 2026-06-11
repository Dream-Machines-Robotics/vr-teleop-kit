"""Mujoco viewer driven by the VR teleop server.

Connects to the server's `/ws` as a *passive listener*, reads broadcast
`ik_state` messages, and applies the qpos to a local mujoco model + viewer.

One viewer = one arm. The teleop publishes `left_qpos` and `right_qpos`
in each `ik_state`; this script picks one via `--arm` (default: right).
To watch both arms, run two instances:

    uv run python tools/viewer_client.py --arm right &
    uv run python tools/viewer_client.py --arm left

While the relay is running (DK1_URDF must point at the URDF, or pass
--urdf):
    python tools/viewer_client.py
    python tools/viewer_client.py --url wss://<lan-ip>:8443/ws --arm left

On macOS the script transparently re-execs itself under `mjpython`
(shipped with the mujoco wheel) — `mujoco.viewer.launch_passive` needs
the main thread for UI on Darwin, which the regular Python launcher
doesn't provide. The behavior on Linux is unchanged.

Quit by closing the viewer window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _reexec_under_mjpython_if_needed() -> None:
    """On macOS, `mujoco.viewer.launch_passive` only works under `mjpython`.
    If we're on Darwin and not already running under it, find mjpython in the
    current venv (sibling of sys.executable, since uv-managed envs install it
    there) or PATH and `os.execve` into it. No-op on Linux."""
    if platform.system() != "Darwin":
        return
    if os.environ.get("_MJPYTHON_REEXEC") == "1":
        return  # already re-execed
    cand = Path(sys.executable).with_name("mjpython")
    mjpython = str(cand) if cand.exists() else shutil.which("mjpython")
    if mjpython is None:
        raise RuntimeError(
            "macOS requires `mjpython` for mujoco's passive viewer, but no "
            "mjpython binary was found next to sys.executable or in PATH. "
            "Ensure `mujoco` is installed in the same environment."
        )
    env = os.environ.copy()
    env["_MJPYTHON_REEXEC"] = "1"
    os.execve(mjpython, [mjpython, *sys.argv], env)


_reexec_under_mjpython_if_needed()

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import websockets  # noqa: E402

from vr_teleop_kit.ik.model import build_model_with_tool0_site  # noqa: E402

DEFAULT_URL = "ws://127.0.0.1:8443/ws"


async def consume(ws, model, data, viewer, qpos_key: str, from_id: str | None) -> None:
    while viewer.is_running():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
        except asyncio.TimeoutError:
            # Keep the viewer responsive between messages (window events).
            viewer.sync()
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("type") != "ik_state":
            continue
        # When several teleops publish to the same relay (e.g. a
        # with/without-limit comparison), --from-id picks one stream.
        if from_id is not None and msg.get("teleop_id") != from_id:
            continue
        # Prefer the per-arm key; fall back to the legacy compat field
        # `qpos` (which the teleop populates with the right arm's qpos
        # for older clients).
        qpos = msg.get(qpos_key) or msg.get("qpos")
        if not qpos or len(qpos) < 6:
            continue
        # 6 = arm only, 8 = arm + gripper sliders.
        data.qpos[: min(len(qpos), model.nq)] = qpos[: min(len(qpos), model.nq)]
        mujoco.mj_forward(model, data)
        viewer.sync()


def _ssl_context_for(url: str):
    """For `wss://` we build a permissive SSL context (no cert validation,
    no hostname check). The relay uses a self-signed LAN cert that no
    third party trusts; the viewer is a dev tool talking to a known
    machine on the operator's own network, so validation adds no
    security but would otherwise refuse the handshake."""
    if not url.startswith("wss://"):
        return None
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def runner(url: str, model, data, viewer, qpos_key: str, from_id: str | None) -> None:
    print(f"connecting to {url}  (reading {qpos_key!r}"
          + (f", teleop id {from_id!r})" if from_id else ")"))
    async with websockets.connect(url, ssl=_ssl_context_for(url)) as ws:
        print("connected, listening for ik_state ...")
        await consume(ws, model, data, viewer, qpos_key, from_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help=f"WebSocket URL (default: {DEFAULT_URL})")
    ap.add_argument("--arm", choices=("left", "right"), default="right",
                    help="which arm's qpos to render (the teleop broadcasts both).")
    ap.add_argument("--from-id", default=None,
                    help="only render ik_state from this teleop id (for running "
                         "several teleops against one relay side by side).")
    ap.add_argument("--urdf", default=None,
                    help="path to the DK1 follower URDF (default: DK1_URDF env var).")
    args = ap.parse_args()
    qpos_key = f"{args.arm}_qpos"

    model, data = build_model_with_tool0_site(args.urdf)

    # Start at home pose so the viewer has something to show before any
    # ik_state arrives.
    data.qpos[:] = 0
    data.qpos[1] = np.pi / 2
    data.qpos[2] = np.pi / 2
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        try:
            asyncio.run(runner(args.url, model, data, viewer, qpos_key, args.from_id))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
