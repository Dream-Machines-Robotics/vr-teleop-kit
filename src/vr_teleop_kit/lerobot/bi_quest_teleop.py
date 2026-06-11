"""BiQuestTeleoperator — bimanual VR teleoperation via WebXR/Quest.

Subscribes to a relay-mode FastAPI server's `/ws` and reads `xr_frame`
broadcasts (controller poses + buttons + headset pose). Per arm,
maintains a `ClutchPoseMapper` and a `DecoupledIKSolver`; on the rising edge
of the grip button, captures the engage frame and starts running
differential IK each tick. `get_action()` returns the joint action
dict matching `BiDK1Follower.action_features`:

    {left,right}_joint_{1..6}.pos   (radians)
    {left,right}_gripper.pos        (normalized 0..1)

Standard LeRobot-style loop:

    teleop = BiQuestTeleoperator(BiQuestTeleoperatorConfig(...))
    follower = BiDK1Follower(...)
    teleop.connect(); follower.connect()
    while True:
        follower.send_action(teleop.get_action())
        time.sleep(1/freq)

Per-tick pipeline inside `_update_arm`:

  1. Staleness gate — if no `xr_frame` for > XR_FRAME_STALE_TIMEOUT_S,
     mark `needs_reanchor` and skip the tick. On the first fresh tick
     after recovery, the engage frame is silently re-captured so the
     delta-since-engage restarts at zero (no catch-up motion).
  2. EMA pose filter — smooths sub-tick WebXR jitter on the controller
     pose before downstream consumers see it (`pose_filter_alpha`).
  3. Precision button (A/X) — on transition, re-anchor the engage
     frame before changing the mapper gains. On the legacy absolute
     path this prevents the accumulated delta being reinterpreted
     under the new scale; on the reach-limited incremental path it simply
     realigns hand↔EE correspondence.
  4. Clutch edges — rising → `_anchor_mapper` (captures controller
     pose, EE pose, J4 anchor as rotation pivot, applies yaw
     correction); falling → mapper.disengage().
  5. Yaw correction — R_engage = R_CALIB · R_y(-yaw_now). R_CALIB is a
     fixed body→arm-base rotation; the runtime yaw_now subtraction
     keeps "controller forward" = "arm forward" wherever the operator
     happens to be facing in the room.
  6. IK step — FK at the last commanded qpos gives the current EE
     pose, passed into mapper.target() to enable the absorbing demand
     reach limits (rot_reach_limit / pos_reach_limit); DecoupledIKSolver.solve() then steps
     qpos[:6] toward the reach-limited target. Gripper qpos[6:8] mirrored
     from the trigger for the sim viewer.
  7. Haptic mix — max of (limit_pressure, pos_err_norm,
     singularity_proximity, wrist_gimbal_proximity); EMA-smoothed and
     broadcast in ik_state for the client to vibrate the matching
     controller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lerobot is required to use BiQuestTeleoperator. "
        "Run from a project environment with lerobot installed."
    ) from e

try:
    import websockets
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "websockets is required to use BiQuestTeleoperator. "
        "Install with: uv add websockets   (or pip install websockets)"
    ) from e

from ..core.pose_mapping import ClutchPoseMapper
from ..ik.model import DEFAULT_Q_REST
from ..ik.decoupled_ik import DecoupledIKSolver

logger = logging.getLogger(__name__)


# Default rotation taking Quest `local-floor` world axes into the arm base
# frame: arm_x = -quest_z (operator-forward), arm_y = -quest_x
# (operator-left), arm_z = +quest_y (up). Derived empirically for the
# original lab mounting; override via `BiQuestTeleoperatorConfig.r_calib`
# if your robot faces the operator differently (see the README).
DEFAULT_R_CALIB = [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

GRIP_BUTTON_INDEX = 1  # boolean: clutch
TRIGGER_BUTTON_INDEX = 0  # analog 0..1: gripper closure
PRECISION_BUTTON_INDEX = 4  # boolean: A (right) / X (left) — hold for precision scale
# Note: the precision-scale factor itself is on the config dataclass
# (`precision_factor`) so the web UI can tune it live.
HANDOFF_BUTTON_INDEX = 5  # boolean: B (right) / Y (left) — intervention handoff signal
# Thumbstick click — per-arm "go home" ramp. Free in the WebXR/OpenXR
# standard mapping (Quest controllers expose 0=trigger, 1=squeeze,
# 3=thumbstick-press, 4=A/X, 5=B/Y). Rising edge starts a time-based
# linear ramp from current qpos to `rest_qpos_{hand}`; recording stays
# untouched because the ramp targets are published as the normal action.
REST_RAMP_BUTTON_INDEX = 3
ARM_DOFS = 6
NQ = 8  # arm (6) + 2 gripper-finger sliders

# If no xr_frame has been received in this many seconds, treat the
# controller stream as stale: force-disengage any engaged arm and
# refuse to start a new engagement. Protects against the
# "buffer-burst → catch-up motion" failure mode when the WS connection
# (typically the cloudflared tunnel) stalls and then floods.
XR_FRAME_STALE_TIMEOUT_S = 0.2

# Gripper prismatic range from the URDF (`gripper_left`, `gripper_right`).
# Both fingers share the same qpos scale: -0.045 = closed (fingers converge),
# 0.001 = barely open. Trigger=0 → open (upper), trigger=1 → closed (lower).
GRIPPER_QPOS_OPEN = 0.001
GRIPPER_QPOS_CLOSED = -0.045


def _yaw_from_quat_xyzw(q_xyzw) -> float:
    """Yaw about world-up (+Y), in radians, from a (x, y, z, w) WebXR quat."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _R_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@TeleoperatorConfig.register_subclass("bi_quest_teleop")
@dataclass
class BiQuestTeleoperatorConfig(TeleoperatorConfig):
    """Config for BiQuestTeleoperator.

    `ws_url` points at the FastAPI relay server hosting the WebXR page.
    `publish_ik_state` makes the teleop send its computed qpos back to the
    server so the existing viewer_client.py + Quest UI keep mirroring.
    `connect_timeout_s` bounds connect()'s wait for the WS handshake.

    `rest_qpos_left` / `rest_qpos_right` are six joint angles (rad) per arm.
    They serve two roles: (a) the teleop initialises each arm's qpos here
    so the first `send_action` moves the physical follower toward the
    rest pose; (b) the IK Tikhonov bias pulls toward this pose, breaking
    the elbow-flip ambiguity. Default is the elbow-up
    [0, π/2, π/2, 0, 0, 0]; override per machine to match wherever your
    setup parks the arms.
    """

    ws_url: str = "ws://127.0.0.1:8443/ws"
    publish_ik_state: bool = True
    connect_timeout_s: float = 5.0
    # Path to the DK1 follower URDF (deliberately not vendored — clone
    # github.com/robot-learning-co/trlc-dk1). Empty string falls
    # back to the DK1_URDF environment variable.
    urdf_path: str = ""
    # 3x3 rotation (row-major) taking Quest world vectors into the arm
    # base frame. See DEFAULT_R_CALIB above for the convention and the
    # README for how to re-derive it for a different mounting.
    r_calib: list[list[float]] = field(default_factory=lambda: [row[:] for row in DEFAULT_R_CALIB])
    rest_qpos_left: list[float] = field(default_factory=lambda: DEFAULT_Q_REST.tolist())
    rest_qpos_right: list[float] = field(default_factory=lambda: DEFAULT_Q_REST.tolist())

    # IK / mapping knobs.
    lam: float = 0.05  # Position-solve base damping
    lam0: float = 0.15  # Adaptive-damping ramp amplitude near singular
    w0: float = 0.05  # Manipulability threshold where ramp starts
    mu: float = 0.02  # Tikhonov stiffness toward q_rest
    # Wrist (orientation sub-solve) damping — same recipe, separate scale:
    # the wrist manipulability |det J_rot| ≈ |cos θ5| lives in [0, 1].
    lam_rot: float = 0.05  # Orientation-solve base damping
    lam0_rot: float = 0.4  # Extra damping ramp amplitude near wrist gimbal
    w0_rot: float = 0.5  # Wrist-manipulability threshold where ramp starts
    # Reach limits: the target may run at most this far ahead of the arm's
    # CURRENT pose; the excess is absorbed, mouse-at-screen-edge style (see
    # ClutchPoseMapper docstring). Rotation becomes incremental under the
    # reach limit — pressing past a joint stop / gimbal can neither build up the
    # near-180° error that shakes the wrist at the Δq cap, nor wrap around
    # and snap in from the other side; position overshoot is absorbed the
    # same way so reversal bites immediately. Cost: hand↔EE correspondence
    # drifts within an engagement (absorbed motion is gone; scale_rotation≠1
    # additionally rate-scales rotation increments, path-dependent on curved
    # hand paths) — re-clutching realigns. Smaller reach limit = firmer "wall" at
    # stops and less posture windup near gimbal; larger = more headroom for
    # very fast motions before absorption. 0 disables either reach limit.
    rot_reach_limit: float = 0.6  # rad (~34°)
    pos_reach_limit: float = 0.25  # m
    # Solver backstop: park the wrist when the orientation error exceeds
    # this (rad). With the reach limit on it never fires; exposed mainly so the
    # no-limit comparison demo can disable it (> 3.15 = off, since the
    # error angle can't exceed π) and show the raw absolute-mapping flip
    # at 180°.
    rot_err_hold: float = 2.2
    scale_translation: float = 1.5  # controller→EE translation gain
    scale_rotation: float = 1.5  # controller→EE rotation gain (<1 = softer)
    # While the A/X precision button is held, both scale_translation and
    # scale_rotation are multiplied by this factor for finer positioning.
    # 0.5 is the historical default; the web UI exposes a slider.
    precision_factor: float = 0.5
    # EMA on the incoming controller pose (per arm). 1.0 = no smoothing
    # (raw passthrough), lower = more smoothing + more latency. Smooths
    # sub-tick WebXR jitter before it reaches the IK / pose mapper. At
    # 200 Hz IK rate, 0.5 ≈ 7 ms time constant, 0.3 ≈ 22 ms.
    pose_filter_alpha: float = 0.8
    # Per-joint Δq cap (rad/tick). Bounds the worst-case single-joint snap
    # from any source (e.g. workspace-edge catch-up). None to disable.
    # The web UI exposes two scalar shortcuts (`_pos` for joints 1-3 driving
    # EE position, `_rot` for joints 4-6 driving EE orientation) — operators
    # found that one shared cap held wrist motion back while position was
    # fine, so the groups stay independently tunable. Defaults: 0.06
    # rad/tick for BOTH groups (12 rad/s at a 200 Hz loop, 3 rad/s at
    # 50 Hz). Rotation originally defaulted higher for wrist headroom;
    # after the reach limits landed it was tuned down to match position
    # on hardware — still responsive, and any residual catch-up after
    # pressing into a limit stays gentle.
    max_dq_per_joint: list[float] | None = field(default_factory=lambda: [0.06] * 6)
    # Web-tunable shortcuts. Writing either through `config_update` (a)
    # rebuilds `max_dq_per_joint = [pos]*3 + [rot]*3` and (b) pushes the
    # new array into each live DecoupledIKSolver, so a slider drag takes
    # effect on the very next solve. Initial values mirror the per-joint
    # default above; not used as a source-of-truth after that.
    max_dq_per_joint_scalar_pos: float = 0.06
    max_dq_per_joint_scalar_rot: float = 0.06
    # ── Force haptic (gripper torque → controller vibration) ──
    # Linear scaling with a dead zone:
    #   intensity = clip((|τ| - threshold) / (max - threshold), 0, 1)
    # Defaults sized to the DK1 gripper (max_gripper_torque=1.0 Nm in the
    # follower config). Bump the per-arm threshold to silence idle baseline
    # buzz on that arm specifically (static holding torque differs between
    # individual grippers, so we deadband independently); bump `max_nm` if
    # you want the buzz to stay subtle even at max grip. The web Settings
    # panel exposes a "Calibrate" button that samples idle torque for a few
    # seconds and writes the per-arm thresholds via config_update.
    force_haptic_threshold_nm_left: float = 0.35
    force_haptic_threshold_nm_right: float = 0.35
    force_haptic_max_nm: float = 1.0
    # Inertia / kinetic-friction compensation. Adds `k * |v|` to the
    # effective threshold, where `v` is the numerical gripper velocity
    # (units of normalized pos per second, computed from successive
    # `gripper.pos` samples sent alongside the torque). Higher = more
    # masking during fast opens/closes. 0 disables velocity compensation.
    force_haptic_velocity_comp_nm: float = 0.5
    # Master on/off for the force haptic, toggled live from the Settings
    # panel on the web. When False the intensity is forced to 0; threshold
    # / max / velocity-comp values are kept untouched so flipping back on
    # picks up where you were.
    force_haptic_enabled: bool = True
    # Duration (s) of the per-arm "go home" ramp triggered by a thumbstick
    # press. The ramp linearly interpolates qpos[:6] from current to
    # `rest_qpos_{hand}` over this window. On completion, if the arm is
    # still engaged, the engage frame is re-anchored at the rest pose so
    # the operator's hand motion resumes with zero delta — no jump.
    rest_ramp_duration_s: float = 2.0


class BiQuestTeleoperator(Teleoperator):
    config_class = BiQuestTeleoperatorConfig
    name = "bi_quest_teleop"

    def __init__(self, config: BiQuestTeleoperatorConfig) -> None:
        super().__init__(config)
        self.config = config

        self._r_calib = np.asarray(config.r_calib, dtype=float)
        if self._r_calib.shape != (3, 3):
            raise ValueError(
                f"r_calib must be a 3x3 rotation matrix, got shape {self._r_calib.shape}"
            )

        # Per-arm state. Lock guards reads/writes from the LeRobot thread vs
        # the WS receiver thread.
        self._lock = threading.Lock()
        self._latest_xr_frame: dict | None = None
        # Wall-clock timestamp of the most recent xr_frame. Used to detect
        # network/WS staleness and force a safe disengage so we never
        # capture an engage frame from outdated controller data.
        self._last_xr_frame_time: float = 0.0

        rest_qpos = {
            "left": np.asarray(config.rest_qpos_left, dtype=float),
            "right": np.asarray(config.rest_qpos_right, dtype=float),
        }
        for hand, q in rest_qpos.items():
            if q.shape != (ARM_DOFS,):
                raise ValueError(f"rest_qpos_{hand} must have {ARM_DOFS} values, got shape {q.shape}")

        self._arms: dict[str, dict] = {}
        for hand in ("left", "right"):
            q_rest = rest_qpos[hand]
            qpos_init = np.zeros(NQ)
            qpos_init[:ARM_DOFS] = q_rest
            arm_solver = DecoupledIKSolver(
                urdf_path=config.urdf_path or None,
                lam_pos=config.lam,
                lam0=config.lam0,
                w0=config.w0,
                mu=config.mu,
                lam_rot=config.lam_rot,
                lam0_rot=config.lam0_rot,
                w0_rot=config.w0_rot,
                rot_err_hold=config.rot_err_hold,
                q_rest=q_rest.copy(),
                max_dq_per_joint=config.max_dq_per_joint,
            )
            self._arms[hand] = {
                "solver": arm_solver,
                "mapper": ClutchPoseMapper(
                    R=self._r_calib.copy(),
                    scale=config.scale_translation,
                    scale_rotation=config.scale_rotation,
                    rot_reach_limit=config.rot_reach_limit,
                    pos_reach_limit=config.pos_reach_limit,
                ),
                "qpos": qpos_init,
                "last_grip": False,
                "last_precision": False,  # tracks A/X button — re-anchor on edge
                "trigger": 0.0,
                "engaged": False,
                "haptic": 0.0,  # smoothed 0..1, broadcast in ik_state
                "force_haptic": 0.0,  # gripper-torque-driven haptic (set by send_feedback)
                # Numerical gripper-velocity tracking for inertia masking.
                # `prev_gripper_pos` / `prev_gripper_t` are filled on the
                # first feedback tick; `vel_filt` keeps an EMA-smoothed
                # value (rad-equivalent / s) so per-tick noise doesn't
                # spike the threshold.
                "prev_gripper_pos": None,
                "prev_gripper_t": None,
                "gripper_vel_filt": 0.0,
                "needs_reanchor": False,  # set during stale → re-anchor on recovery
                "pos_filt": None,  # EMA-smoothed controller position
                "quat_filt": None,  # EMA-smoothed controller orientation (wxyz)
                # Per-arm "go home" ramp (thumbstick-click trigger). While
                # `ramp_active`, qpos[:6] is linearly interpolated from
                # `ramp_start_q` toward `ramp_target_q` over
                # `rest_ramp_duration_s`, bypassing IK. The mapper's engage
                # frame is re-anchored at the rest pose on completion so
                # the next operator motion has zero delta — no jump.
                "last_rest_button": False,
                "ramp_active": False,
                "ramp_start_q": np.zeros(ARM_DOFS),
                "ramp_target_q": q_rest.copy(),
                "ramp_start_t": 0.0,
            }
        # Raw per-hand button state, updated synchronously from every
        # xr_frame in the WS reader thread. Decoupled from `arm["engaged"]`
        # (which only updates when get_action runs) so an intervention
        # orchestrator can read live grip/B-Y state even while the policy
        # drives — i.e. when nothing is calling get_action(). Read via
        # is_engaged() / is_handoff_pressed() under self._lock.
        self._buttons: dict[str, dict[str, bool]] = {
            "left": {"grip": False, "handoff": False},
            "right": {"grip": False, "handoff": False},
        }

        # WS plumbing.
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_stop: threading.Event | None = None
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._ws_connected = threading.Event()

        # Idle-torque calibration window (web UI "Calibrate" button). When
        # active, every torque sample fed into `_apply_torque_feedback`
        # contributes to per-arm peak |τ| tracking. On window end the peaks
        # are used to compute new per-arm thresholds (peak + margin) and the
        # result is broadcast as `haptic_calibrate_result` for the web UI to
        # display and persist. Guarded by `self._lock`.
        self._haptic_calib: dict | None = None

        # Measured rate of `get_action()` calls. EMA-smoothed in get_action;
        # published in ik_state so the web UI knows the real tick rate
        # (e.g. 200 Hz for the example loop, the dataset FPS under
        # lerobot-record) and can show the joint-Δq-cap slider in honest
        # rad/s.
        self._last_get_action_t: float | None = None
        self._loop_hz: float | None = None

    # ---------- Teleoperator interface ----------

    @property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {}
        for hand in ("left", "right"):
            for j in range(1, ARM_DOFS + 1):
                feats[f"{hand}_joint_{j}.pos"] = float
            feats[f"{hand}_gripper.pos"] = float
        return feats

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._ws_connected.is_set()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        self._ws_stop = threading.Event()
        self._ws_thread = threading.Thread(
            target=self._ws_thread_main, name="bi-quest-teleop-ws", daemon=True
        )
        self._ws_thread.start()
        if not self._ws_connected.wait(timeout=self.config.connect_timeout_s):
            raise RuntimeError(f"BiQuestTeleoperator: timed out connecting to {self.config.ws_url}")
        logger.info("BiQuestTeleoperator connected to %s", self.config.ws_url)

    def disconnect(self) -> None:
        if self._ws_stop is not None:
            self._ws_stop.set()
        if self._ws_loop is not None and self._ws is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._ws.close(), self._ws_loop)
                fut.result(timeout=1.0)
            except Exception:
                pass
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=2.0)
        self._ws_connected.clear()
        logger.debug("BiQuestTeleoperator disconnected")

    def send_feedback(self, feedback: dict) -> None:
        """Push runtime feedback from the orchestrator (a rollout loop or
        examples/teleop_bi_dk1.py) into the teleop's per-arm haptic state. The values
        are broadcast in the next ``ik_state`` and mixed with the existing
        IK-derived haptic on the web client.

        Recognised keys today:
            ``torques`` — dict of motor torque readings in Nm, keyed by
                          ``{left,right}_gripper.torque`` (and friends).
                          Only ``*_gripper.torque`` is consumed in v1.

        Anything unrecognised is silently ignored so future callers can
        add fields without breaking older teleop builds.
        """
        if not isinstance(feedback, dict):
            return
        torques = feedback.get("torques")
        if isinstance(torques, dict):
            self._apply_torque_feedback(torques)

    def publish_state(self) -> None:
        """Force one ``ik_state`` broadcast on the WS without waiting for
        the next ``get_action()`` call.

        ``ik_state`` is normally piggy-backed onto ``get_action()`` (see
        the call at the end of ``get_action``), which means an orchestrator
        that updates haptic state outside the action loop (e.g. sending a
        zero-torque release frame when the human hands control back to
        the policy) would otherwise leave the Quest controller buzzing
        until the next correction starts. Callers should invoke this
        whenever teleop-side state needs to reach the client now.
        """
        if self.config.publish_ik_state and self._ws is not None and self._ws_loop is not None:
            self._publish_ik_state_async()

    def _apply_torque_feedback(self, torques: dict) -> None:
        """Map raw gripper torques (Nm) to per-arm 0..1 haptic intensity
        via a velocity-aware dead-zone linear scaling.

        Effective threshold widens when the gripper is moving fast::

            θ_eff = θ_base + k_v * |v_gripper|

        That masks the inertial / kinetic-friction torque spike during
        rapid opens/closes (otherwise the operator feels a buzz even with
        nothing in the jaws). At rest, behavior is identical to the
        previous linear-with-deadzone model. Below ``θ_eff`` intensity is
        0; at ``force_haptic_max_nm`` (and above) it's 1.

        Velocity comes from successive ``gripper.pos`` samples included
        alongside the torque by the follower. First call after a fresh
        connect can't differentiate (no prior sample), so velocity starts
        at 0 — that tick uses the static threshold, fine in practice.
        """
        ceiling = max(1e-6, float(self.config.force_haptic_max_nm))
        kv = max(0.0, float(self.config.force_haptic_velocity_comp_nm))
        # EMA smoothing on the numerical velocity to keep the threshold
        # itself from juddering at every tick. 0.4 ≈ ~5 ms time constant
        # at 60 Hz feedback rate; enough to ride out 1-tick noise without
        # introducing visible lag.
        vel_alpha = 0.4
        now = time.time()
        with self._lock:
            for hand in ("left", "right"):
                tau_key = f"{hand}_gripper.torque"
                pos_key = f"{hand}_gripper.pos"
                if tau_key not in torques:
                    # Single-arm wrapper sends unprefixed keys directly — but
                    # rewrites them to be prefixed before reaching here. So
                    # we shouldn't normally hit this branch.
                    continue
                tau = abs(float(torques[tau_key]))
                arm = self._arms[hand]
                # Numerical velocity from successive pos samples.
                if pos_key in torques:
                    pos = float(torques[pos_key])
                    prev_pos = arm["prev_gripper_pos"]
                    prev_t = arm["prev_gripper_t"]
                    if prev_pos is not None and prev_t is not None:
                        dt = max(now - prev_t, 1e-3)
                        v_raw = (pos - prev_pos) / dt
                        arm["gripper_vel_filt"] = (1.0 - vel_alpha) * arm[
                            "gripper_vel_filt"
                        ] + vel_alpha * v_raw
                    arm["prev_gripper_pos"] = pos
                    arm["prev_gripper_t"] = now
                # Per-arm threshold — static holding torque differs between
                # individual grippers, so we deadband independently.
                threshold_base = max(
                    0.0, float(getattr(self.config, f"force_haptic_threshold_nm_{hand}"))
                )
                threshold_eff = threshold_base + kv * abs(arm["gripper_vel_filt"])
                # When the deadband meets or exceeds the ceiling there's no
                # dynamic range left — force intensity to 0 instead of
                # letting `(tau - threshold) / tiny_span` saturate to 1
                # and vibrate at max. The auto-calibration also bumps
                # `force_haptic_max_nm` to keep this from happening in
                # practice; this guard is the belt under the suspenders.
                if threshold_eff >= ceiling:
                    intensity = 0.0
                else:
                    span = ceiling - threshold_eff
                    intensity = max(0.0, min(1.0, (tau - threshold_eff) / span))
                # Single-bit gate: when the operator has turned force
                # haptic off from the Settings panel, we still compute
                # everything (so the toggle can light back up instantly)
                # but publish 0 so the controller stops buzzing.
                if not self.config.force_haptic_enabled:
                    intensity = 0.0
                arm["force_haptic"] = intensity
                # Idle-torque calibration window: track per-arm peak |τ| so
                # we can write a fresh threshold = peak + margin when the
                # window closes. Tick the calibration timer below the loop.
                if self._haptic_calib is not None:
                    self._haptic_calib["peak"][hand] = max(
                        self._haptic_calib["peak"][hand], tau
                    )
            # Close calibration window if duration elapsed. Done after the
            # per-hand loop so both arms get the same final sample.
            if self._haptic_calib is not None:
                calib = self._haptic_calib
                if now - calib["start_t"] >= calib["duration_s"]:
                    self._finalize_haptic_calibration()

    # ---------- Intervention (handoff) hooks ----------
    # Surfaced for human-in-the-loop orchestrators (e.g. the HG-DAgger
    # strategy in our LeRobot fork, which this stack was built for).
    # Reads come from `self._buttons` which the WS reader populates from
    # every xr_frame — independent of whether get_action is running. That
    # matters because an orchestrator typically calls get_action only
    # while the human is correcting; if these getters read
    # `arm["engaged"]` instead, the listener would never see a fresh
    # value while the policy drives and the handoff would never fire.
    # Rising-edge detection is the listener's job; we report the level.

    def is_engaged(self) -> bool:
        with self._lock:
            return self._buttons["left"]["grip"] or self._buttons["right"]["grip"]

    def is_handoff_pressed(self) -> bool:
        with self._lock:
            return self._buttons["left"]["handoff"] or self._buttons["right"]["handoff"]

    def is_pause_pressed(self) -> bool:
        """Right-hand B — pause/resume (stop/continue the policy).

        An orchestrator can split the handoff button per hand: right B
        owns pause/resume, left Y owns reverse (see ``is_reverse_pressed``).
        Level only; the consumer does rising-edge detection.
        """
        with self._lock:
            return self._buttons["right"]["handoff"]

    def is_reverse_pressed(self) -> bool:
        """Left-hand Y — reverse (e.g. replay buffered actions backward).

        Counterpart to ``is_pause_pressed`` (right B). Level only; the
        consumer does rising-edge detection.
        """
        with self._lock:
            return self._buttons["left"]["handoff"]

    def seed_qpos_from_obs(self, obs: dict[str, float]) -> None:
        """Reset internal qpos + gripper + engagement state to match a target
        pose dict. ``obs`` is any ``.pos``-keyed dict — either a robot
        observation (follower-measured pose) or a *commanded* action
        (what a policy/operator last intended; for a human-in-the-loop
        handoff, seeding from the command keeps a gripper that is
        squeezing an object closed instead of recording its blocked-open
        measurement). Call before handing control to the operator so:

        1. The IK anchors to the robot's *actual* pose (not the teleop's
           stale rest pose), and
        2. The gripper output matches what was last commanded (no jolt
           on the first emitted action), and
        3. The next ``_update_arm`` sees a clean grip rising-edge and
           re-anchors the mapper at the operator's *current* hand pose.

        Why (3) matters: if the operator released the grip while nothing
        was calling ``get_action`` (e.g. between interventions), the
        falling-edge handler in ``_update_arm`` never ran, so
        ``arm["last_grip"]`` stays ``True`` and the mapper would keep the
        *previous* intervention's engage anchor — the next intervention
        would drive the robot to "where your hand is now relative to
        where it was last time" instead of "no motion until you move".
        Force-disengaging here makes the next rising-edge detection
        unambiguous.

        Expects bimanual-prefixed keys (``left_joint_1.pos``,
        ``left_gripper.pos``, …); missing keys are silently skipped.
        """
        with self._lock:
            for hand in ("left", "right"):
                arm = self._arms[hand]
                for j in range(ARM_DOFS):
                    key = f"{hand}_joint_{j + 1}.pos"
                    if key in obs:
                        arm["qpos"][j] = float(obs[key])
                # Gripper: store as `trigger` (used by `_build_action`)
                # and also mirror into the sim-viewer qpos slots.
                gkey = f"{hand}_gripper.pos"
                if gkey in obs:
                    g = float(np.clip(obs[gkey], 0.0, 1.0))
                    arm["trigger"] = g
                    grip_qpos = GRIPPER_QPOS_OPEN + g * (GRIPPER_QPOS_CLOSED - GRIPPER_QPOS_OPEN)
                    arm["qpos"][6] = grip_qpos
                    arm["qpos"][7] = grip_qpos
                # Cancel any in-flight go-home rest ramp: it has no
                # ``engaged`` guard, so a ramp started before the handoff
                # would keep overwriting ``qpos[:6]`` and clobber the seed
                # we just wrote. We do NOT touch ``last_rest_button``: the
                # ramp only re-arms on a rising edge of that button, so
                # leaving it avoids spuriously restarting a ramp if the
                # operator is still holding the thumbstick.
                arm["ramp_active"] = False
                # Force-disengage: see docstring above. Equivalent to a
                # clean release at the current pose, so the next tick's
                # _update_arm sees grip rising-edge → re-anchors.
                arm["mapper"].disengage()
                arm["engaged"] = False
                arm["last_grip"] = False
                arm["pos_filt"] = None
                arm["quat_filt"] = None
                arm["needs_reanchor"] = False

    def get_action(self) -> dict[str, float]:
        # Track actual call rate (EMA over inter-call deltas). The web UI
        # reads this from ik_state so the joint-Δq-cap readout shows the
        # *real* rad/s, not a hardcoded assumption — the example loop
        # defaults to 200 Hz, `lerobot-record` runs at the dataset FPS.
        # Whatever the caller's loop runs at, this is the number that
        # matters.
        now = time.perf_counter()
        last = self._last_get_action_t
        self._last_get_action_t = now
        if last is not None:
            dt = max(now - last, 1e-3)
            inst_hz = 1.0 / dt
            # 0.1 EMA ≈ 1 s time-constant at 10 Hz, 200 ms at 50 Hz, 50 ms
            # at 200 Hz — fast enough that the value tracks startup but slow
            # enough that one slow tick doesn't make the readout flicker.
            self._loop_hz = 0.1 * inst_hz + 0.9 * (self._loop_hz or inst_hz)

        with self._lock:
            xr = self._latest_xr_frame
            last_frame_time = self._last_xr_frame_time

        # No xr_frame yet — caller gets the home pose.
        if xr is None:
            return self._build_action()

        gap_s = time.time() - last_frame_time
        viewer_orient = (xr.get("viewer") or {}).get("orientation")
        yaw_now = _yaw_from_quat_xyzw(viewer_orient) if viewer_orient is not None else None

        ctrls = xr.get("controllers") or {}
        for hand in ("left", "right"):
            self._update_arm(hand, ctrls.get(hand), yaw_now, gap_s)

        action = self._build_action()

        if self.config.publish_ik_state and self._ws is not None and self._ws_loop is not None:
            self._publish_ik_state_async()

        return action

    # ---------- internals ----------

    def _build_action(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for hand in ("left", "right"):
            arm = self._arms[hand]
            for j in range(ARM_DOFS):
                out[f"{hand}_joint_{j + 1}.pos"] = float(arm["qpos"][j])
            # LeRobot follower convention: gripper.pos in [0, 1] (0=open, 1=closed).
            # Our trigger is already 0..1 with the same polarity.
            out[f"{hand}_gripper.pos"] = float(np.clip(arm["trigger"], 0.0, 1.0))
        return out

    def _anchor_mapper(
        self, hand: str, arm: dict, pos: np.ndarray, quat_wxyz: np.ndarray, yaw_now: float | None, label: str
    ) -> None:
        """Capture the mapper's engage frame at the current robot+controller
        state. Used by every re-anchor path — rising-edge engage,
        stale-recovery, precision-scale toggle, rest-ramp completion —
        same math, different log label."""
        ee_pos, ee_quat = arm["solver"].fk(arm["qpos"])
        j4_pos = arm["solver"].j4_anchor_xpos()
        if yaw_now is not None:
            R_engage = self._r_calib @ _R_y(-yaw_now)
            arm["mapper"].set_R(R_engage)
        arm["mapper"].engage(pos, quat_wxyz, ee_pos, ee_quat, pivot_armbase=j4_pos)
        arm["engaged"] = True
        logger.debug(
            "%s clutch %s  ee_pos=%s  j4_pos=%s  yaw_now=%s",
            hand,
            label,
            ee_pos.round(3),
            j4_pos.round(3) if j4_pos is not None else "—",
            f"{np.degrees(yaw_now):.1f}°" if yaw_now is not None else "—",
        )

    def _update_arm(self, hand: str, ctrl: dict | None, yaw_now: float | None, gap_s: float) -> None:
        if ctrl is None:
            return
        arm = self._arms[hand]

        # Staleness gate. If the WS hasn't delivered an xr_frame recently,
        # the controller pose in `ctrl` is from before the gap. Skip the
        # whole tick (no engage/disengage, no IK). Flag that we need to
        # re-anchor when fresh data resumes, so the engage-delta starts
        # at zero again and there is no catch-up motion.
        if gap_s > XR_FRAME_STALE_TIMEOUT_S:
            if arm["engaged"] and not arm["needs_reanchor"]:
                arm["needs_reanchor"] = True
                logger.warning("%s xr_frame stale (%.2fs gap) — pausing", hand, gap_s)
            arm["haptic"] *= 0.6  # decay so vibration doesn't linger
            return

        pos_raw = np.asarray(ctrl["position"], dtype=float)
        ox, oy, oz, ow = ctrl["orientation"]  # WebXR sends xyzw
        quat_raw = np.array([ow, ox, oy, oz])

        # EMA smoothing on the controller pose. First tick after init /
        # disengage / stale: copy raw so we don't start lerping toward an
        # uninitialised value. nlerp on the quaternion with a hemisphere
        # check (q and −q represent the same rotation; without the check
        # the lerp can drift to the antipodal point and produce a flip).
        alpha = float(self.config.pose_filter_alpha)
        if arm["pos_filt"] is None or arm["quat_filt"] is None:
            arm["pos_filt"] = pos_raw.copy()
            arm["quat_filt"] = quat_raw.copy()
        else:
            arm["pos_filt"] = (1.0 - alpha) * arm["pos_filt"] + alpha * pos_raw
            q_in = quat_raw if np.dot(arm["quat_filt"], quat_raw) >= 0.0 else -quat_raw
            qf = (1.0 - alpha) * arm["quat_filt"] + alpha * q_in
            arm["quat_filt"] = qf / np.linalg.norm(qf)
        pos = arm["pos_filt"]
        quat_wxyz = arm["quat_filt"]

        buttons = ctrl.get("buttons") or []
        grip = bool(buttons[GRIP_BUTTON_INDEX]["p"]) if len(buttons) > GRIP_BUTTON_INDEX else False
        trigger = float(buttons[TRIGGER_BUTTON_INDEX]["v"]) if len(buttons) > TRIGGER_BUTTON_INDEX else 0.0
        precision = (
            bool(buttons[PRECISION_BUTTON_INDEX]["p"]) if len(buttons) > PRECISION_BUTTON_INDEX else False
        )
        # Per-arm "go home" ramp: thumbstick-click rising edge starts a
        # time-based linear ramp from current qpos to `rest_qpos_{hand}`.
        # Ignored if a ramp is already running on this arm. The interpolation
        # itself runs further down so the gripper / engage state stay current.
        rest_btn = (
            bool(buttons[REST_RAMP_BUTTON_INDEX]["p"]) if len(buttons) > REST_RAMP_BUTTON_INDEX else False
        )
        if rest_btn and not arm["last_rest_button"] and not arm["ramp_active"]:
            target = self.config.rest_qpos_left if hand == "left" else self.config.rest_qpos_right
            arm["ramp_start_q"] = arm["qpos"][:ARM_DOFS].copy()
            arm["ramp_target_q"] = np.asarray(target, dtype=float)
            arm["ramp_start_t"] = time.perf_counter()
            arm["ramp_active"] = True
            logger.debug(
                "%s rest-ramp START (duration=%.2fs, target=%s, engaged=%s)",
                hand,
                self.config.rest_ramp_duration_s,
                arm["ramp_target_q"].round(3),
                arm["engaged"],
            )
        arm["last_rest_button"] = rest_btn
        # NB: `arm["trigger"]` and the gripper qpos are intentionally NOT
        # updated unconditionally — that lives in the engaged-only block
        # below so the gripper stays frozen at its last value while the
        # operator isn't holding the clutch (otherwise pulling the trigger
        # would still close the gripper between or before corrections).
        # While the precision button is held, scale down the mapper's
        # translation and rotation gains for finer EE positioning. On a
        # press/release transition, re-anchor the engage frame at the
        # current pose first: on the legacy absolute path the accumulated
        # delta would otherwise be reinterpreted under the new scale (a
        # target snap); on the reach-limited incremental path the re-anchor
        # just realigns hand↔EE correspondence.
        if precision != arm["last_precision"] and arm["engaged"]:
            self._anchor_mapper(
                hand, arm, pos, quat_wxyz, yaw_now, "PRECISION" if precision else "FULL-SCALE"
            )
        arm["last_precision"] = precision
        scale_factor = self.config.precision_factor if precision else 1.0
        arm["mapper"].scale = self.config.scale_translation * scale_factor
        arm["mapper"].scale_rotation = self.config.scale_rotation * scale_factor
        arm["mapper"].rot_reach_limit = self.config.rot_reach_limit
        arm["mapper"].pos_reach_limit = self.config.pos_reach_limit

        # Recovery: if we were engaged through a stale window and the operator
        # is still holding the grip, silently re-anchor the engage frame at
        # the fresh pose. Skipped if they released during the stall — that's
        # handled by the normal edge_disengage path below.
        if arm["needs_reanchor"] and arm["engaged"] and grip:
            self._anchor_mapper(hand, arm, pos, quat_wxyz, yaw_now, "RE-ANCHOR")
        arm["needs_reanchor"] = False

        # Edge-detect clutch.
        if grip and not arm["last_grip"]:
            self._anchor_mapper(hand, arm, pos, quat_wxyz, yaw_now, "ENGAGE")
        elif not grip and arm["last_grip"]:
            arm["mapper"].disengage()
            arm["engaged"] = False
            # Drop the pose filter on release so the next engage captures
            # the operator's fresh pose without lerp-lag from old state.
            arm["pos_filt"] = None
            arm["quat_filt"] = None
            logger.debug("%s clutch RELEASE", hand)
        arm["last_grip"] = grip

        # Gripper tracks the controller trigger ONLY while engaged. When
        # disengaged, `arm["trigger"]` and the sim-mirror qpos[6:8] hold
        # their last values so the gripper doesn't open/close from stray
        # trigger pressure outside an active intervention.
        if arm["engaged"]:
            arm["trigger"] = trigger
            grip_qpos = GRIPPER_QPOS_OPEN + trigger * (GRIPPER_QPOS_CLOSED - GRIPPER_QPOS_OPEN)
            arm["qpos"][6] = grip_qpos
            arm["qpos"][7] = grip_qpos

        # Per-arm "go home" ramp owns qpos while active: linearly interpolate
        # qpos[:6] from start → rest target over rest_ramp_duration_s, then
        # re-anchor (if engaged) so the operator's next motion starts with
        # zero delta. IK is skipped entirely during the ramp so the mapper
        # delta doesn't fight the interpolation. Gripper (qpos[6:8]) still
        # tracks the trigger above when engaged.
        if arm["ramp_active"]:
            duration = max(1e-3, float(self.config.rest_ramp_duration_s))
            elapsed = time.perf_counter() - arm["ramp_start_t"]
            t = min(1.0, elapsed / duration)
            arm["qpos"][:ARM_DOFS] = arm["ramp_start_q"] + t * (arm["ramp_target_q"] - arm["ramp_start_q"])
            if t >= 1.0:
                arm["ramp_active"] = False
                if arm["engaged"]:
                    # Capture the controller pose + new (rest) EE as the
                    # engage frame. mapper.target() will now return ≈ rest
                    # until the hand moves — no jump on the first IK tick.
                    self._anchor_mapper(hand, arm, pos, quat_wxyz, yaw_now, "RAMP-DONE")
                else:
                    logger.debug("%s rest-ramp DONE (disengaged)", hand)
            arm["haptic"] *= 0.6  # decay haptic during the ramp
            return

        # If engaged, drive arm IK toward the controller-relative target.
        # The current EE pose enables the mapper's reach limits (FK at the
        # last commanded qpos; solve() re-runs FK at the seed anyway).
        ee_pos_now, ee_quat_now = arm["solver"].fk(arm["qpos"])
        out = arm["mapper"].target(pos, quat_wxyz, ee_pos_now, ee_quat_now)
        if out is not None:
            tgt_pos, tgt_quat = out
            arm["qpos"][:ARM_DOFS] = arm["solver"].solve(tgt_pos, tgt_quat, arm["qpos"])

            # Haptic feedback: take the max of four signals, all 0..1.
            #  (a) limit_pressure (rad): joints clipped into their stops
            #      this tick (0.05..0.30); also floored at 0.35 by the
            #      solver when the antipode gate parks the wrist, so a
            #      park saturates this signal.
            #  (b) pos_err_norm (m): workspace-boundary reach, proxied by
            #      residual position-task error. 0.03..0.13.
            #  (c) singularity_proximity (0..1): joints-1-3 adaptive-damping
            #      ramp. Heavy dead-zone — only kicks in when ramp ≥ 0.95,
            #      so it warns only on the actual singular configuration,
            #      not the wider damping ramp region.
            #  (d) wrist_gimbal_proximity (0..1): wrist damping ramp, gated
            #      from 0.5 so the buzz starts around ~80° of wrist pitch
            #      and grows into the gimbal. Advance warning: near gimbal
            #      the damped wrist goes sluggish instead of pressing into
            #      limits, so without this there'd be no cue at all.
            pressure = float(getattr(arm["solver"], "last_limit_pressure", 0.0))
            pos_err = float(getattr(arm["solver"], "last_pos_err_norm", 0.0))
            singular = float(getattr(arm["solver"], "last_singularity_proximity", 0.0))
            gimbal = float(getattr(arm["solver"], "last_wrist_gimbal_proximity", 0.0))
            i_limit = min(1.0, max(0.0, (pressure - 0.05) / 0.25))
            i_reach = min(1.0, max(0.0, (pos_err - 0.03) / 0.10))
            i_singular = max(0.0, (singular - 0.95) / 0.2)
            i_gimbal = min(1.0, max(0.0, (gimbal - 0.5) / 0.5))
            raw_intensity = max(i_limit, i_reach, i_singular, i_gimbal)
            arm["haptic"] = 0.6 * arm["haptic"] + 0.4 * raw_intensity
        else:
            # Disengaged → decay the haptic to zero so vibration doesn't
            # linger after the operator releases the grip.
            arm["haptic"] *= 0.6

    def _publish_ik_state_async(self) -> None:
        """Schedule an ik_state send on the WS asyncio loop. Called from the
        LeRobot thread, executed on the WS thread."""
        payload = {
            "type": "ik_state",
            "left_qpos": [float(v) for v in self._arms["left"]["qpos"][:NQ]],
            "right_qpos": [float(v) for v in self._arms["right"]["qpos"][:NQ]],
            "left_engaged": bool(self._arms["left"]["engaged"]),
            "right_engaged": bool(self._arms["right"]["engaged"]),
            # Haptic intensity (0..1) per arm — the EMA'd max of the IK
            # trouble signals (limit pressure, reach error, singularity /
            # gimbal proximity). Client uses this to vibrate the matching
            # controller continuously.
            "left_haptic": float(self._arms["left"]["haptic"]),
            "right_haptic": float(self._arms["right"]["haptic"]),
            # Force haptic (0..1) per arm, populated by `send_feedback` from
            # the gripper torque reading. Mixed with `*_haptic` on the
            # client. None means "no orchestrator wrote this yet" — the
            # client should fall back to ignoring it.
            "left_force_haptic": float(self._arms["left"]["force_haptic"]),
            "right_force_haptic": float(self._arms["right"]["force_haptic"]),
            # Compatibility shim for the existing viewer_client.py/Quest UI
            # which read `qpos` (single arm) — surface the right arm here so
            # things keep rendering until those clients learn the new schema.
            "qpos": [float(v) for v in self._arms["right"]["qpos"][:NQ]],
            "engaged": bool(self._arms["right"]["engaged"]),
            # Which teleop instance produced this state. Lets passive
            # listeners (viewer_client --from-id) pick one stream when
            # several teleops run against the same relay, e.g. a
            # with/without-limit side-by-side comparison.
            "teleop_id": str(self.config.id) if self.config.id is not None else None,
            "server_time": time.time(),
            # Measured tick rate of get_action(). 0.0 until the second call
            # (need a delta) — the UI treats <1 Hz as "unknown" and falls
            # back to its assumed rate.
            "loop_hz": float(self._loop_hz or 0.0),
        }
        text = json.dumps(payload)

        async def _send():
            try:
                if self._ws is not None:
                    await self._ws.send(text)
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(_send(), self._ws_loop)
        except Exception:
            pass

    # ---------- Live config (from the web UI) ----------

    # Web-tunable knobs and their hard safety bounds. Anything outside the
    # range is clamped into it with a warning so a runaway slider can't
    # drive the arm into unsafe gains.
    _LIVE_CONFIG_BOUNDS: dict[str, tuple[float, float]] = {
        "scale_translation": (0.1, 10.0),
        "scale_rotation": (0.1, 10.0),
        "pose_filter_alpha": (0.05, 1.0),
        "precision_factor": (0.05, 1.0),
        # Per-arm gripper-torque deadband (Nm). Lower than the worst static
        # holding torque and the operator gets idle-buzz; higher than the
        # contact-force range and they never feel grasps. Upper bound matches
        # the DK1 follower's `max_gripper_torque` ceiling so a runaway slider
        # can't silence real contact entirely.
        "force_haptic_threshold_nm_left": (0.0, 1.0),
        "force_haptic_threshold_nm_right": (0.0, 1.0),
        # Per-group joint Δq cap (rad/tick). Two scalars, one for position
        # joints (1-3) and one for rotation joints (4-6). Upper bound is
        # generous (0.50 ≈ 25 rad/s @ 50 Hz, 100 rad/s @ 200 Hz) so the
        # operator has plenty of headroom regardless of FPS; lower bound
        # is a sensible "don't disable accidentally". Side-effect below
        # rebuilds the 6-vector and pushes it into both DecoupledIKSolvers.
        "max_dq_per_joint_scalar_pos": (0.005, 0.50),
        "max_dq_per_joint_scalar_rot": (0.005, 0.50),
    }

    # Bool config fields tunable from the web UI. Same mechanism as
    # `_LIVE_CONFIG_BOUNDS` but for on/off toggles; the web sends `true` /
    # `false` (or `1` / `0`) and the field is set directly.
    _LIVE_CONFIG_BOOLS: tuple[str, ...] = ("force_haptic_enabled",)

    def _apply_config_update(self, cfg: dict) -> None:
        """Apply a `config_update` payload from the web UI. Mutates
        `self.config` in place under the WS lock so the per-tick read picks
        up the new value on the next solve. Per-tick code in `_update_arm`
        already re-reads `self.config.scale_translation` etc. each call, so
        no extra plumbing is needed."""
        if not isinstance(cfg, dict):
            return
        applied: dict[str, float | bool] = {}
        with self._lock:
            for key, (lo, hi) in self._LIVE_CONFIG_BOUNDS.items():
                if key not in cfg:
                    continue
                try:
                    v = float(cfg[key])
                except (TypeError, ValueError):
                    logger.warning("config_update: %s=%r not a number; ignoring", key, cfg[key])
                    continue
                if not (lo <= v <= hi):
                    logger.warning(
                        "config_update: %s=%.3f out of range [%.2f, %.2f]; clamping", key, v, lo, hi
                    )
                    v = max(lo, min(hi, v))
                setattr(self.config, key, v)
                applied[key] = v
            # Side-effect for either group Δq cap: rebuild the 6-vector
            # and push to both live DecoupledIKSolver instances so a slider
            # drag takes effect on the very next solve. Done once after
            # the per-key loop so a payload that updates both pos and
            # rot only writes the solver array once.
            if (
                "max_dq_per_joint_scalar_pos" in applied
                or "max_dq_per_joint_scalar_rot" in applied
            ):
                pos = float(self.config.max_dq_per_joint_scalar_pos)
                rot = float(self.config.max_dq_per_joint_scalar_rot)
                arr = [pos] * 3 + [rot] * 3
                self.config.max_dq_per_joint = arr
                for hand in ("left", "right"):
                    solver = self._arms[hand].get("solver")
                    if solver is not None:
                        solver.max_dq_per_joint = np.asarray(arr, dtype=float).copy()
            for key in self._LIVE_CONFIG_BOOLS:
                if key not in cfg:
                    continue
                v_bool = bool(cfg[key])
                setattr(self.config, key, v_bool)
                applied[key] = v_bool
        if applied:
            logger.info(
                "config_update applied: %s",
                ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in applied.items()),
            )

    # ---------- Haptic-threshold calibration (web UI button) ----------

    # Margin (Nm) added on top of the observed idle peak when writing the
    # new threshold. The peak we capture in a few-second window is *not* the
    # worst-case static torque the arm will see over a long session — slow
    # drift (thermal, hard-stop creep) and occasional torque-sense noise
    # push the long-tail higher. 0.20 Nm cushion empirically silences the
    # idle buzz across multi-minute sessions without eating into the real
    # contact-force range (which starts well above 0.5 Nm on this gripper).
    # Bumped up from 0.10 after operators reported intermittent buzz at the
    # tighter setting; the contact range starts high enough that 0.20 still
    # leaves plenty of room.
    _HAPTIC_CALIB_MARGIN_NM: float = 0.20
    # Hard ceiling on what calibration is allowed to auto-write — paranoia
    # against a faulty gripper (cable hung up, hard-stop slam) silencing the
    # haptic entirely. Matches the `_LIVE_CONFIG_BOUNDS` upper bound so the
    # operator can still nudge the slider all the way up by hand if a
    # specific arm needs it.
    _HAPTIC_CALIB_MAX_THRESHOLD_NM: float = 1.0
    # Minimum dynamic range above the highest threshold. After calibration
    # we bump `force_haptic_max_nm` to at least `max(threshold) + this` so
    # the (tau − threshold) / (max − threshold) deadband always has room
    # to ramp; without it, an arm calibrated at threshold ≈ ceiling would
    # collapse the span to ~0 and any real grasp would saturate to max
    # vibration — the exact failure mode this guards against.
    _HAPTIC_CALIB_MIN_SPAN_NM: float = 0.50
    # Peak above this hints at a hardware fault on the arm (jaws jammed,
    # cable hung up, sensor bias). Calibration logs a warning so the
    # operator knows to investigate rather than just trusting the result.
    _HAPTIC_CALIB_SUSPICIOUS_PEAK_NM: float = 0.50

    def _start_haptic_calibration(self, duration_s: float = 5.0) -> None:
        """Begin a per-arm idle-torque sampling window. Called from the WS
        reader thread when a `haptic_calibrate` message arrives. The
        feedback path (`_apply_torque_feedback`) tracks per-arm peak |τ|
        while the window is open; on close we compute new thresholds and
        broadcast `haptic_calibrate_result`."""
        duration_s = max(0.5, min(10.0, float(duration_s)))
        with self._lock:
            if self._haptic_calib is not None:
                logger.info("haptic_calibrate: already running; ignoring duplicate start")
                return
            self._haptic_calib = {
                "start_t": time.time(),
                "duration_s": duration_s,
                "peak": {"left": 0.0, "right": 0.0},
            }
        logger.info("haptic_calibrate: starting %.1fs idle-torque sampling", duration_s)

    def _finalize_haptic_calibration(self) -> None:
        """Close the calibration window: write per-arm thresholds + bump
        the global haptic ceiling so dynamic range stays meaningful, and
        emit a result message for the web UI. MUST be called with
        self._lock held (see the call site in `_apply_torque_feedback`)."""
        calib = self._haptic_calib
        if calib is None:
            return
        margin = self._HAPTIC_CALIB_MARGIN_NM
        threshold_cap = self._HAPTIC_CALIB_MAX_THRESHOLD_NM
        peaks = calib["peak"]
        new = {hand: min(threshold_cap, peaks[hand] + margin) for hand in ("left", "right")}
        self.config.force_haptic_threshold_nm_left = new["left"]
        self.config.force_haptic_threshold_nm_right = new["right"]

        # Keep the (max - threshold) span ≥ MIN_SPAN_NM so real grasps still
        # produce non-saturated intensity. Without this, an arm calibrated
        # at threshold ≈ default max=1.0 Nm collapses the span to ~0 → any
        # tau above threshold saturates → controller buzzes at max (the
        # bug operators saw in the wild). Never shrink the ceiling — only
        # raise it.
        required_max = max(new["left"], new["right"]) + self._HAPTIC_CALIB_MIN_SPAN_NM
        old_max = float(self.config.force_haptic_max_nm)
        new_max = max(old_max, required_max)
        if new_max != old_max:
            self.config.force_haptic_max_nm = new_max

        # Warn on suspiciously high idle peaks. Hardware can creep above the
        # normal idle range (jammed jaws, sensor bias, cable hung on the
        # frame) — we still write the threshold so the operator isn't stuck
        # with constant buzz, but flag it so they investigate rather than
        # treat the result as ground truth.
        suspicious = []
        for hand in ("left", "right"):
            if peaks[hand] > self._HAPTIC_CALIB_SUSPICIOUS_PEAK_NM:
                suspicious.append(f"{hand} peak={peaks[hand]:.3f} Nm")
        if suspicious:
            logger.warning(
                "haptic_calibrate: suspicious idle peak(s) — %s. "
                "Expected <%.2f Nm at rest with empty jaws; check the gripper "
                "isn't jammed, the cable isn't fouled on the frame, and the "
                "command isn't holding closed against a stop.",
                ", ".join(suspicious), self._HAPTIC_CALIB_SUSPICIOUS_PEAK_NM,
            )

        self._haptic_calib = None
        logger.info(
            "haptic_calibrate done: peak L=%.3f R=%.3f → threshold L=%.3f R=%.3f Nm "
            "(margin=%.2f, force_haptic_max_nm=%.2f)",
            peaks["left"], peaks["right"], new["left"], new["right"], margin, new_max,
        )
        # Broadcast back to the web client so it can display the new values.
        # Fire-and-forget on the WS loop so we don't block under the lock.
        payload = {
            "type": "haptic_calibrate_result",
            "left_peak_nm": float(peaks["left"]),
            "right_peak_nm": float(peaks["right"]),
            "left_threshold_nm": float(new["left"]),
            "right_threshold_nm": float(new["right"]),
            "margin_nm": float(margin),
            "max_nm": float(new_max),
            "suspicious": suspicious,
        }
        text = json.dumps(payload)

        async def _send() -> None:
            try:
                if self._ws is not None:
                    await self._ws.send(text)
            except Exception:
                pass

        if self._ws_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(_send(), self._ws_loop)
            except Exception:
                pass

    # ---------- WS thread ----------

    def _ws_thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._ws_loop = loop
        try:
            loop.run_until_complete(self._ws_runner())
        finally:
            loop.close()
            self._ws_loop = None

    def _ssl_context_for(self, url: str):
        """Build an SSL context for `wss://` URLs. When connecting to a local
        relay (`localhost` / `127.0.0.1`) we skip validation: the teleop and
        relay are on the same host, so a TLS handshake adds no security but
        would otherwise reject our self-signed LAN cert. For remote `wss://`
        targets we use the default trust store (production / cloudflared)."""
        if not url.startswith("wss://"):
            return None
        import ssl

        ctx = ssl.create_default_context()
        if "://localhost" in url or "://127.0.0.1" in url:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def _ws_runner(self) -> None:
        backoff = 1.0
        ssl_ctx = self._ssl_context_for(self.config.ws_url)
        while not self._ws_stop.is_set():
            try:
                async with websockets.connect(self.config.ws_url, ssl=ssl_ctx) as ws:
                    self._ws = ws
                    self._ws_connected.set()
                    backoff = 1.0
                    # Ask the page to (re)broadcast its current slider snapshot
                    # so we don't start the session with stale dataclass
                    # defaults. The relay forwards this to all other clients;
                    # the web client responds with a `config_update`. No-op
                    # if no page is connected yet (the relay just drops it).
                    try:
                        await ws.send(json.dumps({"type": "request_settings"}))
                    except Exception as e:
                        logger.debug("request_settings send failed: %s", e)
                    async for raw in ws:
                        if self._ws_stop.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        mtype = msg.get("type")
                        if mtype == "xr_frame":
                            # Extract per-hand grip + B/Y state here so the
                            # DAgger listener has live values even while the
                            # IK pipeline (which lives in get_action) isn't
                            # running.
                            btn_snapshot = {
                                "left": {"grip": False, "handoff": False},
                                "right": {"grip": False, "handoff": False},
                            }
                            ctrls = msg.get("controllers") or {}
                            for hand in ("left", "right"):
                                ctrl = ctrls.get(hand) or {}
                                buttons = ctrl.get("buttons") or []
                                if len(buttons) > GRIP_BUTTON_INDEX:
                                    btn_snapshot[hand]["grip"] = bool(buttons[GRIP_BUTTON_INDEX].get("p"))
                                if len(buttons) > HANDOFF_BUTTON_INDEX:
                                    btn_snapshot[hand]["handoff"] = bool(
                                        buttons[HANDOFF_BUTTON_INDEX].get("p")
                                    )
                            with self._lock:
                                self._latest_xr_frame = msg
                                self._last_xr_frame_time = time.time()
                                self._buttons = btn_snapshot
                        elif mtype == "config_update":
                            # Live-tunable knobs from the web UI. Mutate the
                            # config dataclass in place under the same lock the
                            # tick reads; the per-tick path reads `self.config.*`
                            # every iteration so the new value is picked up on
                            # the very next solve.
                            self._apply_config_update(msg.get("config") or {})
                        elif mtype == "haptic_calibrate":
                            # Operator pressed "Calibrate" in the web UI.
                            # Start a fixed-duration idle-torque sampling
                            # window; the result is broadcast back as
                            # `haptic_calibrate_result` when it closes.
                            self._start_haptic_calibration(
                                float(msg.get("duration_s") or 5.0)
                            )
                        # else: ignore (our own echoed ik_state, pings, etc.)
            except Exception as e:
                logger.warning(
                    "BiQuestTeleoperator WS error (%s); reconnecting in %.1fs", type(e).__name__, backoff
                )
            finally:
                self._ws = None
                self._ws_connected.clear()
            if self._ws_stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 10.0)
