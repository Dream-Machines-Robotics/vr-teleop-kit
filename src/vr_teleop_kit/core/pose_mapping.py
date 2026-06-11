"""Clutch-relative pose mapping: WebXR controller pose -> EE target in arm base.

`ClutchPoseMapper` holds the "engage" state — the controller pose and the
EE pose captured at the moment the operator pressed grip. While the
clutch is held, the EE target is the engaged EE pose composed with an
effective controller delta, rotated from Quest world into the arm's
base frame. In the production path the caller passes the arm's current
EE pose into `target()`: the delta then accumulates per-tick increments
and is reach-limited to within `rot_reach_limit` / `pos_reach_limit` of the current pose,
with the excess absorbed (slipping clutch — see the class docstring).
Callers that pass no EE pose get the legacy absolute delta-since-engage
mapping.

On clutch release, the mapper disengages; further calls to `target()`
return None until the next engage. This lets the operator reposition
their hand without moving the robot.

Run sanity tests:
    python -m vr_teleop_kit.core.pose_mapping
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


# ------- small quaternion helpers (wxyz convention, matching mujoco) -------

def quat_mul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(qa, float), np.asarray(qb, float))
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_negQuat(out, np.asarray(q, float))
    return out


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.ascontiguousarray(R, float).ravel())
    return out


def quat_pow(q: np.ndarray, k: float) -> np.ndarray:
    """Raise a quaternion to a scalar power: keep the axis, scale the angle by k.
    `quat_pow(q, 1) = q`, `quat_pow(q, 0) = identity`, `quat_pow(q, 0.5) = sqrt(q)`.
    Applies `ClutchPoseMapper.scale_rotation`: on the reach-limited path to each
    per-tick increment (a rate gain), on the legacy path to the whole
    delta-since-engage."""
    w = float(q[0])
    v = np.asarray(q[1:], dtype=float)
    half_angle = float(np.arctan2(float(np.linalg.norm(v)), w))
    if half_angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / np.sin(half_angle)
    new_half = k * half_angle
    s = float(np.sin(new_half))
    return np.array([float(np.cos(new_half)), s * axis[0], s * axis[1], s * axis[2]])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis · angle, rad) of a wxyz quaternion, shortest way."""
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, np.asarray(q, float), 1.0)
    return out


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a rotation vector (axis · angle, rad)."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(v, float) / angle
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), s * axis[0], s * axis[1], s * axis[2]])


# ------- mapper -------

@dataclass
class ClutchPoseMapper:
    """One-handed clutch-relative controller→EE mapping.

    Parameters:
        R: 3x3 rotation matrix taking Quest world vectors to arm base
            vectors (i.e. v_armbase = R @ v_quest).
        scale: linear gain on translation (1.0 = 1:1 motion).
        scale_rotation: gain on rotation delta (1.0 = 1:1; 0.5 halves
            the angular response of the EE relative to the controller).
            Useful when working near a wrist singularity — small operator
            wrist twists otherwise propagate to large elbow corrections.
        rotation_pivot: optional 3-vector in arm-base frame. When set,
            controller rotation is interpreted as "rotate the EE about
            this pivot" rather than "rotate the EE in place." Captured
            per-engage by the caller (e.g., at engage time the teleop can
            sample the elbow body position via FK and pass it here).
            None = in-place rotation (legacy behavior).
        rot_reach_limit: max angle (rad) the orientation target may run ahead
            of the arm's CURRENT orientation. Requires the caller to pass
            `ee_quat_armbase` into `target()`. With the reach limit active, the
            orientation channel becomes INCREMENTAL: per-tick controller
            rotation increments accumulate into the effective delta, and
            anything past the reach limit is absorbed (slipping clutch). Two
            problems this kills, both seen on hardware: a demand pressed
            far past a joint stop / gimbal can never build up the ~180°
            error where the shortest-way direction flips (cap-speed
            shaking), and it can never wrap around and "snap" the wrist
            in from the other side (350° clockwise and 10° counter-
            clockwise are the same orientation — an absolute mapping must
            eventually agree with that; the incremental one never has
            to). Trade-offs: absorbed twist is gone, and with
            scale_rotation ≠ 1 the per-increment rate gain makes curved
            hand paths path-dependent — so hand↔EE orientation
            correspondence drifts within an engagement; re-clutching
            realigns. 0/None disables (legacy absolute mapping).
        pos_reach_limit: max distance (m) the position target may run ahead of
            the arm's CURRENT EE position. Requires `ee_pos_armbase` in
            `target()`. Same absorbing (incremental) semantics as the
            rotation reach limit — a mouse at the screen edge: overshoot is
            absorbed, so reversing the hand moves the target immediately
            instead of after retracing the overshoot. Bounds the
            position-error magnitude that drives the arm when reaching
            past the workspace boundary (unbounded error produced
            cap-speed bang-bang of joints 1-3 on hardware). Absorbed
            travel drifts hand↔EE correspondence until re-clutch.
            0/None disables.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0
    scale_rotation: float = 1.0
    rotation_pivot: np.ndarray | None = None
    rot_reach_limit: float | None = 0.6
    pos_reach_limit: float | None = 0.25

    def __post_init__(self) -> None:
        self._engaged: bool = False
        self._ctrl_engage_pos: np.ndarray | None = None
        self._ctrl_engage_quat: np.ndarray | None = None
        self._ee_engage_pos: np.ndarray | None = None
        self._ee_engage_quat: np.ndarray | None = None
        self._R_quat: np.ndarray = mat_to_quat(np.asarray(self.R, float))
        self._R_quat_conj: np.ndarray = quat_conj(self._R_quat)
        # incremental reach-limit state: previous-tick controller pose (quest
        # frame) and the accumulated effective deltas (arm-base frame).
        # Reset on every engage.
        self._ctrl_prev_quat: np.ndarray | None = None
        self._ctrl_prev_pos: np.ndarray | None = None
        self._d_quat_eff: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self._d_pos_eff: np.ndarray = np.zeros(3)

    @property
    def engaged(self) -> bool:
        return self._engaged

    def set_R(self, R: np.ndarray) -> None:
        """Replace the rotation used for delta mapping. Typically called per
        engage with a yaw-corrected R so the operator can turn their body
        between sessions and still have 'controller forward' mean 'robot
        forward'."""
        self.R = np.asarray(R, dtype=float).copy()
        self._R_quat = mat_to_quat(self.R)
        self._R_quat_conj = quat_conj(self._R_quat)

    def engage(
        self,
        controller_pos_quest: np.ndarray,
        controller_quat_quest: np.ndarray,
        ee_pos_armbase: np.ndarray,
        ee_quat_armbase: np.ndarray,
        pivot_armbase: np.ndarray | None = None,
    ) -> None:
        """Capture the engage frame. Call on the rising clutch edge.
        If `pivot_armbase` is supplied, rotation deltas during this
        engagement pivot around that point instead of around the EE."""
        self._ctrl_engage_pos = np.array(controller_pos_quest, float, copy=True)
        self._ctrl_engage_quat = np.array(controller_quat_quest, float, copy=True)
        self._ee_engage_pos = np.array(ee_pos_armbase, float, copy=True)
        self._ee_engage_quat = np.array(ee_quat_armbase, float, copy=True)
        self.rotation_pivot = (
            None if pivot_armbase is None else np.array(pivot_armbase, float, copy=True)
        )
        self._ctrl_prev_quat = self._ctrl_engage_quat.copy()
        self._ctrl_prev_pos = self._ctrl_engage_pos.copy()
        self._d_quat_eff = np.array([1.0, 0.0, 0.0, 0.0])
        self._d_pos_eff = np.zeros(3)
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False

    def target(
        self,
        controller_pos_quest: np.ndarray,
        controller_quat_quest: np.ndarray,
        ee_pos_armbase: np.ndarray | None = None,
        ee_quat_armbase: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute the current EE target in arm-base frame, or None if disengaged.

        `ee_pos_armbase` / `ee_quat_armbase` are the arm's CURRENT EE pose;
        passing them enables the pos/rot reach limits (see the class docstring).
        Without them the legacy absolute mapping applies unchanged.
        """
        if not self._engaged:
            return None
        assert self._ctrl_engage_pos is not None
        assert self._ctrl_engage_quat is not None
        assert self._ee_engage_pos is not None
        assert self._ee_engage_quat is not None

        # Position delta. Reach-limited path: accumulate per-tick increments
        # (vectors commute, so increments with per-tick scale compose to
        # exactly the absolute scaled delta until the reach limit absorbs).
        # Legacy path: absolute delta from the engage frame.
        p_now = np.asarray(controller_pos_quest, float)
        pos_limited = ee_pos_armbase is not None and bool(self.pos_reach_limit)
        if pos_limited:
            assert self._ctrl_prev_pos is not None
            self._d_pos_eff = self._d_pos_eff + self.R @ (self.scale * (p_now - self._ctrl_prev_pos))
            d_pos_arm = self._d_pos_eff
        else:
            d_pos_arm = self.R @ (self.scale * (p_now - self._ctrl_engage_pos))
        self._ctrl_prev_pos = p_now.copy()

        q_now = np.asarray(controller_quat_quest, float)
        rot_limited = ee_quat_armbase is not None and bool(self.rot_reach_limit)
        if rot_limited:
            # Incremental path: accumulate this tick's controller rotation
            # increment (a degree or two — never direction-ambiguous) into
            # the effective delta. At scale_rotation=1 the increments
            # telescope to exactly the absolute delta until the reach limit
            # clamps. At other scales the gain applies per increment (a
            # RATE gain, like mouse sensitivity): straight twists scale
            # exactly, but curved hand paths are path-dependent on SO(3),
            # so a closed hand loop can leave a residual (~23° for a 60°+
            # 60° loop at 1.5×). That is inherent to rate-scaled rotation —
            # the alternative, scaling the total delta, wraps at 360°/scale
            # of raw twist and reintroduces the come-around this path
            # exists to kill. Re-clutching realigns either way.
            assert self._ctrl_prev_quat is not None
            if float(np.dot(q_now, self._ctrl_prev_quat)) < 0.0:
                q_now = -q_now  # hemisphere-align: q and -q are the same rotation
            inc = quat_mul(q_now, quat_conj(self._ctrl_prev_quat))
            self._ctrl_prev_quat = q_now.copy()
            if self.scale_rotation != 1.0:
                inc = quat_pow(inc, self.scale_rotation)
            inc_arm = quat_mul(quat_mul(self._R_quat, inc), self._R_quat_conj)
            d_quat_arm = quat_mul(inc_arm, self._d_quat_eff)
            d_quat_arm /= np.linalg.norm(d_quat_arm)
            self._d_quat_eff = d_quat_arm
        else:
            # Legacy absolute path: rotation delta in Quest world
            # (world-frame composition: now * engage⁻¹), conjugated by
            # R_quat to express in arm base. Keep the incremental state
            # fresh anyway so a live reach limit toggle doesn't see stale prev.
            if self._ctrl_prev_quat is not None:
                if float(np.dot(q_now, self._ctrl_prev_quat)) < 0.0:
                    q_now = -q_now
                self._ctrl_prev_quat = q_now.copy()
            d_quat_quest = quat_mul(q_now, quat_conj(self._ctrl_engage_quat))
            if self.scale_rotation != 1.0:
                d_quat_quest = quat_pow(d_quat_quest, self.scale_rotation)
            d_quat_arm = quat_mul(quat_mul(self._R_quat, d_quat_quest), self._R_quat_conj)

        target_quat = quat_mul(d_quat_arm, self._ee_engage_quat)

        if rot_limited:
            # Rotation reach limit: clamp the target to within rot_reach_limit of the
            # arm's current orientation and ABSORB the excess into the
            # effective delta (slipping clutch — absorbed twist does not
            # come back when the operator reverses).
            e = quat_to_rotvec(quat_mul(target_quat, quat_conj(np.asarray(ee_quat_armbase, float))))
            e_norm = float(np.linalg.norm(e))
            if e_norm > self.rot_reach_limit:
                e *= self.rot_reach_limit / e_norm
                target_quat = quat_mul(rotvec_to_quat(e), np.asarray(ee_quat_armbase, float))
                self._d_quat_eff = quat_mul(target_quat, quat_conj(self._ee_engage_quat))
                d_quat_arm = self._d_quat_eff

        # If a pivot is set, the rotation also moves the EE in an arc
        # around it — operator's wrist twist becomes "swing around pivot."
        # When rotation_pivot is None, the offset term vanishes and we're
        # back to legacy in-place rotation.
        if self.rotation_pivot is not None:
            offset = self._ee_engage_pos - self.rotation_pivot
            rotated_offset = np.zeros(3)
            mujoco.mju_rotVecQuat(rotated_offset, offset, d_quat_arm)
            target_pos = self.rotation_pivot + rotated_offset + d_pos_arm
        else:
            target_pos = self._ee_engage_pos + d_pos_arm

        # Position reach limit: clamp toward the current EE position and ABSORB
        # the excess into the effective delta (mouse-at-screen-edge
        # semantics: overshoot is gone, reversal moves the target
        # immediately instead of after retracing the overshoot).
        if pos_limited:
            ee_p = np.asarray(ee_pos_armbase, float)
            dp = target_pos - ee_p
            dp_norm = float(np.linalg.norm(dp))
            if dp_norm > self.pos_reach_limit:
                clamped = ee_p + dp * (self.pos_reach_limit / dp_norm)
                self._d_pos_eff = self._d_pos_eff + (clamped - target_pos)
                target_pos = clamped

        return target_pos, target_quat


# ------- sanity tests -------

def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.allclose(a, b, atol=tol))


def main() -> None:
    print("=== ClutchPoseMapper sanity tests ===\n")

    ee_engage_pos = np.array([0.50, 0.00, 0.42])
    ee_engage_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity orientation
    ctrl_engage_pos = np.array([0.0, 1.4, -0.3])
    ctrl_engage_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # T1: identity R, no controller motion → target == engage
    m = ClutchPoseMapper()
    m.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    assert m.engaged
    out = m.target(ctrl_engage_pos, ctrl_engage_quat)
    assert out is not None
    p, q = out
    print(f"T1 (no motion):                target=({p.round(4)}, {q.round(4)})  "
          f"[{'ok' if _close(p, ee_engage_pos) and _close(q, ee_engage_quat) else 'FAIL'}]")

    # T2: identity R, controller moves +5 cm in X → EE target += (0.05, 0, 0)
    p, q = m.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = ee_engage_pos + np.array([0.05, 0, 0])
    print(f"T2 (controller +5cm X, R=I):   target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) and _close(q, ee_engage_quat) else 'FAIL'}]")

    # T3: identity R, scale 0.5 → 5cm controller motion → 2.5cm EE motion
    m_scaled = ClutchPoseMapper(scale=0.5)
    m_scaled.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    p, _ = m_scaled.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = ee_engage_pos + np.array([0.025, 0, 0])
    print(f"T3 (scale 0.5):                target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) else 'FAIL'}]")

    # T4: non-trivial R (90° about Z swaps X<->Y, with sign): Quest +X → arm base +Y
    R_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # R_z90 @ (1,0,0) = (0,1,0)  → quest +X maps to arm +Y ✓
    m_R = ClutchPoseMapper(R=R_z90)
    m_R.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    p, _ = m_R.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = ee_engage_pos + np.array([0.0, 0.05, 0.0])
    print(f"T4 (R = Rz90, controller +X):  target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) else 'FAIL'}]")

    # T5: rotation. Identity R, controller rotates 30° about Quest +Y.
    # Should produce target_quat = R_y30 (in arm base, since R=I).
    m = ClutchPoseMapper()
    m.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    ctrl_now_quat = rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0]))
    _, q = m.target(ctrl_engage_pos, ctrl_now_quat)
    print(f"T5 (controller rotates +30° Y, R=I): target_quat={q.round(4)}  "
          f"expected≈{rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0])).round(4)}  "
          f"[{'ok' if _close(q, rotvec_to_quat(np.array([0.0, np.pi / 6, 0.0]))) else 'FAIL'}]")

    # T6: disengage → target returns None
    m.disengage()
    out = m.target(ctrl_engage_pos, ctrl_engage_quat)
    print(f"T6 (after disengage):          target={out}  [{'ok' if out is None else 'FAIL'}]")

    # T7: re-engage at a different EE pose — engage origin must update
    new_ee_pos = np.array([0.55, 0.10, 0.42])
    m.engage(ctrl_engage_pos, ctrl_engage_quat, new_ee_pos, ee_engage_quat)
    p, _ = m.target(ctrl_engage_pos + np.array([0.05, 0, 0]), ctrl_engage_quat)
    expected = new_ee_pos + np.array([0.05, 0, 0])
    print(f"T7 (re-engage at new EE):      target_pos={p.round(4)}  expected={expected.round(4)}  "
          f"[{'ok' if _close(p, expected) else 'FAIL'}]")

    def _ang_deg(qa, qb):
        return float(np.degrees(np.linalg.norm(quat_to_rotvec(quat_mul(qa, quat_conj(qb))))))

    # T8: rotation reach limit — twist the controller 120° in 1° increments while
    # the arm (ee pose) stays put; the target must never run further than
    # rot_reach_limit from the current EE orientation.
    m8 = ClutchPoseMapper(rot_reach_limit=0.5, pos_reach_limit=0.25)
    m8.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    worst = 0.0
    q_ctrl = ctrl_engage_quat
    for i in range(1, 121):
        q_ctrl = rotvec_to_quat(np.array([0.0, np.radians(i), 0.0]))
        _, tq = m8.target(ctrl_engage_pos, q_ctrl, ee_engage_pos, ee_engage_quat)
        worst = max(worst, _ang_deg(tq, ee_engage_quat))
    ok = worst <= np.degrees(0.5) + 0.1
    print(f"T8 (rot reach limit, 120° push):     max target-vs-EE angle={worst:.1f}°  "
          f"(reach limit {np.degrees(0.5):.1f}°)  [{'ok' if ok else 'FAIL'}]")

    # T9: slipping clutch — after the 120° push, reversing by just the reach limit
    # angle brings the target back onto the EE orientation (immediate bite;
    # the absorbed 120°-28.6° never has to be retraced).
    back = 120.0 - np.degrees(0.5)
    _, tq = m8.target(ctrl_engage_pos,
                      rotvec_to_quat(np.array([0.0, np.radians(back), 0.0])),
                      ee_engage_pos, ee_engage_quat)
    resid = _ang_deg(tq, ee_engage_quat)
    print(f"T9 (reversal bites at reach limit):  residual after backing off {np.degrees(0.5):.1f}°: "
          f"{resid:.2f}°  [{'ok' if resid < 1.0 else 'FAIL'}]")

    # T10: position reach limit — mouse-at-screen-edge semantics. A 1 m push
    # clamps to pos_reach_limit from the current EE; the overshoot is absorbed,
    # so a 5 cm reversal moves the target back 5 cm IMMEDIATELY (no
    # retracing the 0.75 m overshoot).
    m10 = ClutchPoseMapper(rot_reach_limit=1.0, pos_reach_limit=0.25)
    m10.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    p, _ = m10.target(ctrl_engage_pos + np.array([1.0, 0, 0]), ctrl_engage_quat,
                      ee_engage_pos, ee_engage_quat)
    ok1 = _close(p, ee_engage_pos + np.array([0.25, 0, 0]))
    p, _ = m10.target(ctrl_engage_pos + np.array([0.95, 0, 0]), ctrl_engage_quat,
                      ee_engage_pos, ee_engage_quat)
    ok2 = _close(p, ee_engage_pos + np.array([0.20, 0, 0]))
    print(f"T10 (pos reach limit, mouse-style):  clamp@1m={'ok' if ok1 else 'FAIL'}  "
          f"reversal-bites={'ok' if ok2 else 'FAIL'}")

    # T11: within the reach limit and with the EE tracking the target, the
    # incremental path reproduces the legacy absolute mapping. Holds
    # exactly only at scale_rotation=1: at other gains the per-increment
    # scaling is a rate gain, path-dependent on curved hand paths (see
    # the comment in target()).
    m_inc = ClutchPoseMapper(rot_reach_limit=3.0, pos_reach_limit=10.0)
    m_abs = ClutchPoseMapper()
    for mm in (m_inc, m_abs):
        mm.engage(ctrl_engage_pos, ctrl_engage_quat, ee_engage_pos, ee_engage_quat)
    ee_q = ee_engage_quat
    worst = 0.0
    for i in range(1, 41):
        q_ctrl = rotvec_to_quat(np.radians(i) * np.array([0.5, 0.7, 0.2]))
        _, tq_i = m_inc.target(ctrl_engage_pos, q_ctrl, ee_engage_pos, ee_q)
        _, tq_a = m_abs.target(ctrl_engage_pos, q_ctrl)
        worst = max(worst, _ang_deg(tq_i, tq_a))
        ee_q = tq_i  # EE follows perfectly
    print(f"T11 (incremental == absolute within reach limit): max diff={worst:.4f}°  "
          f"[{'ok' if worst < 0.01 else 'FAIL'}]")


if __name__ == "__main__":
    main()
