"""Decoupled IK for the TRLC-DK1 arm — strict joint decoupling.

Joints 1-3 satisfy position; joints 4-6 satisfy orientation. Both
sub-problems are one damped-least-squares step per call, warm-started
from the caller's qpos. The 6.2 cm wrist non-sphericity becomes a
residual EE position error when the wrist rotates — that's intentional.
The operator's visual feedback loop closes the gap.

Always returns a valid qpos6 (never None). Four boundary cases are
handled in-line so the arm degrades gracefully instead of freezing:

  1. Near workspace boundary — position Jacobian becomes ill-conditioned.
     Manipulability-adaptive damping; step magnitude smoothly shrinks
     to zero at the singularity.
  2. Near wrist gimbal lock (θ5 → ±π/2, joint-4/6 axes aligned) — the
     orientation Jacobian loses rank. Same adaptive-damping recipe on
     the wrist sub-solve: J4/J6 steps stay bounded and the lost rotation
     direction is simply not tracked until the operator backs off.
  3. Joint limit violation — elementwise clamp into URDF limits. Caller
     sees the arm reach as far as joints allow, with the residual EE
     error left for the operator's visual loop.
  4. Near-antipodal orientation demand (error angle past rot_err_hold) —
     the shortest-way error direction is unstable there, so the wrist
     parks and reports saturating limit pressure; tracking resumes once
     the operator backs off below the gate.

A `max_dq_per_joint` cap is applied at the end as a per-joint velocity
bound — the operator-safety layer against any residual fast motion.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

try:
    from .model import build_model_with_tool0_site
except ImportError:  # allow running this file directly as a script
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from vr_teleop_kit.ik.model import build_model_with_tool0_site


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


class DecoupledIKSolver:
    """Decoupled IK for TRLC-DK1.

    Holds a mujoco model with the tool0 site, plus cached site/body-id
    lookups. `solve()` is the main entry point.

    Tunables:
      lam_pos           DLS base damping on the 3-DoF position sub-solve.
                        Default 0.05.
      lam0              Extra damping ramp amplitude near the joints-1-3
                        singularity. Total λ² = lam_pos² + lam0² · ramp²
                        where ramp = max(0, 1 - w/w0) and
                        w = |det(J_pos_arm)|. Default 0.15.
      w0                Manipulability threshold where the ramp starts.
                        Default 0.05.
      mu                Tikhonov stiffness pulling joints 1-3 toward q_rest.
                        Default 0.02.
      lam_rot           DLS base damping on the 3-DoF wrist orientation
                        sub-solve. Default 0.05.
      lam0_rot          Extra damping ramp amplitude near wrist gimbal lock
                        (θ5 → ±π/2). Same λ² recipe as the arm. Default 0.4.
      w0_rot            Wrist-manipulability threshold where that ramp
                        starts; w = |det(J_rot)| ≈ |cos θ5|. Default 0.5
                        (ramp begins at θ5 ≈ 60° — early enough that the
                        damping has built up by the time the Jacobian's
                        weakest direction collapses around 70-80°).
      rot_err_hold      Park the wrist when the orientation error angle
                        exceeds this (rad). Near the antipode (180°) the
                        shortest-way error direction is unstable — it
                        flips sign under tiny target jitter — and chasing
                        it with capped steps turns into a bang-bang
                        oscillation at the Δq cap (operator-reported
                        shaking when twisting far past the θ5 stop;
                        reproduced in sim at ~150° error with 0.2° hand
                        tremor). Beyond the gate the wrist holds still
                        and reports limit pressure; tracking resumes as
                        soon as the operator backs off below it.
                        Default 2.2 (~126°).
      q_rest            Rest pose joints 1-3 the Tikhonov term pulls toward.
      max_dq_per_joint  Per-joint Δq cap, length 6. Bounds worst-case
                        single-joint snap regardless of source.
    """

    def __init__(
        self,
        urdf_path: Path | None = None,
        lam_pos: float = 0.05,
        lam0: float = 0.15,
        w0: float = 0.05,
        mu: float = 0.02,
        lam_rot: float = 0.05,
        lam0_rot: float = 0.4,
        w0_rot: float = 0.5,
        rot_err_hold: float = 2.2,
        q_rest: np.ndarray | None = None,
        max_dq_per_joint: list[float] | None = None,
    ) -> None:
        self.lam_pos = float(lam_pos)
        self.lam0 = float(lam0)
        self.w0 = float(w0)
        self.mu = float(mu)
        self.lam_rot = float(lam_rot)
        self.lam0_rot = float(lam0_rot)
        self.w0_rot = float(w0_rot)
        self.rot_err_hold = float(rot_err_hold)
        self.q_rest_123 = (
            np.array([0.0, np.pi / 2, np.pi / 2], dtype=float)
            if q_rest is None else np.asarray(q_rest, dtype=float)[:3].copy()
        )
        self.max_dq_per_joint = (
            None if max_dq_per_joint is None
            else np.asarray(max_dq_per_joint, dtype=float).reshape(6).copy()
        )

        self.model, self.data = build_model_with_tool0_site(urdf_path)
        # Joint limits straight from the compiled model (i.e. the URDF) —
        # nothing transcribed by hand.
        self.joint_limits = self.model.jnt_range[:6].copy()

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tool0"
        )
        # Position-task anchor — a named site on link3-4, 10 cm past joint
        # 4 along the link3→link4 direction. Fully wrist-invariant since
        # it lives upstream of joint 4. See ik/solver.py for the exact
        # site placement and the rationale.
        self.j4_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "j4_anchor"
        )
        if -1 in (self.site_id, self.j4_site_id):
            raise RuntimeError("DecoupledIKSolver: required site missing from model")

        # Set by each solve(). Signals fed into the controller-haptic
        # mix downstream:
        #   last_limit_pressure (rad)          — joint-limit clip this tick.
        #   last_pos_err_norm (m)              — workspace-boundary reach error.
        #   last_singularity_proximity (0..1)  — joints-1-3 damping ramp.
        #   last_wrist_gimbal_proximity (0..1) — wrist (gimbal) damping ramp.
        self.last_limit_pressure: float = 0.0
        self.last_pos_err_norm: float = 0.0
        self.last_singularity_proximity: float = 0.0
        self.last_wrist_gimbal_proximity: float = 0.0

    def _fk(self, qpos: np.ndarray) -> None:
        """Run kinematics + comPos (required for the site Jacobians in solve())."""
        self.data.qpos[: self.model.nq] = 0
        self.data.qpos[: len(qpos)] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # for jacobians

    def fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics. Returns (tool0 world pos, tool0 world quat wxyz)."""
        self._fk(qpos)
        pos = self.data.site_xpos[self.site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return pos, quat

    def j4_anchor_xpos(self) -> np.ndarray:
        """World position of the j4_anchor site at the most recent FK. This
        is the wrist-invariant point the position task targets; the pose
        mapper should use it as the rotation pivot so pure controller
        rotations leave joints 1-3 at rest."""
        return self.data.site_xpos[self.j4_site_id].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        qpos_seed: np.ndarray,
    ) -> np.ndarray:
        """One step of decoupled IK.

        Always returns a valid 6-vector. Four boundary cases (workspace
        edge, joint limits, wrist gimbal lock, near-antipodal demand)
        are handled in-line — the arm degrades gracefully instead of
        freezing. See module docstring.
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        R_target = _quat_wxyz_to_R(target_quat_wxyz)

        # ----- Step 1: FK at the seed; read the *current* (tool0 → j4)
        # vector in tool0's local frame. Re-deriving this every tick (vs
        # caching at rest-wrist) keeps the position-task math
        # self-consistent with the wrist configuration we're actually at,
        # so a re-anchor (engage / scale change / stale recovery) with
        # target == current_ee produces pos_err = 0 exactly. -----
        self._fk(qpos_seed)
        current_tool0 = self.data.site_xpos[self.site_id].copy()
        current_R_tool0 = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        current_J4 = self.data.site_xpos[self.j4_site_id].copy()
        ee_to_j4_in_tool0 = current_R_tool0.T @ (current_J4 - current_tool0)

        # ----- Step 2: target J4 anchor in arm-base; pos_err for Newton step -----
        target_J4 = target_pos + R_target @ ee_to_j4_in_tool0
        pos_err = target_J4 - current_J4

        self.last_pos_err_norm = float(np.linalg.norm(pos_err))

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.j4_site_id)
        J_pos_arm = jacp[:, :3]                  # 3x3, joints 1-3 only

        # Manipulability-adaptive damping. w = |det(J_pos_arm)| measures
        # how far joints 1-3 are from a singular configuration; damping
        # climbs smoothly as w → 0, bounding joint velocities near the
        # shoulder singularity without affecting tracking elsewhere.
        w = abs(float(np.linalg.det(J_pos_arm)))
        ramp = max(0.0, 1.0 - w / max(self.w0, 1e-12))
        self.last_singularity_proximity = float(ramp)
        lam2 = self.lam_pos ** 2 + (self.lam0 ** 2) * (ramp ** 2)
        mu2 = self.mu ** 2
        A = J_pos_arm.T @ J_pos_arm + (lam2 + mu2) * np.eye(3)
        b = J_pos_arm.T @ pos_err + mu2 * (self.q_rest_123 - qpos_seed[:3])
        dq_arm = np.linalg.solve(A, b)

        new_q123 = qpos_seed[:3] + dq_arm

        # ----- Step 3: FK at (new_q123, seed_q456); read tool0 orientation -----
        qpos_after_arm = qpos_seed.copy()
        qpos_after_arm[:3] = new_q123
        self._fk(qpos_after_arm)
        R_cur = self.data.site_xmat[self.site_id].reshape(3, 3)

        # ----- Step 4: orientation error as a world-frame rotation vector -----
        # R_target = exp([e_rot]) · R_cur, i.e. e_rot is the angular
        # displacement carrying the current tool0 orientation onto the
        # target — the same (world) frame mj_jacSite's rotational Jacobian
        # maps joint velocities into.
        R_err = R_target @ R_cur.T
        q_err = np.zeros(4)
        mujoco.mju_mat2Quat(q_err, np.ascontiguousarray(R_err).ravel())
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, q_err, 1.0)

        # ----- Step 5: damped LS on the wrist — same recipe as the arm -----
        # w = |det(J_rot)| is the wrist manipulability: the scalar triple
        # product of the three (unit) wrist axis directions, ≈ |cos θ5| on
        # this arm. It hits 0 at gimbal lock (joint-4/6 axes aligned), where
        # the closed-form Euler extraction this replaced blew up (measured
        # 68° of J4/J6 demand per 1° of target twist at θ5 = 89.5°). Damping
        # ramps in smoothly instead: J4/J6 steps stay bounded and the lost
        # rotation direction simply stops being tracked until the operator
        # backs off. Differential steps are also continuous by construction,
        # so the old ±2π branch-unwrap heuristics are unnecessary.
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J_rot = jacr[:, 3:6]                     # 3x3, wrist joints 4-6
        w_rot = abs(float(np.linalg.det(J_rot)))
        ramp_rot = max(0.0, 1.0 - w_rot / max(self.w0_rot, 1e-12))
        self.last_wrist_gimbal_proximity = float(ramp_rot)
        # Antipode gate: with the error angle past `rot_err_hold`, the
        # shortest-way direction of e_rot is unstable (it flips at 180°) —
        # park the wrist instead of chasing it (see the class docstring).
        wrist_parked = float(np.linalg.norm(e_rot)) > self.rot_err_hold
        if wrist_parked:
            dq_wrist = np.zeros(3)
        else:
            lam2_rot = self.lam_rot ** 2 + (self.lam0_rot ** 2) * (ramp_rot ** 2)
            A_rot = J_rot.T @ J_rot + lam2_rot * np.eye(3)
            dq_wrist = np.linalg.solve(A_rot, J_rot.T @ e_rot)
        new_q456 = qpos_seed[3:6] + dq_wrist

        # ----- Step 6: joint-limit clamp; limit-pressure metric -----
        qpos6 = np.concatenate([new_q123, new_q456])
        qpos6_reachable = np.clip(qpos6, self.joint_limits[:, 0], self.joint_limits[:, 1])

        # Limit pressure = L2 distance from the unclamped step to the
        # clamped one — how hard this tick pushes joints into their stops.
        # Unreachable orientations near gimbal no longer surface here (the
        # damped step stays small); they show up as
        # last_wrist_gimbal_proximity instead. Excludes the per-tick rate
        # caps below, which are speed limits, not unreachability. A parked
        # wrist (antipode gate above) takes no step at all, so it reports a
        # saturating pressure directly — the operator is pushing somewhere
        # maximally unreachable and should feel it.
        self.last_limit_pressure = float(
            np.linalg.norm(qpos6 - qpos6_reachable)
        )
        if wrist_parked:
            self.last_limit_pressure = max(self.last_limit_pressure, 0.35)

        # ----- Step 7: per-joint Δq cap (operator-safety velocity bound) -----
        qpos6 = qpos6_reachable
        dq_total = qpos6 - qpos_seed[:6]
        if self.max_dq_per_joint is not None:
            dq_total = np.clip(dq_total, -self.max_dq_per_joint, self.max_dq_per_joint)
        qpos6 = qpos_seed[:6] + dq_total

        # ----- Step 8: clamp into joint limits (don't reject) -----
        qpos6 = np.clip(qpos6, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return qpos6


# ---------- self-test ----------

def _self_test() -> None:
    """Round-trip: pick known qpos → FK → solve from that target → check."""
    solver = DecoupledIKSolver(max_dq_per_joint=[0.05] * 6)

    test_set = [
        ("home elbow-up",   np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0])),
        ("slight rotation", np.array([0.2, np.pi/2, np.pi/2, 0.1, 0.1, 0.1])),
        ("forward reach",   np.array([0.0, 1.0,    1.0,    0.0, 0.0, 0.0])),
        ("yaw + pitch",     np.array([0.3, 1.4,    1.6,   -0.2, 0.3, 0.4])),
    ]

    print(f"{'pose':22s}  Δqpos  |  pos_err  |  rot_err_frob")
    print("-" * 70)
    for label, q in test_set:
        # FK to get target.
        solver._fk(q)
        target_pos = solver.data.site_xpos[solver.site_id].copy()
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, solver.data.site_xmat[solver.site_id])

        # Seed with home pose (equals q only for the first case; the rest
        # exercise a single capped Newton step from a distant seed).
        seed = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = solver.solve(target_pos, target_quat, seed)

        full_q = np.zeros(8)
        full_q[:6] = result
        solver._fk(full_q)
        actual_pos = solver.data.site_xpos[solver.site_id].copy()
        actual_R = solver.data.site_xmat[solver.site_id].reshape(3, 3)
        target_R = _quat_wxyz_to_R(target_quat)
        rot_err = float(np.linalg.norm(actual_R - target_R, ord='fro'))
        pos_err = float(np.linalg.norm(actual_pos - target_pos))
        diff = float(np.linalg.norm(result - q[:6]))
        print(f"{label:22s}  {diff:.4f}  |  {pos_err*1000:5.1f} mm  |  {rot_err:.3f}")

    # Boundary-handling smoke tests — all should return a valid 6-vector,
    # no None, no exception.
    print("\nboundary checks (target placed deliberately bad):")
    seed = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
    far_pos = np.array([2.0, 0.0, 0.4])   # way past reach
    out = solver.solve(far_pos, np.array([1.0, 0.0, 0.0, 0.0]), seed)
    print(f"  far-out-of-reach: qpos6 finite={bool(np.all(np.isfinite(out)))}  "
          f"|Δq|={float(np.linalg.norm(out - seed[:6])):.4f}")
    # Force a near-gimbal target by setting θ5 near ±π/2.
    q_gimbal = np.array([0.0, np.pi/2, np.pi/2, 0.0, np.pi/2 - 1e-4, 0.0])
    solver._fk(q_gimbal)
    g_pos = solver.data.site_xpos[solver.site_id].copy()
    g_quat = np.zeros(4)
    mujoco.mju_mat2Quat(g_quat, solver.data.site_xmat[solver.site_id])
    out = solver.solve(g_pos, g_quat, seed)
    print(f"  wrist near gimbal: qpos6 finite={bool(np.all(np.isfinite(out)))}  "
          f"θ5={out[4]:.4f}")

    # Gimbal conditioning: a 1° twist of the target near θ5 = 88° must not
    # demand a large J4/J6 step. The closed-form Euler extraction this
    # solver replaced demanded ~27° here; the damped wrist stays bounded.
    raw = DecoupledIKSolver()   # no Δq cap — observe the raw step
    q88 = np.array([0.2, 1.4, 1.6, 0.3, np.radians(88.0), 0.2, 0.0, 0.0])
    pos88, quat88 = raw.fk(q88)
    R88 = _quat_wxyz_to_R(quat88)
    worst = 0.0
    for axis in np.eye(3):
        for sign in (1.0, -1.0):
            ang = np.radians(1.0) * sign
            K = np.array([[0.0, -axis[2], axis[1]],
                          [axis[2], 0.0, -axis[0]],
                          [-axis[1], axis[0], 0.0]])
            R_t = R88 @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K))
            q_t = np.zeros(4)
            mujoco.mju_mat2Quat(q_t, np.ascontiguousarray(R_t).ravel())
            o = raw.solve(pos88, q_t, q88)
            worst = max(worst, abs(o[3] - q88[3]), abs(o[5] - q88[5]))
    ok = worst < np.radians(5.0)
    print(f"  gimbal conditioning @θ5=88°: worst J4/J6 step for 1° twist = "
          f"{np.degrees(worst):.2f}°  [{'ok' if ok else 'FAIL'}]")
    assert ok, "wrist step near gimbal not bounded — damping broken?"

    # Convergence: iterating the differential wrist on a fixed reachable
    # target must drive the orientation error to ~zero (the closed form
    # was exact in one step; the damped step must get there iteratively).
    q_t = np.array([0.3, 1.4, 1.6, -0.2, 0.3, 0.4])
    raw._fk(q_t)
    t_pos = raw.data.site_xpos[raw.site_id].copy()
    t_quat = np.zeros(4)
    mujoco.mju_mat2Quat(t_quat, raw.data.site_xmat[raw.site_id])
    cur = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        cur[:6] = raw.solve(t_pos, t_quat, cur)
    _, q_reached = raw.fk(cur)
    R_a, R_b = _quat_wxyz_to_R(q_reached), _quat_wxyz_to_R(t_quat)
    cos_err = (float(np.trace(R_a @ R_b.T)) - 1.0) / 2.0
    rot_err_deg = float(np.degrees(np.arccos(np.clip(cos_err, -1.0, 1.0))))
    pos_err_mm = float(np.linalg.norm(raw.data.site_xpos[raw.site_id] - t_pos)) * 1000
    ok = rot_err_deg < 1.0 and pos_err_mm < 15.0
    print(f"  convergence (100 iters): rot_err={rot_err_deg:.3f}°  "
          f"pos_err={pos_err_mm:.1f} mm  [{'ok' if ok else 'FAIL'}]")
    assert ok, "iterated solve did not converge to the target pose"

    # Antipode gate: a demand ~150° away must PARK the wrist (zero step,
    # saturating limit pressure) while ~100° away still tracks. Guards the
    # bang-bang oscillation seen on hardware when the operator twists far
    # past the θ5 stop and keeps going (near 180° the shortest-way error
    # direction flips under tiny jitter).
    q0 = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.3, 0.0, 0.0, 0.0])
    p0, quat0 = raw.fk(q0)
    R0 = _quat_wxyz_to_R(quat0)

    def _twisted_target(deg: float) -> np.ndarray:
        ang = np.radians(deg)
        K = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        R_t = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)) @ R0
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, np.ascontiguousarray(R_t).ravel())
        return qt

    out150 = raw.solve(p0, _twisted_target(150.0), q0)
    parked = bool(np.all(out150[3:] == q0[3:6]))
    pressure_at_park = raw.last_limit_pressure
    out100 = raw.solve(p0, _twisted_target(100.0), q0)
    tracks = float(np.linalg.norm(out100[3:] - q0[3:6])) > 1e-3
    ok = parked and pressure_at_park >= 0.35 and tracks
    print(f"  antipode gate: parked@150°={parked} (pressure={pressure_at_park:.2f})  "
          f"tracks@100°={tracks}  [{'ok' if ok else 'FAIL'}]")
    assert ok, "antipode gate broken (wrist should park at 150°, track at 100°)"


if __name__ == "__main__":
    _self_test()
