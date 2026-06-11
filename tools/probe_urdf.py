"""Probe the TRLC-DK1 URDF as loaded by mujoco.

Two passes:
  1. Plain `MjModel.from_xml_path` of the URDF — confirms what mujoco's
     URDF importer produces (in particular, whether `tool0` survives as
     its own body — it doesn't, because fixed joints get merged).
  2. `MjSpec`-based load that re-attaches a `tool0` site to `link6-7`
     using the offset from the URDF's `gripper_tool0` joint. Sites are
     preserved by the importer and give us first-class Jacobian access.

Run:
    python tools/probe_urdf.py path/to/TRLC-DK1-Follower.urdf
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

# URDF gripper_tool0 fixed joint, transcribed from the URDF.
TOOL0_OFFSET_XYZ = np.array([0.158, 0.0, 0.0])
TOOL0_OFFSET_RPY = np.array([-np.pi / 2, 0.0, -np.pi / 2])  # roll, pitch, yaw

JOINT_TYPE_NAMES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def name_of(model: mujoco.MjModel, obj_type, idx: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, idx)
    return name if name is not None else f"<unnamed#{idx}>"


def rpy_to_wxyz(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy convention: R = Rz(yaw) · Ry(pitch) · Rx(roll). Returns (w,x,y,z)."""
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


def print_model_summary(model: mujoco.MjModel) -> None:
    print("=== Model summary ===")
    print(f"  nq    (generalized coords) = {model.nq}")
    print(f"  nv    (DoFs)               = {model.nv}")
    print(f"  njnt  (joints)             = {model.njnt}")
    print(f"  nbody (bodies, incl world) = {model.nbody}")
    print(f"  nsite (sites)              = {model.nsite}")


def print_joints(model: mujoco.MjModel) -> None:
    print("\n=== Joints ===")
    for j in range(model.njnt):
        jname = name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = JOINT_TYPE_NAMES.get(int(model.jnt_type[j]), f"type#{int(model.jnt_type[j])}")
        qadr = int(model.jnt_qposadr[j])
        rng = model.jnt_range[j]
        limited = bool(model.jnt_limited[j])
        rng_str = f"[{rng[0]:+.3f}, {rng[1]:+.3f}]" if limited else "(unlimited)"
        print(f"  [{j}] {jname:20s} type={jtype:6s} qpos_idx={qadr:>2}  range={rng_str}")


def print_bodies(model: mujoco.MjModel) -> None:
    print("\n=== Bodies ===")
    for b in range(model.nbody):
        bname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, b)
        parent = int(model.body_parentid[b])
        pname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, parent) if parent != b else "(self/world)"
        print(f"  [{b}] {bname:20s} parent={pname}")


def print_sites(model: mujoco.MjModel) -> None:
    if model.nsite == 0:
        return
    print("\n=== Sites ===")
    for s in range(model.nsite):
        sname = name_of(model, mujoco.mjtObj.mjOBJ_SITE, s)
        body = int(model.site_bodyid[s])
        bname = name_of(model, mujoco.mjtObj.mjOBJ_BODY, body)
        pos = model.site_pos[s]
        quat = model.site_quat[s]  # (w, x, y, z)
        print(
            f"  [{s}] {sname:20s} attached_to={bname:12s} "
            f"local_pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
            f"local_quat_wxyz=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
        )


def fk_dump(model: mujoco.MjModel, data: mujoco.MjData, label: str) -> None:
    print(f"\n=== FK at {label} ===")
    mujoco.mj_forward(model, data)

    bodies = ["world", "link1-2", "link2-3", "link3-4", "link4-5", "link5-6", "link6-7"]
    for tname in bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tname)
        if bid == -1:
            continue
        pos = data.xpos[bid]
        q = data.xquat[bid]  # mujoco quaternion is (w, x, y, z)
        print(
            f"  body  '{tname:12s}' (id={bid:>2}): "
            f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
            f"quat_wxyz=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})"
        )

    for sname in ["tool0"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid == -1:
            continue
        pos = data.site_xpos[sid]
        # Site world rotation is stored as a 3x3 row-major matrix in site_xmat.
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, data.site_xmat[sid])
        print(
            f"  site  '{sname:12s}' (id={sid:>2}): "
            f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
            f"quat_wxyz=({q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f})"
        )


def load_with_tool0_site(urdf_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the URDF via MjSpec and add a `tool0` site to `link6-7` reproducing
    the URDF's `gripper_tool0` fixed-joint offset."""
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    parent = spec.body("link6-7")
    if parent is None:
        raise RuntimeError("link6-7 not found in spec")
    parent.add_site(
        name="tool0",
        pos=TOOL0_OFFSET_XYZ.tolist(),
        quat=rpy_to_wxyz(TOOL0_OFFSET_RPY).tolist(),
    )
    model = spec.compile()
    return model, mujoco.MjData(model)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} path/to/TRLC-DK1-Follower.urdf")
    urdf = Path(sys.argv[1]).expanduser()
    print(f"loading: {urdf}\n")
    if not urdf.exists():
        raise SystemExit(f"URDF not found at {urdf}")

    print("##########  Pass 1: plain URDF import  ##########\n")
    model = mujoco.MjModel.from_xml_path(str(urdf))
    data = mujoco.MjData(model)
    print_model_summary(model)
    print_joints(model)
    print_bodies(model)
    print_sites(model)
    data.qpos[:] = 0
    fk_dump(model, data, "qpos = 0")

    print("\n\n##########  Pass 2: MjSpec with re-attached tool0 site  ##########\n")
    model2, data2 = load_with_tool0_site(urdf)
    print_model_summary(model2)
    print_sites(model2)
    data2.qpos[:] = 0
    fk_dump(model2, data2, "qpos = 0")

    # Bonus: small custom pose so we can see the chain move.
    custom = np.zeros(model2.nq)
    custom[1] = np.pi / 2   # joint2 raise shoulder
    custom[2] = np.pi / 2   # joint3 elbow
    data2.qpos[:] = custom
    fk_dump(model2, data2, "qpos = [0, π/2, π/2, 0, 0, 0, 0, 0]")


if __name__ == "__main__":
    main()
