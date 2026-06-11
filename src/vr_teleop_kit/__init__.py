"""WebXR (Meta Quest) teleoperation stack for the TRLC-DK1 arm.

Layers (see the README for the architecture):
  vr_teleop_kit.core    — robot-agnostic clutch-relative pose mapping
                          with absorbing reach limits
  vr_teleop_kit.ik      — DK1-tuned decoupled IK
  vr_teleop_kit.relay   — FastAPI WebSocket relay + WebRTC cameras +
                          the WebXR page served to the Quest
  vr_teleop_kit.lerobot — LeRobot ``Teleoperator`` adapters (import
                          this subpackage to register the plugin types;
                          requires ``lerobot`` installed)
"""

__version__ = "0.1.0"
