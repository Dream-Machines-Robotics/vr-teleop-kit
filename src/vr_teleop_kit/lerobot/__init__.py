"""LeRobot ``Teleoperator`` adapters.

Importing the submodules registers the config types with LeRobot's
``TeleoperatorConfig`` registry, after which
``--teleop.type=bi_quest_teleop`` / ``single_arm_quest_teleop`` work in
LeRobot CLIs. Requires ``lerobot`` to be installed.
"""

from .bi_quest_teleop import BiQuestTeleoperator, BiQuestTeleoperatorConfig
from .single_arm_quest_teleop import (
    SingleArmQuestTeleoperator,
    SingleArmQuestTeleoperatorConfig,
)

__all__ = [
    "BiQuestTeleoperator",
    "BiQuestTeleoperatorConfig",
    "SingleArmQuestTeleoperator",
    "SingleArmQuestTeleoperatorConfig",
]
