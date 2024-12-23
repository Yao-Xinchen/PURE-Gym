from .base.legged_robot import LeggedRobot
from .pure.pure import Pure
from .pure.pure_config import PureCfg, PureCfgPPO

import os
from pure_gym.utils.task_registry import task_registry

task_registry.register(
    "pure", Pure, PureCfg(), PureCfgPPO()
)