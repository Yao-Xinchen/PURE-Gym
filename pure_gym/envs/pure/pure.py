from pure_gym import GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, Union, Any

from pure_gym.envs.base.legged_robot import LeggedRobot
from pure_gym.utils.terrain import Terrain
from pure_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from pure_gym.utils.helpers import class_to_dict
from .pure_config import PureCfg, PureCfgPPO

class Pure(LeggedRobot):
    def __init__(self, cfg: PureCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.tensor(np.pi, device=self.device)