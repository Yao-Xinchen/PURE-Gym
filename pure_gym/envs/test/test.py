from pure_gym import GYM_ROOT_DIR, envs
from time import time
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
from .test_config import TestCfg, TestCfgPPO


class Test(LeggedRobot):
    def __init__(self, cfg: TestCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.tensor(np.pi, device=self.device)

        def find_indices(sub_list, main_list):
            return [main_list.index(item) for item in sub_list if item in main_list]

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        # physics step
        for _ in range(self.cfg.control.decimation):
            self.step_counts += 1
            self.action_delay_queue = torch.cat(
                (self.actions.unsqueeze(1), self.action_delay_queue[:, :-1]),
                dim=1
            )
            self.torques = self._compute_torques(
                self.action_delay_queue[torch.arange(self.num_envs), self.action_delay_idx, :]
            )
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            if self.cfg.domain_rand.lift_robots:
                self._lift_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # self.compute_dof_vel()

        self.post_physics_step()

        # observation
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.privileged_obs_buf,  # TODO: check this
            self.rew_buf,
            self.reset_buf,
            self.extras
        )

    def _post_physics_step_callback(self):
        # called in post_physics_step()
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # TODO: add more callbacks

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval_s == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        # no command for now
        pass

    def _compute_torques(self, actions):
        torques = torch.zeros([self.num_envs, self.num_dof], dtype=torch.float, device=self.device, )
        return torques

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # TODO: check the diff
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf

    def compute_observations(self):
        pass

    def _get_noise_scale_vec(self, cfg):
        # called in _init_buffers()
        noise_vec = torch.zeros_like(self.obs_buf[0])
        return noise_vec

    def _lift_robots(self):
        running_time = self.step_counts * self.sim_params.dt
        # lift those whose running time is between lift_start and lift_start + lift_duration
        where = (running_time > self.lift_start) & (running_time < self.lift_start + self.cfg.domain_rand.lift_duration)
        lift_height = self.cfg.domain_rand.lift_height
        # move robot to the height immediately and fix it there
        self.root_states[where, 2] = lift_height
        self.root_states[where, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float)
        self.root_states[where, 7:10] = 0.0  # linear velocity
        self.root_states[where, 10:13] = 0.0  # angular velocity
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _init_buffers(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = (gymtorch.wrap_tensor(net_contact_forces)
                               .view(self.num_envs, -1, 3))  # shape: num_envs, num_bodies, xyz axis
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name in self.cfg.init_state.default_joint_angles:
                angle = self.cfg.init_state.default_joint_angles[name]
            else:
                angle = 0.0
            self.default_dof_pos[i] = angle
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # -------------------------------------
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # action delay buffers
        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
        )
        self.action_delay_queue = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.domain_rand.randomize_action_delay:
            self.action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1).long()

        # step counts
        self.step_counts = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        # lift
        if self.cfg.domain_rand.lift_robots:
            earliest = self.cfg.domain_rand.lift_start_at[0]
            latest = self.cfg.domain_rand.lift_start_at[1]
            self.lift_start = (torch.rand([self.num_envs, ], dtype=torch.float, device=self.device)
                               * (latest - earliest) + earliest)
