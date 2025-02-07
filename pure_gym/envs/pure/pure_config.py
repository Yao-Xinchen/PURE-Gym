from pure_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class PureCfg(LeggedRobotCfg):
    action_keys = [
        # TODO
    ]

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 6
        num_observations = 24  # TODO: update this
        num_privileged_obs = 0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 0

    class asset(LeggedRobotCfg.asset):
        file = "{GYM_ROOT_DIR}/resources/robots/pure/urdf/pure.urdf"
        name = "pure"
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        pos_offsets = {
            # TODO
        }

    class domain_rand(LeggedRobotCfg.domain_rand):
        # friction
        randomize_friction = True
        friction_range = [0.5, 1.5]
        # mass
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        # push
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        # delay
        randomize_action_delay = True
        delay_ms_range = [0, 10]
        # lift
        lift_robots = False
        lift_height = 1.0
        lift_duration = 4.0
        lift_start_at = [5, 10]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            # TODO
        }

    class control(LeggedRobotCfg.control):
        decimation = 4
        power_on_time = 2.0  # time to power on the robot

        action_scale_vel = 10.0
        action_scale_pos = 0.25

        p_gains = {
            # TODO
        }

        d_gains = {
            # TODO
        }

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = True
        static_friction = 0.5
        dynamic_friction = 0.5

    class rewards(LeggedRobotCfg.rewards):
        # don't inherit from base
        # to avoid using its reward functions
        class scales():
            torques = -1e-4
            # TODO: define more scales

        only_positive_rewards = False
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        # base_height_target = 1.
        max_contact_force = 1e4

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            torque = 0.05


class PureCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCriticRecurrent'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
