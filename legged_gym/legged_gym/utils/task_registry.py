# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from copy import deepcopy
import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import wandb

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    # register task class(vecenv) and config
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

        # attach env_cfg.domain_randomization to wandb.config
        if not args.no_wandb and not args.debug:
            domain_rand_cfg_dict = class_to_dict(env_cfg.domain_rand)
            # domain_rand_cfg = {
            #     "domain_rand": {
            #         "randomize_friction"    : env_cfg.domain_rand.randomize_friction,
            #         "randomize_base_mass"   : env_cfg.domain_rand.randomize_base_mass,
            #         "randomize_base_com"    : env_cfg.domain_rand.randomize_base_com,
            #         "push_robots"           : env_cfg.domain_rand.push_robots,
            #         "randomize_motor"       : env_cfg.domain_rand.randomize_motor,
            #         "action_delay"          : env_cfg.domain_rand.action_delay,

            #         "randomize_camera"      : env_cfg.domain_rand.randomize_camera,
            #         "randomize_depth_noise" : env_cfg.domain_rand.randomize_depth_noise
            #     },
            #     "domain_rand_para": {
            #         "friction_range"        : env_cfg.domain_rand.friction_range,
            #         "added_mass_range"      : env_cfg.domain_rand.added_mass_range,
            #         "added_com_range"       : env_cfg.domain_rand.added_com_range,
            #         "push_interval_s"       : env_cfg.domain_rand.push_interval_s,
            #         "max_push_vel_xy"       : env_cfg.domain_rand.max_push_vel_xy,
            #         "motor_strength_range"  : env_cfg.domain_rand.motor_strength_range,
            #         "static_motor_strength" : env_cfg.domain_rand.static_motor_strength,
            #         "delay_update_global_steps" : env_cfg.domain_rand.delay_update_global_steps,
            #         "action_curr_step"      : env_cfg.domain_rand.action_curr_step,
            #         "action_curr_step_scratch" : env_cfg.domain_rand.action_curr_step_scratch,
            #         "action_delay_view"     : env_cfg.domain_rand.action_delay_view,
            #         "action_buf_len"        : env_cfg.domain_rand.action_buf_len
            #     },
            #     "domain_rand_noise": {
            #         "camera_pos_range"      : env_cfg.domain_rand.camera_pos_range,
            #         "camera_angle_range"    : env_cfg.domain_rand.camera_angle_range,
            #         "camera_fov_range"      : env_cfg.domain_rand.camera_fov_range,
            #         "light_intensity_prob"  : env_cfg.domain_rand.light_intensity_prob,
            #         "max_intensity"         : env_cfg.domain_rand.max_intensity,
            #         "light_rand_type"       : env_cfg.domain_rand.light_rand_type,
            #         "reflectivity_prob"     : env_cfg.domain_rand.reflectivity_prob,
            #         "texture_scale"         : env_cfg.domain_rand.texture_scale,
            #         "max_occlusion_num"     : env_cfg.domain_rand.max_occlusion_num,
            #         "max_occ_width"         : env_cfg.domain_rand.max_occ_width,
            #         "max_occ_height"        : env_cfg.domain_rand.max_occ_height,
            #         "noise_type"            : env_cfg.domain_rand.noise_type
            #     }
            # }
            wandb.config.update(domain_rand_cfg_dict)

        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, init_wandb=True, log_root="default", **kwargs) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name) # like 'logs/rough_a1/Jul01_12-34-56_run_name'
        elif log_root is None:
            log_dir = None
        else:
            log_dir = log_root
            # os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, 
                                train_cfg_dict, 
                                log_dir, 
                                init_wandb=init_wandb,
                                device=args.rl_device, **kwargs)
        # save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if args.resumeid:
            log_root = LEGGED_GYM_ROOT_DIR + f"/logs/{args.proj_name}/" + args.resumeid
            resume = True
        if resume:
            # load previously trained model
            print(log_root)
            print(train_cfg.runner.load_run)
            # load_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', "rough_a1", train_cfg.runner.load_run)
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            runner.load(resume_path)
            if not train_cfg.policy.continue_from_last_std:
                runner.alg.actor_critic.reset_std(train_cfg.policy.init_noise_std, 12, device=runner.device)

        if "return_log_dir" in kwargs:
            return runner, train_cfg, os.path.dirname(resume_path)
        else:    
            return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
