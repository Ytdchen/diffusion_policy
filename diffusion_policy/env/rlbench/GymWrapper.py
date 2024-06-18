from typing import Union, Dict, Tuple


import numpy as np

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor


from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
import rlbench.tasks as tasks

import gym
from gym import spaces

    
class GymWrapper(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='vision', 
                 render_mode: Union[None, str] = None):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)

        action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        task_class = getattr(tasks, task_class)
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.env.action_shape)

        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        elif observation_mode == 'vision':
            self.observation_space = spaces.Dict(
                spaces={
                # "state": spaces.Box(
                #     low=-np.inf, high=np.inf,
                #     shape=obs.get_low_dim_data().shape),
                # "left_shoulder_rgb": spaces.Box(
                #     low=0, high=1, 
                #     shape=obs.left_shoulder_rgb.shape),
                # "right_shoulder_rgb": spaces.Box(
                #     low=0, high=1, 
                #     shape=obs.right_shoulder_rgb.shape),
                # "state": spaces.Box(
                #     low=-1.0, high=1.0, shape=self.env.action_shape),
                # "wrist_rgb": spaces.Box(
                #     low=0, high=1, 
                #     shape=obs.wrist_rgb.shape),
                "image": spaces.Box(
                    low=0, high=1, 
                    shape=obs.front_rgb.shape),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.gripper_pose.shape),
                
                }
            )

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)
        

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode == 'vision':
            # img = np.transpose(obs.front_rgb, (2,1,0))
            img = np.moveaxis(obs.front_rgb,-1,0)/255
            dic ={
                
                # "state": obs.get_low_dim_data(),
                # "left_shoulder_rgb": obs.left_shoulder_rgb,
                # "right_shoulder_rgb": obs.right_shoulder_rgb,
                # "state": np.append(obs.joint_velocities, obs.gripper_open),
                # "wrist_rgb": obs.wrist_rgb,
                "image": img,
                "state": obs.gripper_pose,
            
            }
                
            return dic
        

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def reset(self, seed=None) -> Tuple[Dict[str, np.ndarray], dict]:
        descriptions, obs = self.task.reset()
        # del descriptions  # Not used.
        # if seed is not None:
        #     self._np_random, seed = seeding.np_random(seed)
        # return self._extract_obs(obs), {}
        return self._extract_obs(obs)
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        obs, reward, terminate = self.task.step(action)
        # return self._extract_obs(obs), reward, terminate, False,  {}
        return self._extract_obs(obs), reward, terminate, {}
    
    def close(self) -> None:
        self.env.shutdown()

