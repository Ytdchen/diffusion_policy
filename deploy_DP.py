import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.rlbench.GymWrapper import GymWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply
from collections import deque


# Run4, n_obs_steps=2, horizon=1
# checkpoint = '/home/david/diffusion_policy/data/outputs/2024.05.28/18.00.36_train_ibc_dfo_hybrid_gexp/checkpoints/epoch=0030-train_action_mse_error=0.148.ckpt'
# checkpoint = '/home/david/diffusion_policy/data/outputs/2024.05.28/18.00.36_train_ibc_dfo_hybrid_gexp/checkpoints/epoch=0080-train_action_mse_error=0.155.ckpt'
# checkpoint = '/home/david/diffusion_policy/data/outputs/2024.05.30/12.35.21_train_ibc_dfo_hybrid_gexp/checkpoints/epoch=0070-train_action_mse_error=0.155.ckpt'
# checkpoint = '/home/david/diffusion_policy/data/outputs/2024.05.30/13.46.04_train_ibc_dfo_hybrid_gexp/checkpoints/epoch=0260-train_action_mse_error=0.162.ckpt'
checkpoint = '/home/david/diffusion_policy/data/outputs/2024.05.30/13.46.04_train_ibc_dfo_hybrid_gexp/checkpoints/epoch=0810-train_action_mse_error=0.154.ckpt'

# Run4, n_obs_steps=1, horizon=1
# checkpoint = '/home/david/diffusion_policy/data/outputs/2024.05.29/17.15.48_train_ibc_dfo_hybrid_gexp/checkpoints/epoch=0020-train_action_mse_error=0.129.ckpt'

# Run4, n_obs_steps=2, n_action_steps=8, horizon=10
# checkpoint='/home/david/diffusion_policy/data/outputs/2024.06.16/18.49.07_train_ibc_dfo_hybrid_gexp/checkpoints/latest.ckpt'

# Run4, n_obs_steps=2, n_action_steps=8, horizon=16
c# heckpoint='/home/david/diffusion_policy/data/outputs/2024.06.17/10.37.18_train_ibc_dfo_hybrid_gexp/checkpoints/latest.ckpt'

env = GymWrapper(task_class='PutRubbishInBin', observation_mode='vision', render_mode='human')
env = MultiStepWrapper(env=env, n_action_steps=8, n_obs_steps=2)

# load checkpoint
payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# get policy from workspace
policy = workspace.model
# if cfg.training.use_ema:
#     policy = workspace.ema_model

# device = torch.device(device)
# policy.to(device)
policy.eval()
env.seed(0)
observation = env.reset()
policy.reset()

# for key in observation:
#     print(observation[key].shape)

while True:

   

    # run policy
    with torch.no_grad():
        action_dict = policy.predict_action(observation)


    action = action_dict['action'][0]

    # step env
    observation, reward, done, info = env.step(action)
    
    env.render()


