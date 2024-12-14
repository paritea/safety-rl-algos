import torch
import omnisafe
import gymnasium as gym
import safety_gymnasium
from typing import Any, Dict, Tuple
from omnisafe.envs.core import CMDP, env_register, env_unregister
from gymnasium import make
from UnsafeStateCounterWrapper import UnsafeStateCounterWrapper
from benchmarks.omnisafe_reg import SpiceEnvironment
from benchmarks.gymnasium_wrapper import GymnasiumWrapper

@env_register
@env_unregister
class CustomSpiceEnv(GymnasiumWrapper):
    example_configs = 2


# env = CustomSpiceEnv('Example-v0')
env = CustomSpiceEnv('BipedalWalker-v1')

custom_cfgs = {
    'train_cfgs': {
        'total_steps': 102400,
        'vector_env_nums': 1,
        'parallel': 1
    },
    'algo_cfgs': {
        'steps_per_epoch': 1024,
        'update_iters': 10,
    },
    'logger_cfgs': {
        'use_wandb': False,
        'log_dir': './logs'
    }
}

# Initialize the agent
agent = omnisafe.Agent('CPO', "BipedalWalker-v1", custom_cfgs=custom_cfgs)

agent.learn()
agent.evaluate(10)

# # Evaluate the agent
# agent.evaluate(num_episodes=500)
# print(f"Total number of unsafe states for trained policy: {agent.env._env.total_unsafe_states}")
