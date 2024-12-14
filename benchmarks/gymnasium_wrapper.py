# import all we need
from __future__ import annotations

import random
import omnisafe
from typing import Any, ClassVar

import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register, env_unregister
from benchmarks.bipedalwalker import  BipedalWalkerEnv

class GymnasiumWrapper(CMDP):
    _support_envs: ClassVar[list[str]] = ['BipedalWalker-v1']  # Supported task names

    need_auto_reset_wrapper = True  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = True  # Whether `TimeLimit` Wrapper is needed

    def __init__(self, env_id: str, **kwargs) -> None:
        self.env = BipedalWalkerEnv()
        self._count = 0
        self._num_envs = 1

        #passing no reduced dims in env creation
        if env_id == 'BipedalWalker-v1':
            self._observation_space = self.env.observation_space
            self._action_space = self.env.action_space
        else:
            raise NotImplementedError

    def set_seed(self, seed: int) -> None:
        self.env.seed(seed)

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            self.set_seed(seed)
        self._count = 0
        state, info = self.env.reset()
        return  torch.as_tensor(state, dtype=torch.float32), info

    def render(self) -> Any:
        self.env.render()

    def render(self) -> Any:
        self.env.render()

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        return 1600

    def close(self) -> None:
        return self.env.close()

    def step(
            self,
            action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1
        state, reward, cost, done, truncation, info = self.env.step(action)

        obs = torch.as_tensor(state, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        cost = torch.as_tensor(cost, dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.bool)  # Convert boolean to tensor
        truncation = torch.as_tensor(truncation, dtype=torch.bool)  # Convert boolean to tensor
        final_info = {
            'state_original': torch.as_tensor(info['state_original'], dtype=torch.float32)
        }
        return obs, reward, cost, done, truncation, final_info

# env = GymnasiumWrapper(env_id='BipedalWalker-v1')
# env.reset(seed=0)
# n = 10
# while n > 0:
#     action = env.action_space.sample()
#     obs, reward, cost, terminated, truncated, info = env.step(action)
#     print('-' * 20)
#     print(f'obs: {obs}')
#     print(f'reward: {reward}')
#     print(f'cost: {cost}')
#     print(f'terminated: {terminated}')
#     print(f'truncated: {truncated}')
#     print('*' * 20)
#     if terminated or truncated:
#         break
#     n=n-1
# env.close()