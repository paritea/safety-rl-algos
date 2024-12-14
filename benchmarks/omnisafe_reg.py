

from __future__ import annotations

# import all we need
import random
import omnisafe
from typing import Any, ClassVar

import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register, env_unregister
from benchmarks.safety_gym import SafetyPointGoalEnv


# Define environment class
class SpiceEnvironment(CMDP):
    _support_envs: ClassVar[list[str]] = ['Example-v0', 'Example-v1']  # Supported task names

    need_auto_reset_wrapper = True  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = True  # Whether `TimeLimit` Wrapper is needed

    def __init__(self, env_id: str, **kwargs) -> None:
        self.env = SafetyPointGoalEnv()
        self._count = 0
        self._num_envs = 1

        #passing no reduced dims in env creation
        if env_id == 'Example-v0':
            self._observation_space = spaces.Box(low=-0.5, high=0.5, shape=self.env.observation_space.shape)
            self._action_space = spaces.Box(low=-1.0, high=1.0, shape=self.env.action_space.shape)
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

    @property
    def max_episode_steps(self) -> None:
        """The max steps per episode."""
        return 10

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


