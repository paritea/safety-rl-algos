from omnisafe.envs.core import Wrapper
import torch
from typing import Any, Dict, Tuple

class UnsafeStateCounterWrapper(Wrapper):    
    def __init__(self, env, device=None):
        super().__init__(env, device)
        self.total_unsafe_states = 0

    def reset(self, seed: int = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment and clear the unsafe state counter."""
        self.total_unsafe_states = 0
        return self._env.reset(seed)

    def step(self, action: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        """Take a step in the environment and track unsafe states."""
        observation, reward, cost, terminated, truncated, info = self._env.step(action)
        if cost > 0:  # Increment unsafe state counter if cost is positive
            self.total_unsafe_states += 1
        if terminated or truncated:
            print(f"Total unsafe states reached in finished episode: {self.total_unsafe_states}")
        return observation, reward, cost, terminated, truncated, info
