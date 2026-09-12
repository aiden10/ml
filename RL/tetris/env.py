import gymnasium as gym
import numpy as np
import tetris_gymnasium.envs

class Env:
    def __init__(self, seed: int = 42, render_mode: str = "human"):
        self.seed = seed
        self.env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        return obs

    def compute_reward(self, obs, info, terminated: bool) -> float:
        reward = 0.0
        if terminated:
            return -10.0

        lines_cleared = info.get("lines_cleared", 0)
        if lines_cleared > 0:
            reward += (lines_cleared ** 2) * 0.75  # greater reward for clearing multiple lines
        
        board = obs["board"]
        occupied = board > 0

        # Cumulative max down columns: once a block appears, stays True all the way down
        has_block_above = np.maximum.accumulate(occupied, axis=0)

        # A hole is empty, but has a block somewhere above it
        holes_mask = (~occupied) & has_block_above

        # Count total holes and penalize
        total_holes = int(np.sum(holes_mask))
        reward -= total_holes * 1.5  
        
        return reward
    
    def step(self, action: int):
        # step returns 5 values: obs, reward, terminated, truncated, info
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.compute_reward(obs, info, terminated)
        return obs, reward, terminated, truncated, info

    def get_action_space(self):
        return self.env.action_space
    
    def get_observation_space(self):
        return self.env.observation_space

    def render(self):
        # tetris-gymnasium uses np.integer as an astype dtype, which NumPy 2
        # rejects. Keep the compatibility change limited to the render call.
        integer_dtype = np.integer
        try:
            np.integer = np.int64
            return self.env.render()
        finally:
            np.integer = integer_dtype
    
    def close(self):
        self.env.close()
