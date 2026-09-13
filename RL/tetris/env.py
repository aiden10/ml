import gymnasium as gym
import numpy as np
import random
import tetris_gymnasium.envs

class Env:
    ROTATIONS = 4

    def __init__(self, render_mode: str = "human"):
        self.prev_holes = 0
        self.env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
        self.macro_action_space = gym.spaces.Discrete(self.ROTATIONS * self.env.unwrapped.width)

    def reset(self):
        obs, info = self.env.reset(seed=random.randint(0, 2**16))
        self.prev_holes = 0
        return obs

    def compute_reward(self, obs, info, terminated: bool) -> float:
        reward = 0.0
        if terminated:
            return -5.0
        
        lines_cleared = info.get("lines_cleared", 0)
        if lines_cleared > 0:
            reward += (lines_cleared ** 2) * 2.5  # greater reward for clearing multiple lines
        
        board = obs["board"]
        occupied = board > 0

        # Cumulative max down columns: once a block appears, stays True all the way down
        has_block_above = np.maximum.accumulate(occupied, axis=0)

        # A hole is empty, but has a block somewhere above it
        holes_mask = (~occupied) & has_block_above

        # Count total holes and penalize
        total_holes = int(np.sum(holes_mask))
        reward -= (total_holes - self.prev_holes) * 1.5  
        self.prev_holes = total_holes
        
        return reward
    
    def step(self, action: int):
        """Execute one placement: rotate, move to a column, then hard-drop."""
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid macro action: {action}")

        rotation, target_column = divmod(action, self.env.unwrapped.width)
        tetris = self.env.unwrapped
        piece = tetris.active_tetromino
        for _ in range(rotation):
            piece = tetris.rotate(piece, clockwise=True)

        tetris.active_tetromino = piece
        tetris.x = tetris.padding + target_column
        tetris.y = 0

        obs, _, terminated, truncated, info = self.env.step(tetris.actions.hard_drop)
        reward = self.compute_reward(obs, info, terminated)
        return obs, reward, terminated, truncated, info

    def get_action_space(self):
        return self.macro_action_space

    def get_valid_actions(self) -> list[int]:
        """Return macro actions whose rotated piece can be placed at the top."""
        tetris = self.env.unwrapped
        if tetris.game_over:
            return []

        valid_actions = []
        piece = tetris.active_tetromino
        for rotation in range(self.ROTATIONS):
            max_column = tetris.width - piece.matrix.shape[1]
            for column in range(max_column + 1):
                x = tetris.padding + column
                if not tetris.collision(piece, x, 0):
                    valid_actions.append(rotation * tetris.width + column)
            piece = tetris.rotate(piece, clockwise=True)

        return valid_actions
    
    def get_observation_space(self):
        return self.env.observation_space

    def render(self):
        integer_dtype = np.integer
        try:
            np.integer = np.int64
            return self.env.render()
        finally:
            np.integer = integer_dtype
    
    def close(self):
        self.env.close()
