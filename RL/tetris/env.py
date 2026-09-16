import gymnasium as gym
import numpy as np
import random
import tetris_gymnasium.envs

class Env:
    ROTATIONS = 4

    def __init__(self, render_mode: str = "human"):
        self.prev_holes = 0
        self.curriculum_pieces_remaining = 0
        self.curriculum_solution = []
        self.env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
        self.macro_action_space = gym.spaces.Discrete(self.ROTATIONS * self.env.unwrapped.width)

    def reset(self):
        obs, info = self.env.reset(seed=random.randint(0, 2**16))
        self.prev_holes = 0
        self.curriculum_pieces_remaining = 0
        self.curriculum_solution = []
        return obs

    def reset_n_placements_from_line_clear(self, placements: int):
        """Reset to a board with a guaranteed vertical-I-piece solution.

        The bottom row is full except for ``placements`` cells. The active
        piece and the next pieces are forced to I-pieces, and one vertical I
        placement in each listed column completes the row. ``curriculum_solution``
        exposes that valid macro-action sequence for diagnostics or tests.
        """
        tetris = self.env.unwrapped
        i_piece_index = 0
        i_piece = tetris.tetrominoes[i_piece_index]

        vertical_actions = {}
        piece = i_piece
        for rotation in range(self.ROTATIONS):
            occupied_columns = np.flatnonzero(np.any(piece.matrix > 0, axis=0))
            if len(occupied_columns) == 1:
                occupied_column = int(occupied_columns[0])
                for target_column in range(tetris.width - piece.matrix.shape[1] + 1):
                    board_column = target_column + occupied_column
                    vertical_actions.setdefault(
                        board_column, rotation * tetris.width + target_column
                    )
            piece = tetris.rotate(piece, clockwise=True)

        if not 1 <= placements <= len(vertical_actions):
            raise ValueError(
                f"placements must be between 1 and {len(vertical_actions)}"
            )

        self.reset()
        tetris.board = tetris.create_board()

        available_columns = sorted(vertical_actions)
        selected_indices = np.linspace(
            0, len(available_columns) - 1, placements, dtype=int
        )
        gap_columns = [available_columns[index] for index in selected_indices]
        bottom_row = tetris.height - 1
        playable_columns = slice(tetris.padding, tetris.padding + tetris.width)
        tetris.board[bottom_row, playable_columns] = i_piece.id
        for column in gap_columns:
            tetris.board[bottom_row, tetris.padding + column] = 0

        self.curriculum_pieces_remaining = placements
        self.curriculum_solution = [vertical_actions[column] for column in gap_columns]
        self._set_curriculum_i_piece()
        self.prev_holes = 0
        return tetris._get_obs()

    def _set_curriculum_i_piece(self):
        """Expose I-pieces while a generated curriculum solution is active."""
        tetris = self.env.unwrapped
        tetris.active_tetromino = tetris.tetrominoes[0]
        tetris.reset_tetromino_position()
        tetris.queue.queue.clear()
        for _ in range(tetris.queue.size):
            tetris.queue.queue.append(0)

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
        if self.curriculum_pieces_remaining:
            self.curriculum_pieces_remaining -= 1
            if not (terminated or truncated) and self.curriculum_pieces_remaining:
                self._set_curriculum_i_piece()
                obs = tetris._get_obs()
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
