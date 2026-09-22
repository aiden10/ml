import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class TetrisAfterstateValue(nn.Module):
    """Estimate the value of a complete, post-placement Tetris state."""

    def __init__(self, input: int, hidden: int, learning_rate: float, discount_factor: float):
        super().__init__()
        self.lr = learning_rate
        self.df = discount_factor
        self.relu = nn.ReLU()
        self.input_layer = nn.Linear(input, hidden)
        self.hidden_layer = nn.Linear(hidden, hidden)
        self.output_layer = nn.Linear(hidden, 1)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    def process_obs(self, obs) -> torch.Tensor:
        board = obs["board"].flatten()
        mask = obs["active_tetromino_mask"].flatten()
        holder = obs["holder"].flatten()
        queue = obs["queue"].flatten()
        flat = np.concatenate([board, mask, holder, queue])
        return torch.as_tensor(flat, dtype=torch.float32)
    
    def score_afterstates(self, outcomes) -> torch.Tensor:
        """Return ``reward + discount * V(afterstate)`` for each placement.

        Each outcome is ``(action, reward, afterstate, done)``.  The action is
        intentionally not an input to this network: the simulated afterstate
        already captures exactly what that action did to the board.
        """
        afterstate_batch = torch.stack(
            [self.process_obs(afterstate) for _, _, afterstate, _ in outcomes]
        )
        rewards = torch.tensor(
            [reward for _, reward, _, _ in outcomes], dtype=torch.float32
        )
        dones = torch.tensor(
            [done for _, _, _, done in outcomes], dtype=torch.float32
        )
        values = self(afterstate_batch).squeeze(1)
        return rewards + (1.0 - dones) * self.df * values

    def learn(self, target_network: nn.Module, replay_buffer, batch_size: int = 64):
        if len(replay_buffer) < batch_size:
            return None

        # A replay item contains one source state and every legal placement
        # from it. Value iteration trains V(state) toward the best backed-up
        # outcome, rather than averaging conflicting targets for its actions.
        batch = replay_buffer.sample(batch_size)
        states, outcome_groups = zip(*batch)
        state_batch = torch.stack([self.process_obs(s) for s in states])

        with torch.no_grad():
            target_values = torch.stack(
                [target_network.score_afterstates(outcomes).max() for outcomes in outcome_groups]
            ).unsqueeze(1)

        loss = self.criterion(self(state_batch), target_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
