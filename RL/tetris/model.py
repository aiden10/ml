import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class TetrisDQN(nn.Module):
    def __init__(self, input: int, hidden: int, output: int, learning_rate: float, discount_factor: float):
        super().__init__()
        self.lr = learning_rate
        self.df = discount_factor
        self.relu = nn.ReLU()
        self.input_layer = nn.Linear(input, hidden)
        self.hidden_layer = nn.Linear(hidden, hidden)
        self.output_layer = nn.Linear(hidden, output)
        
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
        flat = np.concatenate([board, mask])
        return torch.as_tensor(flat, dtype=torch.float32)
    
    def learn(self, target_network: nn.Module, replay_buffer: list, batch_size: int = 64):
        if len(replay_buffer) < batch_size:
            return None

        # Get random batch of replays
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert chosen batch to tensors
        state_batch = torch.stack([self.process_obs(s) for s in states])
        next_state_batch = torch.stack([self.process_obs(ns) for ns in next_states])
        
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        current_q = self(state_batch).gather(1, action_batch)
        
        # Do the actual Q learning formula
        with torch.no_grad():
            max_next_q = target_network(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch + (1.0 - done_batch) * (self.df * max_next_q)

        # Backprop
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()