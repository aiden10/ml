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
    
    def learn(self, target_network: nn.Module, state, action: int, reward: float, next_state, done: bool):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)
        
        q_values = self.forward(state_tensor)
        current_q = q_values[action]
        
        with torch.no_grad():
            next_q_values = target_network.forward(next_state_tensor)
            max_next_q = torch.max(next_q_values)
            
            target_q = reward if done else reward + self.df * max_next_q
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()