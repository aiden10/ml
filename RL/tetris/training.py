import os
import random
import numpy as np
import torch
from model import TetrisDQN
from env import Env

CHECKPOINT_PATH = "tetris_checkpoint.pt"

if __name__ == "__main__":
    gym = Env()
    action_space = gym.get_action_space()
    observation_space = gym.get_observation_space()
    input_size = (
        np.prod(observation_space["board"].shape)
        + np.prod(observation_space["active_tetromino_mask"].shape)
    )

    num_actions = action_space.n if hasattr(action_space, "n") else len(action_space)

    agent = TetrisDQN(input=input_size, hidden=64, output=num_actions, learning_rate=0.01, discount_factor=0.99)
    target_network = TetrisDQN(input=input_size, hidden=64, output=num_actions, learning_rate=0.01, discount_factor=0.99)
    target_network.load_state_dict(agent.state_dict())

    # hyperparameters
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    start_episode = 0
    total_episodes = 1000
    target_sync_freq = 10  # Sync target network every 10 episodes

    # Resume from checkpoint if it exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        target_network.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"] + 1
        epsilon = checkpoint["epsilon"]
        print(f"Resumed from episode {start_episode} with epsilon {epsilon:.4f}")

    replay_buffer = []

    for episode in range(start_episode, total_episodes):
        obs = gym.reset()
        done = False

        while not done:
            # Don't always take best expected value option, explore with decay to find other possibilities
            if random.random() < epsilon:
                action = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    output = agent(agent.process_obs(obs))
                    action = torch.argmax(output).item()

            next_obs, reward, terminated, truncated, info = gym.step(action)
            done = terminated or truncated

            # Store transition (tuple)
            replay_buffer.append((obs, action, reward, next_obs, done))

            agent.learn(target_network, replay_buffer)

            obs = next_obs

        # Decay epsilon per episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Sync target network weights
        if episode % target_sync_freq == 0:
            target_network.load_state_dict(agent.state_dict())

        # Save checkpoint periodically (every 50 episodes) and on final episode
        if (episode + 1) % 50 == 0 or (episode + 1) == total_episodes:
            checkpoint = {
                "episode": episode,
                "agent_state_dict": agent.state_dict(),
                "target_state_dict": target_network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "epsilon": epsilon,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Saved checkpoint at episode {episode + 1} (epsilon: {epsilon:.4f})")

    gym.close()
