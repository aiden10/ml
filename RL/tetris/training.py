import os
import random
import csv
import numpy as np
import torch
from model import TetrisDQN
from env import Env

CHECKPOINT_PATH = "tetris_checkpoint.pt"
METRICS_PATH = "tetris_metrics.csv"

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
    total_episodes = 10000
    target_sync_freq = 10  # Sync target network every 10 episodes
    print_count = 0

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
    episode_returns = []

    if os.path.exists(METRICS_PATH):
        metrics_file = open(METRICS_PATH, "a", newline="")
        metrics_writer = csv.writer(metrics_file)
    else:
        metrics_file = open(METRICS_PATH, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(["episode", "return", "average_return"])

    for episode in range(start_episode, total_episodes):
        obs = gym.reset()
        done = False
        episode_return = 0.0

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
            episode_return += reward

            # Store transition (tuple)
            replay_buffer.append((obs, action, reward, next_obs, done))

            agent.learn(target_network, replay_buffer)

            obs = next_obs

        # Decay epsilon per episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_returns.append(episode_return)
        average_return = sum(episode_returns) / len(episode_returns)
        metrics_writer.writerow([episode + 1, episode_return, average_return])
        metrics_file.flush()
        if print_count % 50 == 0:
            print(
                f"Episode {episode + 1}: return={episode_return:.2f}, "
                f"average return={average_return:.2f}"
            )
        print_count += 1

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
    metrics_file.close()
