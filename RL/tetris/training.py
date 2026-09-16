import os
import random
import csv
from collections import deque
from pathlib import Path
import numpy as np
import torch
from model import TetrisDQN
from env import Env

TETRIS_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = TETRIS_DIR / "tetris_checkpoint.pt"
METRICS_PATH = TETRIS_DIR / "tetris_metrics.csv"
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 50_000
EPSILON_DECAY = (EPSILON_MIN / EPSILON_START) ** (1 / EPSILON_DECAY_STEPS)
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = 20_000
REPLAY_WARMUP_STEPS = 5_000
TARGET_SYNC_STEPS = 2_000
CURRICULUM_STAGES = (
    (0, 1),       # Start one placement from a line clear.
    (15_000, 2),   
    (30_000, 3),
    (50_000, None),  # Continue from normal, empty-board starts.
)

def curriculum_placements_for_episode(episode: int) -> int | None:
    placements = None
    for start_episode, stage_placements in CURRICULUM_STAGES:
        if episode < start_episode:
            break
        placements = stage_placements
    return placements

if __name__ == "__main__":
    gym = Env(render_mode=None)
    action_space = gym.get_action_space()
    observation_space = gym.get_observation_space()
    input_size = sum(
        np.prod(observation_space[key].shape)
        for key in ("board", "active_tetromino_mask", "holder", "queue")
    )

    num_actions = action_space.n if hasattr(action_space, "n") else len(action_space)

    agent = TetrisDQN(input=input_size, hidden=64, output=num_actions, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)
    target_network = TetrisDQN(input=input_size, hidden=64, output=num_actions, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR)
    target_network.load_state_dict(agent.state_dict())

    # hyperparameters
    epsilon = EPSILON_START

    start_episode = 0
    total_episodes = 50000
    print_count = 0
    total_steps = 0

    # Resume from checkpoint if it exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        target_network.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE
        start_episode = checkpoint["episode"] + 1
        epsilon = checkpoint["epsilon"]
        total_steps = checkpoint.get("total_steps", 0)
        print(f"Resumed from episode {start_episode} with epsilon {epsilon:.4f}")

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    episode_returns = []

    if os.path.exists(METRICS_PATH):
        metrics_file = open(METRICS_PATH, "a", newline="")
        metrics_writer = csv.writer(metrics_file)
    else:
        metrics_file = open(METRICS_PATH, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(["episode", "return", "average_return", "lines_cleared"])

    for episode in range(start_episode, total_episodes):
        curriculum_placements = curriculum_placements_for_episode(episode)
        if curriculum_placements is None:
            obs = gym.reset()
        else:
            obs = gym.reset_n_placements_from_line_clear(curriculum_placements)
        done = False
        episode_return = 0.0
        episode_lines_cleared = 0

        while not done:
            valid_actions = gym.get_valid_actions()
            if not valid_actions:
                raise RuntimeError("No valid macro actions are available before the episode ended.")

            # Explore or select the highest-valued valid final placement.
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    output = agent(agent.process_obs(obs))
                    valid_action_tensor = torch.tensor(valid_actions, dtype=torch.int64)
                    action = valid_actions[torch.argmax(output[valid_action_tensor]).item()]

            next_obs, reward, terminated, truncated, info = gym.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_lines_cleared += info.get("lines_cleared", 0)
            next_valid_actions = [] if done else gym.get_valid_actions()

            # Store transition (tuple)
            replay_buffer.append(
                (obs, action, reward, next_obs, done, next_valid_actions)
            )

            if len(replay_buffer) >= REPLAY_WARMUP_STEPS:
                agent.learn(target_network, replay_buffer)

            obs = next_obs
            total_steps += 1
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            if total_steps % TARGET_SYNC_STEPS == 0:
                target_network.load_state_dict(agent.state_dict())

        episode_returns.append(episode_return)
        average_return = sum(episode_returns) / len(episode_returns)
        metrics_writer.writerow(
            [episode + 1, episode_return, average_return, episode_lines_cleared]
        )
        metrics_file.flush()
        if print_count % 50 == 0:
            print(
                f"Episode {episode + 1}: return={episode_return:.2f}, "
                f"average return={average_return:.2f}, "
                f"lines cleared={episode_lines_cleared}"
            )
        print_count += 1

        # Save checkpoint periodically (every 50 episodes) and on final episode
        if (episode + 1) % 50 == 0 or (episode + 1) == total_episodes:
            checkpoint = {
                "episode": episode,
                "agent_state_dict": agent.state_dict(),
                "target_state_dict": target_network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "epsilon": epsilon,
                "total_steps": total_steps,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Saved checkpoint at episode {episode + 1} (epsilon: {epsilon:.4f})")

    gym.close()
    metrics_file.close()
