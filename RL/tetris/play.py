import os
import time
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from env import Env
from model import TetrisDQN


CHECKPOINT_PATH = Path(__file__).resolve().parent / "tetris_checkpoint.pt"
STEP_DELAY_SECONDS = 0.15


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

    gym = Env(render_mode="human")
    observation_space = gym.get_observation_space()
    action_space = gym.get_action_space()
    input_size = sum(
        np.prod(observation_space[key].shape)
        for key in ("board", "active_tetromino_mask", "holder", "queue")
    )
    num_actions = action_space.n if hasattr(action_space, "n") else len(action_space)

    agent = TetrisDQN(
        input=input_size,
        hidden=64,
        output=num_actions,
        learning_rate=0.01,
        discount_factor=0.99,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    obs = gym.reset()
    done = False
    episode_return = 0.0
    gym.render()
    cv2.waitKey(1)

    with torch.no_grad():
        while not done:
            valid_actions = gym.get_valid_actions()
            if not valid_actions:
                raise RuntimeError("No valid macro actions are available before the episode ended.")
            output = agent(agent.process_obs(obs))
            valid_action_tensor = torch.tensor(valid_actions, dtype=torch.int64)
            action = valid_actions[torch.argmax(output[valid_action_tensor]).item()]
            obs, reward, terminated, truncated, _ = gym.step(action)
            episode_return += reward
            done = terminated or truncated
            gym.render()
            cv2.waitKey(1)
            time.sleep(STEP_DELAY_SECONDS)

    print(f"Episode return: {episode_return:.2f}")
    gym.close()
