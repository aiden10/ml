import os
import time

import cv2
import numpy as np
import torch

from env import Env
from model import TetrisDQN


CHECKPOINT_PATH = "tetris_checkpoint.pt"
STEP_DELAY_SECONDS = 0.15


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

    gym = Env(render_mode="human")
    observation_space = gym.get_observation_space()
    action_space = gym.get_action_space()
    input_size = (
        np.prod(observation_space["board"].shape)
        + np.prod(observation_space["active_tetromino_mask"].shape)
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
            action = torch.argmax(agent(agent.process_obs(obs))).item()
            obs, reward, terminated, truncated, _ = gym.step(action)
            episode_return += reward
            done = terminated or truncated
            gym.render()
            cv2.waitKey(1)
            time.sleep(STEP_DELAY_SECONDS)

    print(f"Episode return: {episode_return:.2f}")
    gym.close()
