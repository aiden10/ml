import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from env import Env
from model import TetrisAfterstateValue


CHECKPOINT_PATH = Path(__file__).resolve().parent / "tetris_afterstate_checkpoint.pt"
STEP_DELAY_SECONDS = 0.15


if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

    gym = Env(render_mode="human")
    observation_space = gym.get_observation_space()
    input_size = sum(
        np.prod(observation_space[key].shape)
        for key in ("board", "active_tetromino_mask", "holder", "queue")
    )
    agent = TetrisAfterstateValue(
        input=input_size,
        hidden=64,
        learning_rate=0.001,
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
            candidate_outcomes = gym.simulate_all_valid_placements()
            if not candidate_outcomes:
                raise RuntimeError("No valid macro actions are available before the episode ended.")
            scores = agent.score_afterstates(candidate_outcomes)
            action = candidate_outcomes[torch.argmax(scores).item()][0]
            obs, reward, terminated, truncated, _ = gym.step(action)
            episode_return += reward
            done = terminated or truncated
            gym.render()
            cv2.waitKey(1)
            time.sleep(STEP_DELAY_SECONDS)

    print(f"Episode return: {episode_return:.2f}")
    gym.close()
