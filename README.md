# About
trying to learn about machine learning

## Experiments
### MNIST Digit Classifier 
Decided to start with the "Hello World" of ML. I made it from scratch in Python and it worked but felt a bit underwhelming?

### Tetris RL
This one I spent quite a bit of time on, and tried quite a few different strategies but still can't seem to get to actually play Tetris.

Things I tried, in chronological order (later experiments build on earlier ones):
- **Initial DQN:** Started with Q-learning using a replay buffer and a target network, with the board and active piece as inputs. Added checkpointing and training metrics, plus a way to watch the agent play.
- **Rewards and exploration:** Compared the environment's default reward with my custom reward for score/line clears and board holes. Tried a lower learning rate and slower epsilon decay to encourage exploration, adjusted the hole penalty, lowered the discount factor, and randomized the seed each episode. I also found that the held piece and queue were missing from the model's inputs and added them.
- **Macro actions:** Replaced individual button presses with a placement choice: select the number of rotations and a column, then hard drop. Faster epsilon decay did not help. I also tried a board-quality reward based on height, unevenness, and holes, and switched to a fixed-capacity replay buffer.
- **Replay training adjustments:** Increased replay-buffer capacity and target-network sync interval, and added a warmup period so training began after the buffer had filled. These changes did not solve the line-clearing problem.
- **Reverse curriculum:** Started training from boards close to a line clear, then gradually moved to less-complete boards and normal starts. This helped the agent clear more lines in easier states, but it did not reliably clear lines from regular board states. I considered prioritizing rare line-clear experiences in replay, but did not implement that experiment.
- **Evaluate possible placements and use afterstates:** Simulated every valid placement from the current board and evaluated the resulting states, instead of estimating a value for each action directly. The latest version uses an afterstate value network: it scores each post-placement board using its immediate reward plus the discounted value of that resulting state, then trains toward the best candidate outcome. This was better than earlier approaches, but still did not consistently clear lines.
