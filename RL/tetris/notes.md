# Setup
Gym will be gym-tetris. Runs an emulator playing the original Tetris. 

## Input
Because gym-tetris doesn't directly expose the game state, I need to read the screen. Reading the screen can be done in a few different ways. Simplest way is to feed every pixel into the network as the inputs. Not sure what the resolution is, but it's likely going to be a lot. Could crop it or downscale it though. There's only a few parts that matter; the board and the next piece. Alternatively, I could use a CNN for the image processing part and feed its output into a separate agent model. Or I just combine them and have initial convolutional layers which feed into linear layers. That latter strategy is apparently more common. Colors are also not relevant for Tetris so to not need to look at RGB, I can convert the image to black and white first. And also combine that with the cropping.

## Output
Outputs will be the possible button presses. Tetris doesn't really require any continous actions, and I don't think it's ever optimal to wait either. So the only options are going to be button presses.

## Strategy
Since Tetris doesn't require any continous actions, a value based approach seems appropriate. Only issue is that rewards are based not on single actions, but based on chains of actions.

### Rewards
A reward would be granted for getting a higher score. Actually I might now need OCR for that since I don't think that score is going to be exposed directly as a number. 
Another unique trait about Tetris is that losing is actually going to be related to the most recent moves you did. I think there's almost always a way to recover, even when the pieces are stacked very close to the top. So losses can heavily penalize the n most recent actions?  
It might actually be even simpler, and I might not even need to consider sparse rewards like losses. I think the only two things I need are a punishment for covering gaps and a reward for earning points. 

## Backprop
Should be standard method. Only difference will be calculating "loss" since it'll be based on the Q-learning formula instead.

## Model
If I'm not analyzing the pixels, I can skip the CNN layers. Then the input will just be the board state, current piece(?), and next piece data read straight from memory. Unsure about how many hidden layers I should have and how many nodes per layer. These layers will be fully connected. Output layer will have:
- D-Pad left
- D-Pad right
- D-Pad down
- A button
- B button
- Select button
As the output nodes/actions. 

# Plan
- Get environment setup
- Implement model structure, layers, forward pass, and map outputs to actions
- Read memory, feed model output into environment
- Setup rewards based on game memory, need to know where score is located and parse the board state so I can detect gaps
- Handle restarting on loss
- Integrate rewards into training loop

## Flow
- Read memory, extract score, board state, and next piece info
- Feed that data into the model
- Forward pass to get output action
- Perform action
- Analyze memory to determine rewards. Did score go up? Did any gaps get covered?
- Calculate the loss and backprop

## Files
- env.py
    - Runs the environment
    - Functions which expose game state as Python objects
    - Step function
    - Function which takes in an input and performs it

- model.py
    - Contains the actual network
    - Public functions: forward(input) -> action, train(reward_from_action) -> updates weights

- train.py
    - Initializes the environment and the agent
    - Contains the step -> action -> train loop

# Notes
I couldn't get the NES Tetris gym library to work, so I am just going to use a pure simulation one instead. This environment exposes:
- Observation Space: 
    - 'active_tetromino_mask': Box(0, 1, (24, 18), uint8), 
    - 'board': Box(0, 9, (24, 18), uint8), 
    - 'holder': Box(0, 9, (4, 4), uint8), 
    - 'queue': Box(0, 9, (4, 16), uint8)
- Action Space: Discrete(8)
    - 1: Do nothing
    - 2: Move left
    - 3: Move right
    - 4: Rotate clockwise
    - 5: Rotate counter-clockwise
    - 6: Soft drop
    - 7: Hard drop
    - 8: Swap

active_tetromino_mask: all 0s and 1s. 1 represents a tile where a block exists, and 0 means it's empty. Importantly, it only shows the moving pieces.
board: a filled tile represents a piece is there. The number determines the kind of piece it is.
holder: the block that is currently held
queue: the next pieces in queue

Well at least I don't need to worry about reading memory to extract the info.

Training aspect is a bit confusing. The model only outputs expected Q values for every possible action. But all I have to update the weights is the reward gained from the action. I can compare the expected Q value and the actual reward that was gained. What's confusing is getting the max Q value of the next state. Would it require another forward pass? Like:
- Get expected Q value for each action
- Perform action with highest Q value
- Calculate reward
- Do another forward pass with the input being the env's state after performing the action
- Now you have all the data to calculate the target and loss
So the target is what you want the current state's Q value be. It's the next state's max Q-value + the reward that was obtained from doing the current state's chosen action. Also Q value represents the total sum of all future rewards from some point. So we add rewards to the next state. 

Apparently there are two major issues.
1. The differences between each step will basically be identical. This is bad because there's too much variance and not enough has actually changed. That variance might mean losing actual progress. By variance I mean too many steps and weight update cycles. Actually the issue is things getting "lost". Rare events wouldn't be able to be effectively learned because they would just get forgotten.
2. Using the same weights for both passes means both get updated. Updating the present and future states means moving the target moves and the "present" also moves.

To fix issue 1, I can use a buffer. The buffer stores the previous n state, action, reward, and next state states and the model periodcally randomly samples that buffer to use for training. Rather than operating in a continuous: forward -> do action -> train loop.

For issue 2, I can use two networks. The first is updated in real-time, but the second remains static and is only updated after a game finishes or maybe every few hundred steps. That frozen one would be the "target network". The one which fetches the future state's Q value.

Apparently pytorch accumulates weight gradients by default. I assume it's so that you wouldn't need to weight -= (gradient * learning_rate) tons of times and could instead just do it a single time. Although at the cost of the model only updating periodically. 
Pytorch optimizers are what handle updating the weights given the gradient. SGD does the approach I just did: weight -= (gradient * learning_rate). But apparently there are better options. Most common seems to be "Adam". Which probably does more advanced statistical calculations to determine how to update the weights. Apparently, it's actually more that you usually don't want to constantly update models. Usually you do it periodically which means accumulating gradients. 

Well it doesn't seem to have learned anything even after 10k episodes, it just lets the blocks pile up and dies. Inputs are being made though because I saw it spin some blocks. How do you debug AI models? It doesn't have any explicit problem, all you know is that it doesn't work as expected. To confirm if it's an issue with my rewards structure, I'll try doing training with the default reward from the gym environment. It seems slightly better, but still not great. Yeah after 5000 episodes it's slightly better? Still very bad, only difference is that it starts placing blocks to the left or right to avoid losing as quickly. Might be an issue with the exploration. It could be getting stuck in suboptimal states and not able to find the way out.
After almost 10k episodes, I can see that it is beginning to survive longer, but it's only suriving by avoiding stacking pieces vertically in the same spot. It hasn't learned to clear lines. And I'm not sure it can discover that anymore either.

Tried decreasing learning rate and making epsilon decay slower to have more exploration. Unfortunately neither of those changes seemed to have worked. The change which had the biggest impact was going back to my custom reward calculation, but changing the logic for the holes. Before I was calculating the amount of holes and multiplying that by 1.5. However, that meant that if there were 10 holes, 10 * 1.5 would be subtracted from the reward at every single step. But that change only brought me back to where the other reward system was at. It moves the pieces to avoid losing as fast, but still didn't learn to clear lines.

- Tried lowering discount factor a bit
- Now using random seed every episode
- Apparently I didn't even have the queue and held piece visible to the model

Now trying with macro actions. Instead of every action being a controller input, the possible actions are now just "one" bigger action. The agent specifies:
- Amount of rotations
- Which column to move to
And then the piece gets hard dropped.
This also didn't really help. Maybe now because there are fewer actions, the epsilon decays too slowly?
Quicker decay didn't make a difference. Now trying with changed reward structure which considers the board quality based unevenness, height and holes. Also trying with a replay buffer with a fixed size.   

Trying again with: 
- increased replay buffer size
- increased target sync step count
- a "warmup stage" so the learning only begins after the replay buffer has filled up enough

However, none of these changes were successful. Some other things I want to try:
- "Reverse curriculum", starting with states close to what you want and then working backwards. In this case it would be starting with boards which are maybe only one piece away from having a line cleared. Then you move on to boards which are less complete. 
- When a line clear happens, it should be prioritized in the replays since it's so rare. A basic way of doing something like this would be adding the rare clears into the replay buffer multiple times. Or giving each one a probability/priority of some kind.

Reverse curriculum did work for clearing more lines. Though it wasn't able to consistently clear a line in the single line board states and failed with subsequent, harder board states. Might just need more training time. If this, plus the replay priority still fails, I will probably move to evaluation of every possible piece placement. It is better, but still not able to clear lines when it goes to regular board states. Interestingly, despite not clearing more lines, the average return continues to get higher. I wonder when it would reach the limit.
Pretty interesting results after 50k episodes, didn't manage to ever truly learn how to clear lines properly, but it definitely seemed a bit better. Weird because it saw a huge spike in lines cleared episodes 31500-32000, and then it just immediately went back down. Average return did still continue to climb. 

Evaluating possible actions was still not doing too good. Better but still not consistently getting to the point of clearing lines. So now I'm wondering if the structure of my network is wrong.