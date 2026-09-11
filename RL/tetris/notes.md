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

- agent.py
    - Contains the actual network
    - Public functions: forward(input) -> action, train(reward_from_action) -> updates weights

- train.py
    - Initializes the environment and the agent
    - Contains the step -> action -> train loop