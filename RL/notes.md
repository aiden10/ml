I'm more familiar with this kind due to my previous attempt at making an AI for Super Smash Bros. While it didn't really work, it wasn't too bad from a learning perspective. It provided me with the fundamentals. 

# Strategies
These are still built upon the same neural network foundation. There seems to be no single best strategy. You have to mix and match them depending on your use-case. 

## Value Based
### Q-Learning
This is what I tried with SSBM. Basically you do an action, calculate a reward/loss based on the result of the action and the probability of doing that action is updated based on the result. And you feed the current game state into the model and it outputs a controller input. Well, more accurately, it would output the expected Q-value that would be obtained from performing each of those actions. Additionally you use a special formula. This formula updates the probabilities based on immediate reward, uses a learning rate, and a discount factor. 

Because Q-learning involves the next state's expected Q-value, it allows desirable sequences of actions to be learned.
State A -> move1 -> State B -> move2 -> State C -> move3 -> reward earned.
Only the Q value of move3 from State C is updated. However, the next time that the agent finds itself in state B and does move2, it still gets no immediate reward, but it inherits some of the new state's expected Q-value. Now making move2 more valuable from state B. And the process would repeat.  

## Policy Based
Outputs a list of probabilities for actions. Still operates in a similar way. You do a random roll for picking the action, and then you update the odds based on the result.

# Challenges
No immediate results makes it hard to know what to update. If there were hundreds of actions that led to a result, it can be hard to say which actually contributed.

Large, complex action spaces. Chess, for example has many possible moves from any given position. But the issue is that each board state, even if it's only different by one tile, can completely change what moves are considered good. 

# Backpropagation
Unlike with supervised learning, there is no expected value for each of the model's outputs. In value based strategies, the "expected value" is the expected value/reward that would be gained from performing that action. For policy based strategies, the randomly chosen action is performed, and then based on the result, that action's probability is tweaked, either lowering or increasing it.