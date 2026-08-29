import random 
import json
import math

class Node:
    def __init__(self, input_size: int = 0, weights: list[float] = [], bias: float = 0.0):
        limit = math.sqrt(2.0 / input_size) if input_size > 0 else 1.0
        self.weights = [random.uniform(-limit, limit) for _ in range(input_size)] if not weights else weights
        self.bias = 0.0 if not bias else bias
        self.last_inputs = []
        self.pre_activation = 0
        self.delta = 0
        
    def activation(self, input: float) -> float:
        # ReLU
        return input if input > 0 else 0.01 * input

    def activate(self, inputs: list) -> float:
        out = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.last_inputs = inputs
        self.pre_activation = out
        return self.activation(out)

def relu_derivative(z: float):
    return 1 if z > 0 else 0.01

class Network:
    """
    I guess the key thing to note is that each hidden layer node takes in the outputs of every node from the previous layer as its inputs.
    The first number in layers_def is basically just used as the input size definition.  
    """
    learning_rate = 0.001

    # The first and last layers in layers_def are the input and output.
    # [2, 4, 2] would represent a network with 2 input nodes, 4 hidden layer nodes, and 2 output nodes 
    def __init__(self, layers_def: list[int] = [], load_path: str = ""):
        self.layers = []
        if not load_path:
            for i in range(1, len(layers_def)):
                layer = [Node(layers_def[i - 1]) for _ in range(layers_def[i])]
                self.layers.append(layer)
            return

        saved_layers = []
        with open(load_path, "r", encoding="utf-8") as f:
            saved_layers = json.load(f)
        
        for layer in saved_layers:
            nodes = []
            for node_data in layer:
                nodes.append(Node(input_size=0, weights=node_data["weights"], bias=node_data["bias"]))
            
            self.layers.append(nodes)
        
    def forward_pass(self, inputs: list[float]):
        current_input = inputs
        for layer in self.layers:
            outputs = []
            for node in layer:
                outputs.append(node.activate(current_input))
            current_input = outputs
        
        return current_input

    def calculate_loss(expected: list[float], actual: list[float]) -> float:
        # Mean Squared Error
        n = len(expected)
        return (sum((a - e)**2 for a, e in zip(actual, expected))) * 1/n
    
    def backprop(self, expected: list[float], actual: list[float]):
        """
        derivative of loss function with respect to a particular weight
        loss function depends on "actual output", which is dependent on the outputs of each node 
        
        I'll have to go over each node and update their weights and biases. Doing so requires the gradient of every node's weight and bias.
        
        ∂L/∂w = . How do I do the partial derivative of MSE? Would the entire summation be written out? 
        With real numbers in a network with just 2 nodes and 2 outputs:
        inputs = [1, 2]
        w1 = [0.1, 0.2]
        b1 = 0.5
        w2 = [0.4, 0.5]
        b2 = 0.1
        output = [1, 1.5]
        expected = [3, 4]
        MSE: 1/2 * ((1 - 3) + (1.5 - 4))^2
        MSE is my loss function. I have to take the partial derivative of it with respect to a weight or bias. I have to break the above
        down further into more of its base components until I reach the part with that weight.
        MSE = 1/2 * ((out1 - 3)^2 + (out2 - 4)^2)
        So what is the function for the output?
        out1 = ReLU((1 * 0.1 + 2 * 0.2) + 0.5) 
        out2 = ReLU((1 * 0.4 + 2 * 0.5) + 0.1) 
        Now I can rewrite my MSE out fully, and sub in the actual values except for the weight or bias I'm doing the derivative with respect to.
        MSE = 1/2 * ((ReLU((1 * w1[0] + 2 * 0.2) + 0.5)  - 3)^2 + (ReLU((1 * 0.4 + 2 * 0.5) + 0.1)  - 4)^2)
        Although, for it to be a parital derivative, shouldn't it have other variables? If there's only one isn't it just a regular derivative? 
        Well apparently it is correct.

        The second part can be dropped because it contains no variable and would just become 0.
        So I only have to do the derivative of:
        1/2 * ((ReLU((1 * w1[0] + 2 * 0.2) + 0.5) - 3)^2
        This now contains 3 separate parts:
            - z = 1 * w1[0] + 2 * 0.2 + 0.5
            - a = ReLU(z)
            - L = 1/2 * (a - 3)^2

        Have to do the chain rule, differentiate each part, and multiply them together to get ∂L/∂w1[0].
        ∂z/∂w1[0] = 1. Basically like d/dx of x + 2 or any other constant
        ∂a/∂z = 1 if z > 0 else 0
        ∂L/∂a = a - 3
        Now I know ∂L/∂w1[0]:
        = (a - 3) * [1 if z > 0 else 0] * 1
        a is going to be the output value and z is node's sum before ReLU was applied.

        This happens after a forward pass. I have to calculate the deltas and gradients for every node. 
        The delta is ∂a/∂z * ∂L/∂a. The delta of each node should be saved in each node. 
        
        What are all the values I need other than the weight and bias?
        - output
        - expected output
        - loss
        - inputs every node received?
        ReLU derivative is really simple.
        Only one of the outputs need to be considered (the one with the weight or bias we do the derivative with respect to)
        And then I multiply them together and save the delta. Wait I forgot why I save the delta.
        """
        for layer_idx in reversed(range(len(self.layers))):
            current_layer = self.layers[layer_idx]
            if layer_idx == len(self.layers) - 1:
                for i, node in enumerate(current_layer):
                    grad_a = actual[i] - expected[i]
                    node.delta = grad_a * relu_derivative(node.pre_activation)
            
            else:
                next_layer = self.layers[layer_idx + 1]
                for i, node in enumerate(current_layer):
                    grad_a = 0
                    for next_node in next_layer:
                        grad_a += next_node.delta * next_node.weights[i]
                        
                    node.delta = grad_a * relu_derivative(node.pre_activation)
                
        for layer in self.layers:
            for node in layer:
                for i in range(len(node.weights)):
                    node.weights[i] -= self.learning_rate * node.delta * node.last_inputs[i]
                node.bias -= self.learning_rate * node.delta
        
    def train(self, epochs: int, inputs: list[list[float]], expected: list[list[float]]):
        training_results = []
        correct = 0
        incorrect = 0
        
        n = len(inputs)
        for epoch in range(epochs):
            for i in range(n):
                result = self.forward_pass(inputs[i])
                loss = Network.calculate_loss(expected[i], result)
                training_results.append({"loss": loss, "result": result, "expected": expected[i]})
                self.backprop(expected=expected[i], actual=result)
                predicted_label = result.index(max(result))
                expected_label = expected[i].index(max(expected[i]))
                
                if predicted_label == expected_label: correct += 1
                else: incorrect += 1
                    
            print(f'epoch: {epoch+1}/{epochs}')
            
        # Save weights/biases and training results to json file
        network_record = []
        for layer in self.layers:
            nodes_in_layer = [{"weights": node.weights, "bias": node.bias} for node in layer]
            network_record.append(nodes_in_layer)
        
        with open("classifier/results/training_results.json", "w", encoding="utf-8") as tr:
            json.dump(training_results, tr, indent=4)
        
        with open("classifier/results/network_record.json", "w", encoding="utf-8") as mr:
            json.dump(network_record, mr, indent=4)
        
        print(f"results: {correct}/{correct + incorrect}")
        print(f"accuracy: {round(correct/(correct + incorrect), 2) * 100}%")