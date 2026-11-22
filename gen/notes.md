# Generative AI
To my understanding, these usually involve outputting one small piece and feeding that output back into itself as new input. Networks which do this are known as recurrent neural networks.

## LLMs
Revolve around the transformer architecture.

A transformer takes in a vector, and outputs a token. Each transformer also has three linear layers: the query, key, and value weights. 

### Pseudocode
```python

class Transformer:
    def __init__(self, embed_dim):
            self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_vocab = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_vecs: list[float]):
        queries, keys, values = [], [], []
        for vec in input_vecs:
            queries.append(self.W_q.forward(vec))
            keys.append(self.W_k.forward(vec))
            values.append(self.W_v.forward(vec))

        outputs = []
        for i in range(len(input_vecs)):
            scores = []
            for j in range(len(input_vecs)):
                score = queries[i].dot(keys[j])
                scores.append(score)
   
            scores = [s / sqrt(len(queries[i])) for s in scores]
            weights = softmax(scores)

            new_vec = zeros_like(values[0])
            for weight, val_vec in zip(weights, values):
                new_vec += weight * val_vec

            outputs.append(new_vec)
            
        return outputs
    
    def get_logits(self, outputs):
        last_vec = outputs[-1]                 
        return self.W_vocab.forward(last_vec)  

    def get_next_token(self, outputs):
        logits = self.get_logits(outputs)
        probs = softmax(logits)
        choice_idx = random.choices(range(len(vocab)), weights=probs, k=1)[0]
        return vocab[choice_idx]

    def train(self, input_vecs, expected_token_idx):
        outputs = self.forward(input_vecs)
        logits = self.get_logits(outputs)
        
        loss = cross_entropy_loss(logits, expected_token_idx)
        
        loss.backward()
        optimizer.step()
```

### Training
Requires quite a bit of work for the data too.

Get random samples of text and try to predict the next word/token.
Need to save checkpoints.   