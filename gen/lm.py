import torch
import torch.nn as nn
import random
import math

class Transformer(nn.Module):
    def __init__(self, embed_dim, vocab):
            super().__init__()
            self.vocab = vocab
            self.embed_dim = embed_dim
            self.embed = nn.Embedding(len(vocab), embed_dim)
            self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_vocab = nn.Linear(embed_dim, len(vocab), bias=False)

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
   
            scores = [s / math.sqrt(len(queries[i])) for s in scores]
            weights = torch.softmax(torch.stack(scores), dim=0)

            new_vec = torch.zeros_like(values[0])
            for weight, val_vec in zip(weights, values):
                new_vec += weight * val_vec

            outputs.append(new_vec)
            
        return outputs
    
    def get_logits(self, outputs):
        last_vec = outputs[-1]                 
        return self.W_vocab.forward(last_vec)  

    def get_next_token(self, outputs):
        logits = self.get_logits(outputs)
        probs = torch.softmax(logits)
        choice_idx = random.choices(range(len(self.vocab)), weights=probs.detach().tolist(), k=1)[0]
        return self.vocab[choice_idx]

    def predict_logits(self, token_ids):
        input_vecs = list(self.embed(token_ids))
        return self.get_logits(self.forward(input_vecs))

    @torch.no_grad()
    def generate(
        self,
        token_ids,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        max_context: int = 128,
        stop_token: str = "<eos>",
    ) -> list[int]:
        ids = torch.as_tensor(token_ids, dtype=torch.long)

        for _ in range(max_new_tokens):
            logits = self.predict_logits(ids[-max_context:])

            if temperature <= 0:
                next_id = logits.argmax()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_k is not None:
                    cutoff = torch.topk(probs, min(top_k, len(probs))).values[-1]
                    probs = probs.masked_fill(probs < cutoff, 0.0)
                next_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_id.reshape(1)])
            if self.vocab[int(next_id)] == stop_token:
                break

        return ids.tolist()

    def train_step(self, input_ids, expected_token_idx, optimizer):
        optimizer.zero_grad()

        losses = [
            nn.functional.cross_entropy(self.predict_logits(ids), target)
            for ids, target in zip(input_ids, expected_token_idx)
        ]
        loss = torch.stack(losses).mean()

        loss.backward()
        optimizer.step()

        return loss.item()
