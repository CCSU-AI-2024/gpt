import torch as t
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 4 # amount of sequences to go thru parallel processing
block_size = 8 # max context length for predictions
#---

t.manual_seed(1337) # setting manual seed ensures rng is reproducible

with open('training_data.txt', 'r', encoding='utf-8') as f:
    shakespeare_data: str = f.read()

unique_chars: list = sorted(list(set(shakespeare_data)))
vocab_size = len(unique_chars)

str_to_int: dict = { char:i for i, char in enumerate(unique_chars) }
int_to_str: dict = { i:char for i, char in enumerate(unique_chars) }

def encode(string: str) -> list:
    return [str_to_int[char] for char in string]

def decode(ints: list) -> str:
    return ''.join([int_to_str[i] for i in ints])

encoded_data: t.Tensor = t.tensor(encode(shakespeare_data), dtype = t.long)
train_size: int = int(0.9 * len(shakespeare_data))
training: t.Tensor = encoded_data[:train_size] # 90% of data goes to training
validation: t.Tensor = encoded_data[train_size:] # 10% of data goes to validation

def get_batch(data: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    random_start: t.Tensor = t.randint(len(data) - block_size, (batch_size, 1)) # ensures batches contain different segments of dataset
    input: t.Tensor = t.stack([data[i:i + block_size] for i in random_start])
    target: t.Tensor = t.stack([data[i + 1:i + block_size + 1] for i in random_start])

    return (input, target)

input, target = get_batch(training) # input is a 4x8 tensor, each row is a random chunk of the training set
# target is also 4x8 tensor, containing the targets given the context of our input
# transformer will look up the correct char to predict using this

#Simplest possible neural network, the Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
            # every int in our input will refer to this embedding table, and get the corresponding row

    def forward(self, input, targets=None):
        logits = self.token_embedding_table(input) # essentially the predictions for the next character in the sequence
        # Batch x Time x Channel (characters) tensor

        if targets is None:
            loss = None
        else:
            # reshaping logits and targets for cross_entropy
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # makes logits 2d instead of 3d
            targets = targets.view(B*T) # makes targets 1d instead of 2d
            loss = F.cross_entropy(logits, targets) # measures the quality of the logits with respect to the targets
        return logits, loss

    def generate(self, input, max_new_tokens): # takes the input (context) and expand it using the prediction
        for _ in range(max_new_tokens):
            logits, loss = self(input) # refers to forward function
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_inp = t.multinomial(probs, num_samples=1)
            input = t.cat((input, next_inp), dim=1) # whatever is generated gets concatenated with previous input
        return input

m = BigramLanguageModel(vocab_size)
logits, loss = m(input, target)
print(logits.shape)
print(loss)

idx = t.zeros((1,1), dtype=t.long) # start generating from a single 0
print(decode(m.generate(idx, max_new_tokens = 100)[0].tolist()))
