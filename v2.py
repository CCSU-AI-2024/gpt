import torch as t
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # amount of sequences to go thru parallel processing
block_size = 8 # max context length for predictions
learning_rate = 1e-3
max_new_tokens = 500  # how many characters do we generate
max_iters = 5000 # how many times are we going to "learn" and decrease our loss
eval_interval = 300 # how often are we going to evaluate our loss
eval_iters = 200 # how many batch iterations we will take the average loss of
n_embd = 32 # number of embedding dimensions
dropout = 0.2
device = 'cuda' if t.cuda.is_available() else 'cpu'
n_layer = 6
n_head = 6
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
# input is a 32x8 tensor, each row is a random chunk of the training set
# target is also 32x8 tensor, containing the targets given the context of our input
# transformer will look up the correct char to predict using this in the other model

@t.no_grad()  # this just lets pytorch know that we don't call backward on this function
def estimate_loss():
    out = {}
    m.eval()
    for split in [training,validation]:
        losses = t.zeros(eval_iters)
        for k in range(eval_iters):
            input, target = get_batch(split)
            logits, loss = m(input, target)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', t.tril(t.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout) # dropout randomly drops, or zeroes, some elements during training
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T), softmax is normalization operation.Rescales input tensors, so output tensors lie in range [0,1] and sum to 1
        wei = self.dropout(wei)
        
        v = self.value(x) # (B,T,hs)
        out = wei @v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
      
class MultiHeadAttention(nn.Module):
    #multiple self attention heads running in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = t.cat([h(x) for h in self.heads], dim=-1) #concatenates all of the outputs. 
        out = self.dropout(self.proj(out)) #applies projection
        return out
        
class FeedFoward(nn.Module):
    #single layer linear, followed by a relu non-linearity
    #if self attention is communication, this is thinking upon it individually
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    #intersperses communication and computation 
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #communication
        self.ffwd = FeedFoward(n_embd) #computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #communication
        x = x + self.ffwd(self.ln2(x)) #computation
        return x

#Simplest possible neural network, the Bigram Language Model
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            # every int in our input will refer to this embedding table ^, and get the corresponding row
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.self_attention_head: Head = Head(n_embd)
        self.feed_forward = FeedFoward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input, targets=None):
        B, T = input.shape

        token_embds = self.token_embedding_table(input) # essentially the predictions for the next character in the sequence
        # Batch x Time x Channel (characters) tensor
        position_embds = self.position_embedding_table(t.arange(T, device=device))
        embds_sum = token_embds + position_embds
        embds_sum = self.self_attention_head(embds_sum)
        embds_sum = self.feed_forward(embds_sum)
        logits = self.lm_head(embds_sum)

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
            input_cond = input[:, -block_size:]
            logits, loss = self(input_cond) # refers to forward function
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_inp = t.multinomial(probs, num_samples=1)
            input = t.cat((input, next_inp), dim=1) # whatever is generated gets concatenated with previous input
        return input

model = GPTLanguageModel()
m = model.to(device)

# training the model to not be random
optimizer = t.optim.AdamW(m.parameters(), lr = learning_rate) # try experimenting with the learning rate

for steps in range(max_iters):
    # every so often, evaluate the loss on the training and validation sets
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses[training]:.4f}, val loss {losses[validation]:.4f}")
    input, target = get_batch(training)  # sample a new batch of data

    logits, loss = m(input, target)  # evaluate the loss
    optimizer.zero_grad(set_to_none=True)  # zero out from previous step
    loss.backward()  # getting gradients of all the parameters
    optimizer.step()  # using those gradients to update our parameters

# generate from the BigramLanguageModel
context = t.zeros((1,1), dtype = t.long)  # start generating from a single 0
print(decode(m.generate(context, max_new_tokens)[0].tolist()))