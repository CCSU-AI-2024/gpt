import torch as t
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
block_size = 8 #max context length for predictions
n_embd = 32 #number of embedding dimensions
n_head = 2 #the number of heads we'd like
dropout = .2 #used as parameter for dropout. represents probability of an element to be zeroed

#---

with open('training_data.txt', 'r', encoding='utf-8') as f:
    shakespeare_data: str = f.read()

unique_chars: list = sorted(list(set(shakespeare_data)))

str_to_int: dict = { char:i for i, char in enumerate(unique_chars) }
int_to_str: dict = { i:char for i, char in enumerate(unique_chars) }

def encode(string: str) -> list:
    return [str_to_int[char] for char in string]

def decode(ints: list) -> str:
    return ''.join([int_to_str[i] for i in ints])

encoded_data: t.Tensor = t.tensor(encode(shakespeare_data), dtype = t.long)

train_size: int = int(0.9 * len(shakespeare_data))
training_data: t.Tensor = encoded_data[:train_size]
validation_data: t.Tensor = encoded_data[train_size:]


class Head(nn.Module):

    def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', t.tril(t.ones(block_size, block_size)))
      self.dropout = t.Dropout(dropout) # dropout randomly drops, or zeroes, some elements during training
    
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