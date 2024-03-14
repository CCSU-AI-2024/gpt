import torch as t

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
      slef.query = nn.Linear(n_embd, head_size, bias=False)
      self.value = Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    self.dropout = nn.Dropout(dropout) # dropout randomly drops, or zeroes, some elements during training
    
    def forward(self, x):
      B,T,C = x.shape
      k = self.key(x) # (B,T,hs)
      q = self.query(x) # (B,T,hs)
      
      wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
      wei = F.softmax(wei, dim=-1) # (B, T, T)
      wei = self.dropout(wei)
      
      v = self.value(x) # (B,T,hs)
      out = wei @v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
      return out