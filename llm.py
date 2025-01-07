import requests
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config_dict = json.load(f)

# Convert dictionary to Config object
config = Config(config_dict)


# The Complete Works of William Shakespeare
try:
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    response = requests.get(url)
    text = response.text
    with open('data.txt', 'w', encoding='utf-8') as f:
        f.write(text)
print(f"Read {len(text)} characters")


# Create vocab
vocab = sorted(list(set(text)))
config.vocab_size = len(vocab)
print(f"Vocabulary size: {len(vocab)}")

# Create encode and decode functions
encode = lambda s: [vocab.index(c) for c in s]
decode = lambda l: ''.join([vocab[i] for i in l])

# encode the text
encoded = encode(text)

# Prepare data
data = torch.tensor(encoded, dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# Create a model
class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_size = config.hidden_dim // config.n_heads
        self.key = nn.Linear(config.hidden_dim, self.head_size, bias=False)
        self.query = nn.Linear(config.hidden_dim, self.head_size, bias=False)
        self.value = nn.Linear(config.hidden_dim, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_heads * config.hidden_dim // config.n_heads, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.register_buffer('pe', self.position_embedding(config.block_size))
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def position_embedding(self, T):
        pos = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        
        # Calculate positional encodings using sine/cosine
        dim = torch.arange(0, self.config.hidden_dim, 2, dtype=torch.float)
        div = torch.exp(-np.log(10000.0) * dim / self.config.hidden_dim)
        pe = torch.zeros(T, self.config.hidden_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        
        return pe.unsqueeze(0)

    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.embedding(x)
        x += self.pe[:, :T]

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = TLM(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'mps' if torch.backends.mps.is_available() else device
model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


# Training
def get_batch(data, config):
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for data, split in [(train_data, 'train'), (val_data, 'val')]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

for iter in range(config.max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch(train_data, config)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# Save model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved to model_weights.pth")

# generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))