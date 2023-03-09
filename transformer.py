"""
 Simple Implementation of Attention is all you need, 
 followed along/created with Karpathy's video working through it
 applied to my Latin Corpus
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from Data import dataExp
import os 

# hyperparameters
batch_size = 64 
context_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
# every head is 64 dim, 384/6 = 64
n_head = 6
n_layer = 6
dropout = .2
# ------------

torch.manual_seed(1337)

# load in the data
CI = dataExp.CorpusInterface(corpus_name="text_corpus.pickle", shouldTokenize = False)
text = CI.get_total_data().replace("\t","")

chars = sorted(list(set(text)))
vocab_size = len(chars)

# We just create a mapping between our character vocabulary
# and their corresponding integer value, and define lambda funcs to do this mapping for us
stoi = {ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Let's create a train/val split
n = int(.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    # create a batch by context size tensor of the data
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
            This performs the self-attention that we worked through did in the notebook
        """
        batch, time, channel = x.shape
        
        k = self.key(x) # (batch, time, head_size)
        q = self.query(x) # (batch, time, head_size)

        # compute the affinities/scaled attention scores
        weights = q @ k.transpose(-2, -1) * channel **-.5
        # don't want to interact with subsequent time step tokens
        # i.e. makes it a decoder block
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
        # make the probability nicely distributed
        weights = F.softmax(weights, dim=-1) # (batch, time, time)
        weights = self.dropout(weights)
        # weighted aggreagation of values
        v = self.value(x)
        out = weights @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ just a simple linear layer followed by non-linearity, as there 
        was a FF part in the paper too
        
        on a per-token layer
    """ 
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        # last nn.Linear is the projectin layer back into residual pathway
    
    def forward(self, x):
        return self.net(x)
        

class Block(nn.Module):
    """ Transformer block: communication intersperesed with calculation """
    
    def __init__(self, n_embd, n_head):
        """ n_embd: embedding dimension, n_head: the number of heads we want """
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd= FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # make them residual connnections by doing x + 
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        batch, time = idx.shape

        # idx and targets are both (batch, time) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (batch, time, channel)
        pos_emb = self.position_embedding_table(torch.arange(time, device=device)) # (time, channel)
        x = tok_emb + pos_emb # (batch,time, channel)
        x = self.blocks(x) # (batch,time, channel)
        x = self.ln_final(x) # (batch,time, channel)
        logits = self.lm_head(x) # (batch,time,vocab_size)

        if targets is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch*time, channel)
            targets = targets.view(batch*time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (batch,time) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last context_size tokens
            idx_cond = idx[:, -context_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (batch, channel)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (batch, channel)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (batch, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (batch, time step+1)
        return idx

model = LanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('Results/lat_transformer_pred.txt', 'w+').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
torch.save(model.state_dict(), os.getcwd()+"/LatinTransformer/")