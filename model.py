import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads along the feature dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity and projection back """
    def __init__(self, n_embd, dropout):
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
    """ Transformer block: communication (attention) followed by computation (feed-forward) """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-norm formulation with residual (skip) connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    
class Gpt(nn.Module):
    def __init__(self,vocab_size,block_size,n_embd,n_head,n_layer,dropout):
        super().__init__()
        self.block_size=block_size
        
        # 1 Tokens and position embeddings 
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        
        # 2 Stack of Tansformer blocks 
        self.blocks=nn.Sequential(*[
            Block(n_embd,n_head,block_size,dropout) for _ in range(n_layer)
        ])
        
        # 3 Final layer Norm and output Linear Head 
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
        
        # Initialize weights Gpt 
        self.apply(self._init_weights)
        
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            
    def forward(self,idx,targets=None):
        B,T=idx.shape
        
        assert T<=self.block_size,f"Cannot forward sequence of lenght {T}, Block_size {self.block_size} "
        tok_emb=self.token_embedding_table(idx)
        
        pos_emb=self.position_embedding_table(torch.arange(
            T,device=idx.device))
        x=tok_emb+pos_emb    
        x=self.blocks(x)
        x=self.ln_f(x)
            
        # compute logits 
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape    
            logits_flat=logits.view(B*T,C)
            targets_flat=targets.view(B*T)
            loss=F.cross_entropy(logits_flat,targets_flat)
            
        return logits,loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Takes a conditioning sequence of indices idx (B, T) and completes 
        it by sampling new tokens max_new_tokens times.
        """
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens if necessary
            idx_cond = idx[:, -self.block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
# if __name__ == "__main__":
#     from data import load_data, get_batch
    
#     # Load sample data to get vocabulary size
#     train_ids, val_ids, stoi, itos = load_data("data/input.txt")
#     vocab_size = len(stoi)
    
#     # Hyperparameters
#     block_size = 64
#     batch_size = 4
#     n_embd = 128
#     n_head = 4
#     n_layer = 4
#     dropout = 0.1
#     device = "cpu"
    
#     # Instantiate model
#     model = Gpt(
#         vocab_size=vocab_size,
#         block_size=block_size,
#         n_embd=n_embd,
#         n_head=n_head,
#         n_layer=n_layer,
#         dropout=dropout
#     ).to(device)
    
#     # Check 1: Print parameter count
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"Model parameter count: {n_params / 1e6:.2f}M parameters")
    
#     # Check 2: Logits shape and Initial Loss check
#     xb, yb = get_batch(train_ids, batch_size, block_size, device)
#     logits, loss = model(xb, yb)
    
#     print(f"Logits shape: {logits.shape} (Expected: [{batch_size}, {block_size}, {vocab_size}])")
#     print(f"Initial loss: {loss.item():.4f}")
    
#     # Theoretical initial loss for uniform random logits is ln(vocab_size)
#     expected_initial_loss = math.log(vocab_size)
#     print(f"Expected initial loss (ln(vocab_size)): {expected_initial_loss:.4f}")