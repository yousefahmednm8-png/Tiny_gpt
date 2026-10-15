import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_data, get_batch

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Simple lookup table from vocabulary to logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] # Take the last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def main():
    # Setup
    batch_size = 32
    block_size = 64
    max_iters = 2000
    eval_interval = 500
    learning_rate = 1e-1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Bigram baseline on: {device}")
    
    # Load data using your existing data.py
    train_ids, val_ids, stoi, itos = load_data("data/input.txt")
    vocab_size = len(stoi)
    
    # Initialize model and optimizer
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate train loss on one batch
                xb, yb = get_batch(train_ids, batch_size, block_size, device)
                _, loss = model(xb, yb)
                print(f"Step {iter:4d} | Train Loss: {loss.item():.4f}")
            model.train()
            
        xb, yb = get_batch(train_ids, batch_size, block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    # Generate some text from the trained bigram model
    print("\n--- Bigram Generated Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=300)[0].tolist()
    print("".join([itos[i] for i in generated_indices]))

if __name__ == "__main__":
    main()