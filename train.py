import os 

import torch 
import torch.nn as nn 
from data import load_data,get_batch
from model import Gpt 

config=dict(
    batch_size=16,
    block_size=128,
    n_embd=128,
    n_heads=4,
    n_layer=4,
    dropout=0.1,
    lr=3e-4,
    max_iters=5000,
    eval_interval=250,
    eval_iters=100,
    seed=1337
)

@torch.no_grad()
def estimate_loss(model,train_data,val_data,config):
    """
    Computes average loss over multiple batches for both train and val splits.
    Switches model between eval and train modes.
    """
    out={}
    model.eval()
    for split,data in [("train",train_data),("val",val_data)]:
        losses=torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X,Y=get_batch(data,config["batch_size"],config['block_size'], next(model.parameters()).device)
            _,loss=model(X,Y)
            losses[k]=loss.item()
            out[split] = losses.mean().item()
        model.train()
    return out
    
    
def main():
    # 1. Set seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # 2. Select device (CPU or GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 3. Load data and build vocabulary
    train_ids, val_ids, stoi, itos = load_data("data/input.txt")
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size} characters")
    print(f"Train data length: {len(train_ids):,} tokens")
    print(f"Val data length: {len(val_ids):,} tokens")
    
    # 4. Build model
    model = Gpt(
        vocab_size=vocab_size,
        block_size=config['block_size'],
        n_embd=config['n_embd'],
        n_head=config['n_heads'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    ).to(device)
    
    # Print total parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {n_params / 1e6:.2f}M parameters")
    
    # 5. Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    best_val_loss = float('inf')
    
    # 6. Training Loop
    print("\nStarting training loop...")
    for iter in range(config['max_iters']):
        
        # Periodically evaluate train and validation loss
        if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
            losses = estimate_loss(model, train_ids, val_ids, config)
            print(f"Step {iter:4d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")
            
            # Save best checkpoint when validation loss improves
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model_state': model.state_dict(),
                    'config': config,
                    'stoi': stoi,
                    'itos': itos
                }
                os.makedirs('out', exist_ok=True)
                torch.save(checkpoint, 'out/best_model.pt')
                print(f"--> Saved new best checkpoint with val loss: {best_val_loss:.4f}")
        
        # Sample a batch of data
        xb, yb = get_batch(train_ids, config['batch_size'], config['block_size'], device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backpropagation steps
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    print("Training complete!")

if __name__ == "__main__":
    main()
            