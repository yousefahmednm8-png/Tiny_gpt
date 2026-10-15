import torch
import torch.nn.functional as F
from model import Gpt

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    """
    Takes a conditioning sequence idx (B, T) and generates max_new_tokens.
    Includes temperature scaling and optional top-k filtering.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop context to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        
        # Forward pass to get logits
        logits, _ = model(idx_cond)
        
        # Focus on the last time step and apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Optional: Top-k filtering to prevent wild samples
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution (or use torch.argmax for greedy decoding)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading checkpoint on device: {device}")
    
    # Load checkpoint saved during training
    checkpoint = torch.load('out/best_model.pt', map_location=device)
    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Rebuild model architecture
    model = Gpt(
        vocab_size=len(stoi),
        block_size=config['block_size'],
        n_embd=config['n_embd'],
        n_head=config['n_heads'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    
    # Helper functions
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Test prompts and temperatures
    prompt = "\n"
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    temperatures = [0.5, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n=================== Temperature: {temp} ===================")
        # Reset context to prompt
        generated_idx = generate(model, idx, max_new_tokens=300, block_size=config['block_size'], temperature=temp)
        output_text = decode(generated_idx[0].tolist())
        print(output_text)

    # Greedy Decoding comparison (temperature approaching 0 or using argmax)
    print(f"\n=================== Greedy Decoding (Argmax) ===================")
    model.eval()
    curr_idx = idx.clone()
    for _ in range(300):
        curr_cond = curr_idx[:, -config['block_size']:]
        logits, _ = model(curr_cond)
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        curr_idx = torch.cat((curr_idx, next_token), dim=1)
    print(decode(curr_idx[0].tolist()))

if __name__ == "__main__":
    main()