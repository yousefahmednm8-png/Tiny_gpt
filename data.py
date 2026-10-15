from numpy import block, dtype

import torch 

def build_vocab(text):
    """
    Builds charcter-level Vocabulary mappings (stoi and itos)
    """
    chars=sorted(list(set(text)))
    stoi={ch:i for i,ch in enumerate(chars)}
    itos={i:ch for i,ch in enumerate(chars)}
    return stoi,itos 
def encode(text,stoi):
    """
    Encodes a string into alist of intergers back to string using itos
    """
    return [stoi[ch]for ch in text]

def decode(ids,itos):
    """
    Decodes a list or tensor of int back into string using itos 
    """
    
    if isinstance(ids,torch.Tensor):
        ids=ids.tolist()
    return "".join([itos[i]for i in ids])


def load_data(path,train_split=0.9):
    """
    Loads text from a file, builds vocab, encodes the text and 
    splits it into training and validation torch tensors 
    
"""
    with open(path,"r",encoding="utf-8")as f:
        text=f.read()    
        
        stoi,itos=build_vocab(text)
        data=torch.tensor(encode(text,stoi),dtype=torch.long)
        
        # split data
        n=int(train_split*len(data))
        train_ids=data[:n]
        valid_ids=data[n:]
        return train_ids,valid_ids,stoi,itos
        
def get_batch(data,batch_size,block_size,device):
    """
    Generates a small batch of inputs x and y targets 
     shapes Note x and y are [B,T], where y is x shiffted one position to the right 
    """
    
    max_ids=len(data)-block_size
    ix=torch.randint(0,max_ids,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x=x.to(device)
    y=y.to(device)
    return x,y

   
   
# --- Test Script Example ---
if __name__ == "__main__":
    # Create a small dummy text file for testing
    sample_text = "Hello, world! This is a test file for our character-level language model."
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    # 1. Test load_data and build_vocab
    train_ids, val_ids, stoi, itos = load_data("data\input.txt")
    
    # 2. Check: decode(encode(s)) == s
    original_str = "Hello!"
    encoded_str = encode(original_str, stoi)
    decoded_str = decode(encoded_str, itos)
    assert decoded_str == original_str, f"Mismatch! Got: {decoded_str}"
    print("✓ Round-trip encode/decode check passed!")
    
    # 3. Test get_batch shapes and shifting
    device = torch.device("cpu")
    xb, yb = get_batch(train_ids, batch_size=2, block_size=8, device=device)
    
    print(f"\nBatch x shape: {xb.shape}") # Should be [2, 8]
    print(f"Batch y shape: {yb.shape}") # Should be [2, 8]
    
    # 4. Decode x[0] and y[0] to confirm y is x shifted by one character
    print("\n--- Visual Inspection Check ---")
    print("x[0]:", decode(xb[0], itos))
    print("y[0]:", decode(yb[0], itos))