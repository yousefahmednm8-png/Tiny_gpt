# TinyGPT - Simple Learning Plan (5 files)

Goal: understand every line of a character-level GPT by writing it yourself.
This is a learning project, not a portfolio piece. Few files, small steps, one check per step.

## 1. Structure

```text
tinygpt/
├── data/input.txt        # Tiny Shakespeare (download once)
├── data.py               # tokenizer + train/val split + get_batch
├── model.py              # Head, MultiHead, FeedForward, Block, GPT  (all in one file)
├── train.py              # hyperparameters + training loop + checkpoint
├── generate.py           # load checkpoint + sample text
├── test_tinygpt.py       # small tests (plain pytest, one file)
└── requirements.txt      # torch,pytest, matplotlib 
```

Why this works:
- Each file has one job and you can read it top to bottom in one sitting.
- No config.py: hyperparameters live in a dict at the top of `train.py` and are saved inside the checkpoint.
- No packages or `__init__.py`: run everything from the project root (`python train.py`, `pytest`).
- `model.py` knows nothing about data or training. It only takes numbers like `vocab_size`, `n_embd`.

Dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Reference (read only to understand a specific question, never to copy): Karpathy's "Let's build GPT" and `gpt.py` from `ng-video-lecture`.

## 2. Rules

- Write every file yourself. If you use an AI, use it only to explain an error or review finished code.
- Before each function: write the input and output shapes in a comment.
- After each function: run a tiny check (print shapes, one assert).
- Do not start the next step until the current check passes.
- Allowed: `torch`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `torch.nn.functional`.
- Not allowed in v1: `nn.MultiheadAttention`, Hugging Face, pretrained weights, copied GPT code.

## 3. Steps

### Step 0 - Setup (30 min)
- Create the folder, virtual env, install `torch pytest matplotlib`, download `input.txt`.
- Print: number of characters, number of unique characters (expect 65), first 200 characters.

### Step 1 - `data.py`
Functions to write:

```python
def build_vocab(text) -> (stoi, itos)          # dicts: char -> id, id -> char
def encode(text, stoi) -> list[int]
def decode(ids, itos) -> str
def load_data(path) -> (train_ids, val_ids, stoi, itos)   # 90% / 10%, tensors
def get_batch(data, batch_size, block_size, device) -> (x, y)
```

Shape notes: `x` and `y` are `[B, T]`, and `y` is `x` shifted one position to the right.

Checks:
- `decode(encode(s)) == s`.
- Decode `x[0]` and `y[0]` and confirm by eye that `y` is `x` shifted by one character.
- Random start index never goes past the end of the data.

### Step 2 - `model.py`: one attention head
```python
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout): ...
    def forward(self, x): ...      # x: [B, T, C] -> [B, T, head_size]
```
Write down the shape of: q, k, v, scores, weights, output.
Remember: scale the scores, mask the future *before* softmax, softmax over the last dimension.

Checks:
- Output shape correct for two different `T` values.
- Each row of the attention weights sums to ~1 and is zero for future positions.
- Change a token at position `t` and confirm outputs before `t` do not change (eval mode).
- Compare with `F.scaled_dot_product_attention(q, k, v, is_causal=True)` using `torch.allclose`.

### Step 3 - `model.py`: multi-head, feed-forward, block
```python
class MultiHead(nn.Module): ...      # several Heads -> concat -> projection -> dropout
class FeedForward(nn.Module): ...    # Linear -> activation -> Linear -> dropout
class Block(nn.Module): ...          # attention + feed-forward with residuals and LayerNorm
```
Decide pre-LayerNorm or post-LayerNorm and write the choice as a comment.

Checks:
- Block input and output are both `[B, T, C]`.
- `head_size * n_head == n_embd`.
- Backward pass gives finite gradients.
- In eval mode, running the same input twice gives the same output.

### Step 4 - `model.py`: full GPT
```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout): ...
    def forward(self, idx, targets=None): ...   # returns logits [B,T,V], loss or None
```
Parts: token embedding, position embedding, stack of blocks, final LayerNorm, output Linear.
For the loss: reshape logits to `[B*T, V]` and targets to `[B*T]` before cross-entropy.

Checks:
- Logits shape `[B, T, vocab_size]`.
- Initial loss is close to `ln(vocab_size)` (about 4.17 for 65 characters). If not, something is wrong.
- Print the parameter count.

### Step 5 - Can it learn? (sanity check, inside `train.py` or a scratch script)
- Take one fixed batch, set `dropout=0`, train 200-500 steps on it.
- The loss should go close to 0.
- If it does not, the bug is in the model or the loss, not in the data size.
- Be able to explain why this proves the code works but does not prove generalization.

### Step 6 - `train.py`
Put at the top:

```python
config = dict(batch_size=16, block_size=128, n_embd=128, n_head=4, n_layer=4,
              dropout=0.1, lr=3e-4, max_iters=5000, eval_interval=250,
              eval_iters=100, seed=1337)
```

Functions to write:
```python
@torch.no_grad()
def estimate_loss(model, ...) -> {"train": x, "val": y}   # use model.eval() then model.train()

def main(): ...   # seed, device, load data, build model, AdamW loop, save best checkpoint
```

Loop order every step: get batch -> forward -> `zero_grad` -> `backward` -> `step`.

Save in the checkpoint (when val loss improves): model weights, `config`, `stoi`, `itos`.
That makes `generate.py` work in a separate process with no other files.

Checks:
- Both train and val loss go down; save the numbers and plot them with matplotlib.
- Stop and investigate if loss becomes NaN.
- Be able to explain: underfitting vs overfitting vs a healthy curve.

### Step 7 - `generate.py`
```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0): ...
```
Loop: crop to the last `block_size` tokens -> forward -> take the last time step -> divide by temperature -> softmax -> `multinomial` -> append.

Checks:
- Works from a fresh terminal without retraining: `python generate.py`.
- Try three prompts and temperatures 0.5, 1.0, 1.5.
- Compare with greedy decoding (argmax) and note repetition.

### Step 8 - `test_tinygpt.py` (one file, a few tests)
- Tokenizer round trip.
- `get_batch` shapes and the shift.
- Causal mask: changing a future token does not change earlier outputs.
- GPT output shape and a finite loss.
- Run with `pytest` on CPU.

### Step 9 - Experiments (this is where you learn the most)
Change one thing at a time and record train and val loss in a small table:

| Run | n_head | n_layer | n_embd | block_size | dropout | train loss | val loss |
|---|---|---|---|---|---|---|---|

Also train a simple bigram model (embedding table only) as a baseline and compare it with the GPT.

Optional v2, one at a time, measure before and after: LR warmup + cosine decay, gradient clipping, mixed precision, weight tying, top-k / top-p sampling, KV-cache, BPE tokenizer.

## 4. Debugging checklist (use before asking anyone)

1. Print shapes at every step.
2. Test with tiny sizes (`B=2, T=4, C=8`) so you can read the numbers.
3. Initial loss far from `ln(65)`? Check the loss reshape and the output layer.
4. Loss not decreasing on one batch? Check `zero_grad`, the optimizer parameters, `dropout=0`, and the learning rate.
5. Future leakage? Check the mask and that softmax is over the last dimension.
6. Generation is garbage but loss looks fine? Check `model.eval()`, cropping to `block_size`, and that you use the last time step.

## 5. Questions to answer in your own words (write them in a `notes.md`)

1. Why decoder-only, and why no encoder?
2. What are query, key, value, and their shapes?
3. Why scale the scores before softmax?
4. What does the causal mask prevent?
5. Why are targets shifted by one?
6. What does cross-entropy measure here?
7. Training loss vs validation loss?
8. Why can a model overfit a tiny dataset?
9. Why generate without gradient tracking?
10. What are the limits of character-level tokenization?

## 6. Done when

- `python train.py` trains and saves a checkpoint; `python generate.py` produces text.
- Tests pass.
- Validation loss is in a reasonable range (roughly 1.5-1.8 for the starter config; a much higher value means check the code).
- You can answer the 10 questions without looking at the code.
- You have one experiments table and a short `notes.md`.

Estimated time: about 2-3 weeks at a steady pace.
