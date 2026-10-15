# TinyGPT: Build a GPT Transformer from Scratch

A character-level GPT (Generative Pre-trained Transformer) language model built completely from scratch using PyTorch, inspired by Andrej Karpathy's NanoGPT. This project implements every core component of a modern decoder-only transformer architecture without relying on high-level pre-built transformer libraries.

## Project Structure

`	ext
.
 data/
    input.txt          # Training text corpus (Shakespeare dataset)
 out/                   # Saved model checkpoints (ignored by git)
 data.py                # Vocab building, encoding, decoding, and batching
 model.py               # Transformer blocks, multi-head attention, and full GPT
 train.py               # Training loop, loss estimation, and checkpoint saving
 generate.py            # Text generation with temperature & top-k scaling
 bigram.py              # Simple bigram baseline for comparison
 test_tinygpt.py        # Unit tests (pytest)
 requirements.txt
 README.md
`

## Quick Start

1. Install dependencies:
`ash
pip install -r requirements.txt
`

2. Train the model:
`ash
python train.py
`

3. Generate text:
`ash
python generate.py
`

4. Run tests:
`ash
pytest -v
`
