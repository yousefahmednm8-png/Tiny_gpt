# 🚀 TinyGPT: Build a GPT Transformer from Scratch

A character-level GPT (Generative Pre-trained Transformer) language model built completely from scratch using **PyTorch**, inspired by Andrej Karpathy's NanoGPT. This project implements every core component of a modern decoder-only transformer architecture without relying on high-level pre-built transformer libraries.

---

## 📂 Project Structure

```text
TinyGpt plan/
│
├── data/
│   └── input.txt          # Training text corpus (e.g., Shakespeare dataset)
├── out/
│   └── best_model.pt      # Saved model checkpoint (generated after training)
├── data.py                # Vocab building, encoding, decoding, and batching
├── model.py               # Transformer blocks, multi-head attention, and full GPT
├── train.py               # Training loop, loss estimation, and checkpoint saving
├── generate.py            # Text generation script with temperature & top-k scaling
├── bigram.py              # Simple bigram baseline model for comparison
└── test_tinygpt.py        # Unit tests suite (pytest)
