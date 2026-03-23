# PROTEUS v8 — 11L Int6 + EMA + LoRA TTT (5ep cosine)

## Summary
- **val_bpb**: TBD (fill in after run)
- **Architecture**: 11L transformer, Int6 QAT, EMA, XSA last 4 layers
- **TTT**: LoRA TTT v8 — 5 epochs, cosine LR decay, score-before-train every epoch
- **Artifact**: ~15.x MB

## What changed from v7
- 5 TTT epochs (was 3)
- Cosine LR decay across epochs (was flat)
- Score every token before training on it, **every epoch, every pass**
  (v7 only scored on final epoch)

## Key Techniques

### Training
- 11 transformer layers, d_model=512
- Int6 QAT with GPTQ-lite clip search
- Late QAT onset at 15% of training steps
- EMA weight averaging (decay=0.9999)
- XSA on last 4 layers (shared attention weights)
- BigramHash(2048) + SmearGate
- Muon optimizer (WD=0.04) + AdamW
- Warmdown 3500 steps

### Evaluation (v8 TTT)
- Per-document LoRA adapters (rank=8)
- Reset adapters between documents (no cross-doc leakage)
- 5 epochs per document with cosine LR: starts at 0.01, decays to 0.001
- Every epoch: score token N → then train on token N (causal, backward-looking)
- Sliding window evaluation (stride=64)
- Short documents (<512 tokens): standard eval, no TTT

## TTT Rule Compliance
Following the same sequential pattern as merged PR #77.
Score every token BEFORE training on it, in every epoch, every pass.
No forward-looking access to val tokens.

## Results
| Seed | Post-Quant BPB | TTT BPB | Artifact | Status |
|------|---------------|---------|---------|--------|
| 42   | TBD           | TBD     | TBD MB  | TBD    |
| 1337 | TBD           | TBD     | TBD MB  | TBD    |
| 2024 | TBD           | TBD     | TBD MB  | TBD    |

## Run Command
```bash
NCCL_IB_DISABLE=1 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=3500 \
QAT_START_FRAC=0.15 \
INT_BITS=6 \
XSA_LAYERS=4 \
EMA_DECAY=0.9999 \
TTT_EPOCHS=5 TTT_RANK=8 TTT_LR=0.01 TTT_MIN_DOC_LEN=512 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Platform
Northeastern University HPC Discovery (A100) / RunPod 8×H100 SXM
GitHub: https://github.com/Itsme-aniketghosh
