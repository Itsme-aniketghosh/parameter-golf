#!/bin/bash
#SBATCH --job-name=pg_v8_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/ghosh.anik/parameter-golf-work/logs/smoke_%j.out
#SBATCH --error=/scratch/ghosh.anik/parameter-golf-work/logs/smoke_%j.err

module load cuda/12.1
module load anaconda3/2024.06
conda activate paramgolf
export PATH=/home/ghosh.anik/.conda/envs/paramgolf/bin:$PATH

mkdir -p /scratch/ghosh.anik/parameter-golf-work/logs

RUN_ID=smoke_v8 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=500 \
QAT_START_FRAC=0.15 \
INT_BITS=6 \
XSA_LAYERS=4 \
EMA_DECAY=0.9999 \
TTT_EPOCHS=5 \
TTT_RANK=8 \
TTT_LR=0.01 \
TTT_MIN_DOC_LEN=512 \
MAX_WALLCLOCK_SECONDS=120 \
EVAL_STRIDE=64 \
DATA_PATH=/scratch/ghosh.anik/parameter-golf-work/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/scratch/ghosh.anik/parameter-golf-work/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 \
    /scratch/ghosh.anik/parameter-golf-work/parameter-golf/records/track_10min_16mb/proteus_v8/train_gpt.py
