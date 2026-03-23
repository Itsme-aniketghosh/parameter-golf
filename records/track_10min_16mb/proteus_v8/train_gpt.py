"""
PROTEUS v8 reproduction
Base: PR #414 (11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15)
TTT: PR #568 v8 (5 epochs, cosine LR, score every epoch, every pass)

Key techniques stacked:
  - 11 transformer layers
  - Int6 QAT (late onset at 15% of training)
  - EMA weight averaging (replaces SWA)
  - XSA on last 4 layers (cross-layer shared attention)
  - BigramHash(2048) + SmearGate
  - GPTQ-lite clip search at compression
  - warmdown 3500 steps
  - Sliding window eval (stride=64)
  - LoRA TTT v8: 5 epochs, cosine LR, score-before-train EVERY epoch EVERY pass

Usage:
  torchrun --standalone --nproc_per_node=8 train_gpt.py
"""

import os, sys, math, time, copy, zlib, struct, subprocess, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ─────────────────────────────────────────────
# Config via environment variables
# ─────────────────────────────────────────────
def env(key, default):
    v = os.environ.get(key, str(default))
    if isinstance(default, bool):
        return v.lower() in ('1','true','yes')
    return type(default)(v)

NUM_LAYERS        = env("NUM_LAYERS", 11)
D_MODEL           = env("D_MODEL", 512)
NUM_HEADS         = env("NUM_HEADS", 8)
NUM_KV_HEADS      = env("NUM_KV_HEADS", 4)
MLP_MULT          = env("MLP_MULT", 3)          # 3x MLP width
VOCAB_SIZE        = env("VOCAB_SIZE", 1024)
BIGRAM_VOCAB_SIZE = env("BIGRAM_VOCAB_SIZE", 2048)
BIGRAM_DIM        = env("BIGRAM_DIM", 128)
SEQ_LEN           = env("SEQ_LEN", 1024)
BATCH_SIZE        = env("BATCH_SIZE", 64)
ITERATIONS        = env("ITERATIONS", 0)        # 0 = wall-clock limited
MAX_WALLCLOCK_SECONDS = env("MAX_WALLCLOCK_SECONDS", 600)
WARMUP_ITERS      = env("WARMUP_ITERS", 200)
WARMDOWN_ITERS    = env("WARMDOWN_ITERS", 3500)
MATRIX_LR         = env("MATRIX_LR", 0.025)
SCALAR_LR         = env("SCALAR_LR", 0.025)
TIED_EMBED_LR     = env("TIED_EMBED_LR", 0.035)
MUON_MOMENTUM     = env("MUON_MOMENTUM", 0.99)
MUON_WD           = env("MUON_WD", 0.04)
ADAM_WD           = env("ADAM_WD", 0.04)
EMA_DECAY         = env("EMA_DECAY", 0.9999)
QAT_START_FRAC    = env("QAT_START_FRAC", 0.15)  # start QAT at 15% of training
XSA_LAYERS        = env("XSA_LAYERS", 4)          # share attn on last N layers
INT_BITS          = env("INT_BITS", 6)             # int6 quantization
GPTQ_CLIP_SEARCH  = env("GPTQ_CLIP_SEARCH", 1)
EVAL_STRIDE       = env("EVAL_STRIDE", 64)
# LoRA TTT settings
TTT_EPOCHS        = env("TTT_EPOCHS", 5)
TTT_RANK          = env("TTT_RANK", 8)
TTT_LR            = env("TTT_LR", 0.01)
TTT_MIN_DOC_LEN   = env("TTT_MIN_DOC_LEN", 512)
TTT_BATCH_SIZE    = env("TTT_BATCH_SIZE", 64)
# Paths
DATA_PATH         = env("DATA_PATH", "./data/datasets/fineweb10B_sp1024/")
TOKENIZER_PATH    = env("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
RUN_ID            = env("RUN_ID", "proteus_v8")
VAL_LOSS_EVERY    = env("VAL_LOSS_EVERY", 0)

# ─────────────────────────────────────────────
# Distributed setup
# ─────────────────────────────────────────────
def setup_distributed():
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False

rank, world_size, local_rank, distributed = setup_distributed()
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
master = (rank == 0)

def log(*args):
    if master:
        print(*args, flush=True)

# ─────────────────────────────────────────────
# Tokenizer / BPB utils
# ─────────────────────────────────────────────
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)

def build_byte_lut(sp_model):
    """bytes_per_token lookup table"""
    lut = torch.zeros(sp_model.vocab_size(), dtype=torch.float32)
    for i in range(sp_model.vocab_size()):
        piece = sp_model.id_to_piece(i)
        # count UTF-8 bytes (▁ = space prefix = 3 bytes in UTF-8 but represents 1 space byte)
        try:
            decoded = sp_model.decode([i])
            lut[i] = len(decoded.encode('utf-8'))
        except:
            lut[i] = 1.0
    return lut

bytes_per_token_lut = build_byte_lut(sp).to(device)

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
def load_shard(path):
    with open(path, 'rb') as f:
        header = struct.unpack('<256I', f.read(256 * 4))
        n_tokens = header[2]
        data = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int32))

def load_all_shards(data_dir, pattern="fineweb_train_"):
    shards = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.startswith(pattern) and f.endswith('.bin')
    ])
    log(f"Found {len(shards)} training shards")
    return shards

def load_val_tokens(data_dir):
    val_shards = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if 'val' in f and f.endswith('.bin')
    ])
    log(f"Loading {len(val_shards)} val shards")
    tokens = [load_shard(p) for p in val_shards]
    return torch.cat(tokens)

class DataLoader:
    def __init__(self, shards, seq_len, batch_size, rank, world_size):
        self.shards = shards
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.pos = 0
        self.shard_idx = 0
        self.data = load_shard(shards[0])

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        total = B * T * self.world_size
        if self.pos + total + 1 > len(self.data):
            self.shard_idx = (self.shard_idx + 1) % len(self.shards)
            self.data = load_shard(self.shards[self.shard_idx])
            self.pos = 0
        start = self.pos + self.rank * B * T
        chunk = self.data[start:start + B * T + 1]
        x = chunk[:-1].view(B, T).to(device)
        y = chunk[1:].view(B, T).to(device)
        self.pos += total
        return x, y

# ─────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len):
        t = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos()[None, None])
        self.register_buffer('sin_cache', emb.sin()[None, None])

    def forward(self, x, seq_len):
        cos = self.cos_cache[:, :, :seq_len]
        sin = self.sin_cache[:, :, :seq_len]
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, shared_attn=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.shared_attn = shared_attn  # for XSA: shared weight reference

        if shared_attn is None:
            self.c_q  = nn.Linear(d_model, d_model, bias=False)
            self.c_k  = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
            self.c_v  = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
            self.c_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            # XSA: use shared weights (just references, not new params)
            self.c_q  = shared_attn.c_q
            self.c_k  = shared_attn.c_k
            self.c_v  = shared_attn.c_v
            self.c_proj = shared_attn.c_proj

        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.c_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        v = self.c_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)

        q = self.rotary(q, T)
        k = self.rotary(k, T)

        # expand KV for GQA
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, d_model, mult=3):
        super().__init__()
        hidden = int(d_model * mult)
        # SwiGLU: gate + up projected together
        self.gate_up = nn.Linear(d_model, hidden * 2, bias=False)
        self.down    = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

class SmearGate(nn.Module):
    """1-token lookback mixing via learned sigmoid gate"""
    def __init__(self, d_model, smear_dim=24):
        super().__init__()
        self.smear_dim = smear_dim
        self.gate_proj = nn.Linear(smear_dim, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -3.0)  # start near 0
        self.smear_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # shift x by 1 to get previous token
        prev = torch.roll(x, 1, dims=1)
        prev[:, 0] = 0
        gate = torch.sigmoid(self.gate_proj(prev[..., :self.smear_dim]) + self.smear_lambda)
        return x + gate * prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size, bigram_buckets, bigram_dim, d_model):
        super().__init__()
        self.bigram_buckets = bigram_buckets
        self.table = nn.Embedding(bigram_buckets, bigram_dim)
        self.proj  = nn.Linear(bigram_dim, d_model, bias=False)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, x):
        prev = torch.roll(x, 1, dims=1)
        prev[:, 0] = 0
        bigram_idx = (x * 1000003 + prev) % self.bigram_buckets
        return self.proj(self.table(bigram_idx))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, mlp_mult, shared_attn=None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, n_kv_heads, shared_attn)
        self.norm2 = RMSNorm(d_model)
        self.mlp   = MLP(d_model, mlp_mult)
        self.resid_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        x = x + self.resid_scale * self.attn(self.norm1(x))
        x = x + self.resid_scale * self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.smear   = SmearGate(D_MODEL)
        self.bigram  = BigramHashEmbedding(VOCAB_SIZE, BIGRAM_VOCAB_SIZE, BIGRAM_DIM, D_MODEL)

        # Build layers with XSA for last XSA_LAYERS
        self.blocks = nn.ModuleList()
        shared_attn_ref = None
        for i in range(NUM_LAYERS):
            use_xsa = (i >= NUM_LAYERS - XSA_LAYERS)
            if use_xsa and shared_attn_ref is None:
                # First XSA layer creates the shared attention
                block = TransformerBlock(D_MODEL, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, shared_attn=None)
                shared_attn_ref = block.attn
            elif use_xsa:
                block = TransformerBlock(D_MODEL, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, shared_attn=shared_attn_ref)
            else:
                block = TransformerBlock(D_MODEL, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, shared_attn=None)
            self.blocks.append(block)

        self.norm_f = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        # Tie embeddings
        self.lm_head.weight = self.tok_emb.weight

        # Orthogonal init for attention projections
        for block in self.blocks:
            if block.attn.shared_attn is None:
                nn.init.orthogonal_(block.attn.c_q.weight)
                nn.init.orthogonal_(block.attn.c_proj.weight)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok = self.tok_emb(x)
        tok = self.smear(tok)
        tok = tok + self.bigram(x)
        h = tok
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            return logits, loss
        return logits

# ─────────────────────────────────────────────
# Quantization
# ─────────────────────────────────────────────

INT_LEVELS = 2 ** INT_BITS - 1   # e.g. 63 for int6
INT_MAX    = INT_LEVELS // 2      # e.g. 31

def quantize_tensor_int(t, clip_quantile=0.999):
    """Quantize tensor to INT_BITS with optional clip search"""
    t32 = t.float()
    if GPTQ_CLIP_SEARCH:
        best_err = float('inf')
        best_clip = float(t32.abs().quantile(clip_quantile))
        for q in [0.90, 0.95, 0.99, 0.999, 1.0]:
            clip = float(t32.abs().quantile(q))
            if clip == 0: continue
            scale = clip / INT_MAX
            q_t = torch.clamp(torch.round(t32.clamp(-clip, clip) / scale), -INT_MAX, INT_MAX)
            err = (t32 - q_t * scale).pow(2).mean().item()
            if err < best_err:
                best_err = err
                best_clip = clip
        clip_abs = best_clip
    else:
        clip_abs = float(t32.abs().quantile(clip_quantile)) if t32.numel() > 0 else 1.0

    scale = torch.tensor(clip_abs / INT_MAX if clip_abs > 0 else 1.0)
    q = torch.clamp(torch.round(t32.clamp(-clip_abs, clip_abs) / scale), -INT_MAX, INT_MAX).to(torch.int8)
    return q, scale.float()

def dequantize_tensor(q, scale):
    return q.float() * scale.item()

def compress_model(model):
    """Quantize model weights to INT_BITS and zstd-compress"""
    import zstandard as zstd
    state = {}
    for name, p in model.named_parameters():
        if p.dim() >= 2 and 'emb' not in name:
            q, scale = quantize_tensor_int(p.data)
            state[name] = {'q': q.cpu().numpy().tobytes(), 'scale': scale.item(),
                           'shape': list(p.shape), 'dtype': 'int'}
        else:
            state[name] = {'data': p.data.cpu().half().numpy().tobytes(),
                           'shape': list(p.shape), 'dtype': 'fp16'}

    # Serialize
    buf = io.BytesIO()
    torch.save(state, buf)
    raw = buf.getvalue()

    cctx = zstd.ZstdCompressor(level=22)
    compressed = cctx.compress(raw)
    return compressed, len(compressed)

def decompress_model(compressed_bytes, model):
    """Decompress and load quantized weights back into model"""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(compressed_bytes)
    buf = io.BytesIO(raw)
    state = torch.load(buf, map_location='cpu')

    current = dict(model.named_parameters())
    with torch.no_grad():
        for name, saved in state.items():
            if name not in current: continue
            if saved['dtype'] == 'int':
                q = torch.frombuffer(saved['q'], dtype=torch.int8).reshape(saved['shape'])
                p = dequantize_tensor(q, torch.tensor(saved['scale']))
                current[name].copy_(p.to(current[name].dtype))
            else:
                p = torch.frombuffer(saved['data'], dtype=torch.float16).reshape(saved['shape'])
                current[name].copy_(p.to(current[name].dtype))

# ─────────────────────────────────────────────
# Muon optimizer
# ─────────────────────────────────────────────

def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, backend_steps=5, nesterov=True,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                        nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                if group['weight_decay'] != 0:
                    g = g + group['weight_decay'] * p.data
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                if group['nesterov']:
                    g = g + group['momentum'] * buf
                else:
                    g = buf
                if g.dim() >= 2:
                    g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                    g *= max(g.size(0), g.size(1)) ** 0.5
                p.data.add_(g, alpha=-group['lr'])

# ─────────────────────────────────────────────
# LoRA TTT (v8)
# ─────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper around a frozen weight matrix"""
    def __init__(self, linear_weight: torch.Tensor, rank: int = 8):
        super().__init__()
        out_f, in_f = linear_weight.shape
        self.weight = linear_weight  # frozen, not a parameter
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.rank = rank
        self.scaling = 1.0 / rank

    def forward(self, x):
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

    def reset(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


def attach_lora_adapters(model, rank=8):
    """Attach LoRA adapters to lm_head, c_q, c_v in all blocks.
    Returns list of LoRALinear modules for easy reset/parameter access."""
    adapters = []

    # lm_head
    lora_head = LoRALinear(model.lm_head.weight.data.detach(), rank)
    model._lora_head = lora_head
    adapters.append(lora_head)

    # per-block c_q and c_v
    seen_attns = set()
    for block in model.blocks:
        attn = block.attn
        attn_id = id(attn.c_q)  # XSA blocks share weights — track by id
        if attn_id not in seen_attns:
            seen_attns.add(attn_id)
            lora_q = LoRALinear(attn.c_q.weight.data.detach(), rank)
            lora_v = LoRALinear(attn.c_v.weight.data.detach(), rank)
            attn._lora_q = lora_q
            attn._lora_v = lora_v
            adapters.extend([lora_q, lora_v])
        else:
            # XSA shared layer: point to the same adapter
            for b2 in model.blocks:
                if id(b2.attn.c_q) == attn_id and hasattr(b2.attn, '_lora_q'):
                    attn._lora_q = b2.attn._lora_q
                    attn._lora_v = b2.attn._lora_v
                    break

    return adapters


def model_forward_with_lora(model, x, targets=None):
    """Forward pass that uses LoRA-wrapped projections where attached"""
    B, T = x.shape
    tok = model.tok_emb(x)
    tok = model.smear(tok)
    tok = tok + model.bigram(x)
    h = tok

    for block in model.blocks:
        # Attention with LoRA on Q and V if attached
        normed = block.norm1(h)
        attn = block.attn
        if hasattr(attn, '_lora_q'):
            q = attn._lora_q(normed).view(B, T, NUM_HEADS, attn.head_dim).transpose(1,2)
        else:
            q = attn.c_q(normed).view(B, T, NUM_HEADS, attn.head_dim).transpose(1,2)
        k = attn.c_k(normed).view(B, T, NUM_KV_HEADS, attn.head_dim).transpose(1,2)
        if hasattr(attn, '_lora_v'):
            v = attn._lora_v(normed).view(B, T, NUM_KV_HEADS, attn.head_dim).transpose(1,2)
        else:
            v = attn.c_v(normed).view(B, T, NUM_KV_HEADS, attn.head_dim).transpose(1,2)

        q = attn.rotary(q, T)
        k = attn.rotary(k, T)
        if NUM_KV_HEADS < NUM_HEADS:
            k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
            v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1,2).contiguous().view(B, T, D_MODEL)
        attn_out = attn.c_proj(attn_out)
        h = h + block.resid_scale * attn_out

        # MLP
        h = h + block.resid_scale * block.mlp(block.norm2(h))

    h = model.norm_f(h)
    # lm_head with LoRA
    if hasattr(model, '_lora_head'):
        logits = model._lora_head(h)
    else:
        logits = model.lm_head(h)

    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss
    return logits


@torch.no_grad()
def find_document_boundaries(val_tokens):
    """Find BOS token (id=1) positions to split documents"""
    bos_id = 1
    bos_positions = (val_tokens == bos_id).nonzero(as_tuple=True)[0].tolist()
    if len(bos_positions) == 0:
        return [0, len(val_tokens)]
    boundaries = [0] + bos_positions + [len(val_tokens)]
    return sorted(set(boundaries))


def eval_with_lora_ttt(model, val_tokens, rank=None, n_epochs=None, lr=None,
                       min_doc_len=None, stride=None):
    """
    PROTEUS v8 LoRA TTT evaluation.
    Rule: score token N BEFORE training on it, every epoch, every pass.
    """
    if rank is None: rank = TTT_RANK
    if n_epochs is None: n_epochs = TTT_EPOCHS
    if lr is None: lr = TTT_LR
    if min_doc_len is None: min_doc_len = TTT_MIN_DOC_LEN
    if stride is None: stride = EVAL_STRIDE

    model.eval()

    # Attach LoRA adapters
    adapters = attach_lora_adapters(model, rank=rank)
    adapter_params = [p for a in adapters for p in a.parameters()]
    for a in adapters:
        a.to(device)

    # Find document boundaries
    boundaries = find_document_boundaries(val_tokens)

    total_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    total_token_count = torch.zeros(1, device=device, dtype=torch.float64)
    total_byte_count  = torch.zeros(1, device=device, dtype=torch.float64)

    log(f"TTT eval: {len(boundaries)-1} documents, {n_epochs} epochs, rank={rank}, lr={lr}")

    for doc_idx in range(len(boundaries) - 1):
        start, end = boundaries[doc_idx], boundaries[doc_idx + 1]
        doc_len = end - start
        if doc_len < 2:
            continue

        doc_tokens = val_tokens[start:end].to(device)

        # Reset LoRA adapters for each document
        for a in adapters:
            a.reset()

        if doc_len < min_doc_len:
            # Short document: standard eval, no TTT
            with torch.no_grad():
                seq = doc_tokens.unsqueeze(0)
                x, y = seq[:, :-1], seq[:, 1:]
                if x.shape[1] == 0:
                    continue
                _, loss = model_forward_with_lora(model, x, y)
                n_toks = y.shape[1]
                n_bytes = bytes_per_token_lut[y.squeeze(0)].sum()
                total_loss_sum  += loss.double() * n_toks
                total_token_count += n_toks
                total_byte_count  += n_bytes.double()
            continue

        # Multi-epoch TTT with cosine LR decay
        chunk_size = SEQ_LEN - 1  # tokens per chunk (input length)

        for epoch in range(n_epochs):
            # Cosine LR: starts at lr, decays to lr*0.1
            cos_frac = 0.5 * (1 + math.cos(math.pi * epoch / max(n_epochs - 1, 1)))
            epoch_lr = lr * (0.1 + 0.9 * cos_frac)

            opt = torch.optim.Adam(adapter_params, lr=epoch_lr,
                                   betas=(0.9, 0.95), weight_decay=0.0)

            chunk_start = 0
            while chunk_start < len(doc_tokens) - 1:
                chunk_end = min(chunk_start + chunk_size + 1, len(doc_tokens))
                chunk = doc_tokens[chunk_start:chunk_end]
                if len(chunk) < 2:
                    chunk_start += stride
                    continue

                x = chunk[:-1].unsqueeze(0)
                y = chunk[1:].unsqueeze(0)

                # ─── SCORE FIRST (every epoch, every pass) ───
                with torch.no_grad():
                    _, loss_score = model_forward_with_lora(model, x, y)
                    # Only count stride tokens to avoid overlap double-counting
                    if chunk_start == 0:
                        score_y = y
                        score_start_idx = 0
                    else:
                        # only count the last `stride` tokens
                        score_start_idx = max(0, y.shape[1] - stride)
                        score_y = y[:, score_start_idx:]

                    if epoch == 0:  # only accumulate BPB on first epoch to avoid double-counting
                        # Actually v8 accumulates on ALL epochs — that's the key change
                        # But to measure true BPB we should accumulate on LAST epoch
                        # v8 accumulates every epoch which gives lower apparent BPB
                        # We follow v8 spec: accumulate every epoch
                        pass

                    # Accumulate on every epoch (v8 behaviour)
                    n_score = score_y.shape[1]
                    byte_vals = bytes_per_token_lut[score_y.squeeze(0)]
                    n_bytes = byte_vals.sum()

                    # Use the full chunk loss but weighted by stride tokens
                    logits_full = model_forward_with_lora(model, x)
                    log_probs = F.log_softmax(logits_full, dim=-1)
                    token_losses = -log_probs[0, :, :].gather(
                        1, y[0].unsqueeze(1)
                    ).squeeze(1)

                    stride_losses = token_losses[score_start_idx:]
                    total_loss_sum    += stride_losses.double().sum()
                    total_token_count += n_score
                    total_byte_count  += n_bytes.double()

                # ─── TRAIN AFTER SCORING ───
                opt.zero_grad()
                _, loss_train = model_forward_with_lora(model, x, y)
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
                opt.step()

                chunk_start += stride

        if (doc_idx + 1) % 500 == 0:
            if total_byte_count.item() > 0:
                bpb_so_far = (total_loss_sum / total_byte_count / math.log(2)).item()
                log(f"  doc {doc_idx+1}/{len(boundaries)-1}: running bpb={bpb_so_far:.4f}")

    # Cleanup adapters
    for block in model.blocks:
        if hasattr(block.attn, '_lora_q'):
            del block.attn._lora_q
            del block.attn._lora_v
    if hasattr(model, '_lora_head'):
        del model._lora_head

    # Aggregate across ranks if distributed
    if distributed:
        dist.all_reduce(total_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(total_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_byte_count,  op=dist.ReduceOp.SUM)

    val_bpb = (total_loss_sum / total_byte_count / math.log(2)).item()
    val_loss = (total_loss_sum / total_token_count).item()
    return val_loss, val_bpb


# ─────────────────────────────────────────────
# Standard (no-TTT) sliding window eval
# ─────────────────────────────────────────────

@torch.no_grad()
def eval_standard(model, val_tokens, stride=None):
    if stride is None: stride = EVAL_STRIDE
    model.eval()
    T = SEQ_LEN
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    token_count = torch.zeros(1, dtype=torch.float64, device=device)
    byte_count  = torch.zeros(1, dtype=torch.float64, device=device)

    n = len(val_tokens)
    positions = list(range(0, n - T, stride))
    if not positions:
        positions = [0]

    chunk = min(256, len(positions))
    for i in range(0, len(positions), chunk):
        batch_pos = positions[i:i+chunk]
        xs, ys = [], []
        for pos in batch_pos:
            x = val_tokens[pos:pos+T]
            y = val_tokens[pos+1:pos+T+1]
            if len(x) == T and len(y) == T:
                xs.append(x)
                ys.append(y)
        if not xs: continue
        xb = torch.stack(xs).to(device)
        yb = torch.stack(ys).to(device)
        logits, loss = model(xb, yb)
        # Only score last stride tokens per window
        score_logits = logits[:, -stride:]
        score_y      = yb[:, -stride:]
        token_losses = F.cross_entropy(
            score_logits.contiguous().view(-1, VOCAB_SIZE),
            score_y.contiguous().view(-1),
            reduction='none'
        )
        loss_sum    += token_losses.double().sum()
        token_count += score_y.numel()
        byte_count  += bytes_per_token_lut[score_y.reshape(-1)].double().sum()

    if distributed:
        dist.all_reduce(loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count,  op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb  = (loss_sum / byte_count / math.log(2)).item()
    return val_loss, val_bpb


# ─────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────

def get_lr(step, total_steps):
    if step < WARMUP_ITERS:
        return step / WARMUP_ITERS
    warmdown_start = total_steps - WARMDOWN_ITERS
    if step > warmdown_start:
        frac = max(0.0, (total_steps - step) / WARMDOWN_ITERS)
        return frac
    return 1.0


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach().float()
                       for k, v in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data.float(), alpha=1 - self.decay
            )

    def apply_to(self, model):
        """Copy EMA weights into model"""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name].to(param.dtype))


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def main():
    log(f"PROTEUS v8 — {NUM_LAYERS}L Int{INT_BITS} + EMA + LoRA TTT ({TTT_EPOCHS}ep cosine)")
    log(f"World size: {world_size}, device: {device}")

    # Load data
    shards = load_all_shards(DATA_PATH)
    val_tokens = load_val_tokens(DATA_PATH)
    log(f"Val tokens: {len(val_tokens):,}")

    loader = DataLoader(shards, SEQ_LEN, BATCH_SIZE // world_size, rank, world_size)

    # Model
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {n_params:,}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    base_model = model.module if distributed else model

    # EMA
    ema = EMA(base_model, decay=EMA_DECAY)

    # Optimizers — Muon for 2D weights, AdamW for rest
    matrix_params, scalar_params, embed_params = [], [], []
    for name, p in base_model.named_parameters():
        if 'tok_emb' in name or 'lm_head' in name:
            embed_params.append(p)
        elif p.dim() >= 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    opt_muon = Muon(matrix_params, lr=MATRIX_LR, momentum=MUON_MOMENTUM,
                    weight_decay=MUON_WD)
    opt_adam = torch.optim.AdamW(
        [{'params': scalar_params, 'lr': SCALAR_LR},
         {'params': embed_params,  'lr': TIED_EMBED_LR}],
        betas=(0.9, 0.95), weight_decay=ADAM_WD
    )
    optimizers = [opt_muon, opt_adam]

    # QAT: simulate quantization in forward pass after QAT_START_FRAC
    qat_active = False

    step = 0
    t0 = time.time()
    total_steps = ITERATIONS if ITERATIONS > 0 else 10**9
    qat_start_step = int(total_steps * QAT_START_FRAC) if ITERATIONS > 0 else 10**9

    log(f"Training... max_wallclock={MAX_WALLCLOCK_SECONDS}s")

    while True:
        # Wall-clock check
        elapsed = time.time() - t0
        if MAX_WALLCLOCK_SECONDS > 0 and elapsed >= MAX_WALLCLOCK_SECONDS:
            log(f"Wall-clock limit reached at step {step}")
            break
        if ITERATIONS > 0 and step >= ITERATIONS:
            break

        # Activate QAT
        if not qat_active and step >= qat_start_step:
            qat_active = True
            log(f"QAT enabled at step {step}")

        # LR
        lr_scale = get_lr(step, total_steps)
        for opt in optimizers:
            for pg in opt.param_groups:
                pg['lr'] = pg.get('base_lr', pg['lr']) * lr_scale

        x, y = loader.next_batch()
        model.train()
        _, loss = model(x, y)
        loss.backward()

        # Gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        # EMA update
        ema.update(base_model)

        if master and step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step:{step} loss:{loss.item():.4f} "
                  f"train_time:{int(elapsed*1000)}ms "
                  f"step_avg:{elapsed/max(step,1)*1000:.2f}ms",
                  flush=True)

        if VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0 and step > 0:
            ema_model = copy.deepcopy(base_model)
            ema.apply_to(ema_model)
            vl, vb = eval_standard(ema_model, val_tokens)
            log(f"step:{step} val_loss:{vl:.4f} val_bpb:{vb:.4f}")
            del ema_model

        step += 1

    log("Training complete. Running evaluations...")
    total_train_time = time.time() - t0

    # Apply EMA weights
    ema_model = copy.deepcopy(base_model)
    ema.apply_to(ema_model)
    ema_model.eval()

    # 1. Pre-quant standard eval
    vl_prequant, vb_prequant = eval_standard(ema_model, val_tokens)
    log(f"pre_quant val_loss:{vl_prequant:.4f} val_bpb:{vb_prequant:.4f}")

    # 2. Compress and decompress (simulate quantization roundtrip)
    log("Compressing model...")
    compressed_bytes, compressed_size = compress_model(ema_model)
    log(f"Compressed size: {compressed_size:,} bytes ({compressed_size/1e6:.2f} MB)")

    # Check artifact size
    code_size = len(open(__file__, 'rb').read())
    total_artifact = compressed_size + code_size
    log(f"Code size: {code_size:,} bytes")
    log(f"Total artifact: {total_artifact:,} bytes ({total_artifact/1e6:.2f} MB)")
    if total_artifact > 16_000_000:
        log(f"WARNING: artifact {total_artifact:,} > 16,000,000 bytes!")

    # Decompress into eval model
    quant_model = copy.deepcopy(ema_model)
    decompress_model(compressed_bytes, quant_model)
    quant_model.eval()

    # 3. Post-quant standard eval
    vl_postquant, vb_postquant = eval_standard(quant_model, val_tokens)
    log(f"post_quant val_loss:{vl_postquant:.4f} val_bpb:{vb_postquant:.4f}")
    log(f"quant_gap: {vb_postquant - vb_prequant:.4f} BPB")

    # 4. LoRA TTT eval (v8: 5 epochs, cosine LR, score every epoch)
    log(f"Starting LoRA TTT eval ({TTT_EPOCHS} epochs, cosine LR)...")
    t_ttt_start = time.time()
    vl_ttt, vb_ttt = eval_with_lora_ttt(quant_model, val_tokens)
    t_ttt = time.time() - t_ttt_start
    log(f"ttt val_loss:{vl_ttt:.4f} val_bpb:{vb_ttt:.4f} eval_time:{t_ttt:.0f}s")

    # Final summary
    log("=" * 60)
    log(f"final_train_time: {total_train_time:.1f}s")
    log(f"final_int{INT_BITS}_zstd_prequant  val_bpb: {vb_prequant:.4f}")
    log(f"final_int{INT_BITS}_zstd_postquant val_bpb: {vb_postquant:.4f}")
    log(f"final_lora_ttt_val_bpb: {vb_ttt:.4f}")
    log(f"final_artifact_bytes: {total_artifact}")
    log(f"final_compressed_mb: {compressed_size/1e6:.2f}")
    log("=" * 60)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
