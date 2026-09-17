"""What can this GPU actually do?

Two jobs:

1. **Ask** Transformer Engine which FP8 recipes the hardware supports. This is not the
   same question as "is it new enough" -- support is genuinely patchy. An RTX PRO 6000
   (Blackwell, sm_120) supports NVFP4 but *not* MXFP8, while an H100 (Hopper, sm_90)
   supports neither. Hardcoding a list of recipes is how workshops break on other people's
   machines, so we ask at runtime.

2. **Measure** the peak matmul throughput, instead of hardcoding a number from a spec
   sheet. MFU (model FLOPS utilization) is a fraction, and a fraction is only as
   meaningful as its denominator. We get the denominator by timing big square GEMMs,
   which is the closest thing to "the fastest this GPU will ever multiply matrices".
"""

import json
import os
import time

import torch
import numpy as np

_CACHE = os.path.join(os.path.dirname(__file__), "peak_cache.json")


# --------------------------------------------------------------------------------------
# Device + Transformer Engine support
# --------------------------------------------------------------------------------------

def device_info():
    """Name, compute capability and memory of GPU 0."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; this workshop needs an NVIDIA GPU")
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "name": props.name,
        "compute_capability": f"{major}.{minor}",
        "sm_count": props.multi_processor_count,
        "memory_gb": round(props.total_memory / 1e9, 1),
        # Tensor-core FP8 arrives with Ada (8.9) and Hopper (9.0).
        "supports_fp8": (major, minor) >= (8, 9),
    }


def te_support():
    """Which FP8/FP4 recipes will Transformer Engine actually run here?

    Returns a dict of recipe-name -> (supported, reason). The reason string is TE's own
    explanation, which is worth showing students: it is usually specific and useful.
    """
    try:
        import transformer_engine.pytorch.fp8 as fp8mod
    except ImportError:
        return {"error": "transformer_engine is not installed"}

    def check(fn_name):
        fn = getattr(fp8mod, fn_name, None)
        if fn is None:
            return (False, f"this TE build has no {fn_name}()")
        ok, reason = fn()
        return (bool(ok), reason or "")

    fp8_ok, fp8_reason = check("check_fp8_support")
    return {
        # Per-tensor scaling: one scale factor for a whole tensor.
        "delayed": (fp8_ok, fp8_reason),   # scale from a history of recent maxima
        "current": (fp8_ok, fp8_reason),   # scale from this tensor, right now
        # Fine-grained scaling: many scale factors within one tensor.
        "block":   check("check_fp8_block_scaling_support")
                   if hasattr(fp8mod, "check_fp8_block_scaling_support") else (fp8_ok, fp8_reason),
        "block_rowwise": check("check_fp8_block_scaling_support")
                   if hasattr(fp8mod, "check_fp8_block_scaling_support") else (fp8_ok, fp8_reason),
        "mxfp8":   check("check_mxfp8_support"),
        "nvfp4":   check("check_nvfp4_support"),
    }


def make_fp8_recipe(name, amax_history_len=16):
    """Turn a recipe name from a config into a Transformer Engine recipe object.

    The recipe is the *rulebook* for converting tensors to FP8: how the scale factor is
    chosen, and how finely it is applied. Everything else about the model stays the same.
    """
    from transformer_engine.common import recipe as te_recipe

    name = name.lower()
    if name == "delayed":
        # Scale chosen from a rolling window of recent maxima. Cheap (no extra pass over
        # the tensor) but it is always reacting to slightly stale information.
        return te_recipe.DelayedScaling(
            fp8_format=te_recipe.Format.HYBRID,  # E4M3 forward, E5M2 for gradients
            amax_history_len=amax_history_len,
            amax_compute_algo="max",
        )
    if name == "current":
        # Scale computed from the tensor being cast, right now. Costs an extra reduction
        # but never lags behind a sudden change in activation scale.
        return te_recipe.Float8CurrentScaling()
    if name == "block":
        # Fine-grained scaling. TE's defaults here are worth spelling out, because "block
        # scaling" is really two different granularities at once:
        #
        #   x_block_scaling_dim    = 1  -> activations get ROW-WISE scales (1 x 128)
        #   w_block_scaling_dim    = 2  -> weights get 2D tile scales    (128 x 128)
        #   grad_block_scaling_dim = 1  -> gradients get ROW-WISE scales
        #
        # Row-wise means one scale per row of the activation matrix -- i.e. per token.
        # A token with unusually large activations then gets its own scale instead of
        # dragging every other token's values into a worse part of the FP8 range. This
        # is the scheme DeepSeek-V3 popularized.
        return te_recipe.Float8BlockScaling()
    if name == "block_rowwise":
        # Row-wise on BOTH sides: weights get one scale per row too, rather than 128x128
        # tiles. More scale factors to compute and carry, finer protection from outliers.
        # (2D x 2D is not a legal combination -- TE rejects it, because the GEMM needs at
        # least one operand with a scale layout it can consume directly.)
        return te_recipe.Float8BlockScaling(x_block_scaling_dim=1, w_block_scaling_dim=1)
    if name == "mxfp8":
        # Micro-scaling: one shared exponent per 32 values, defined by the OCP MX spec
        # and implemented in hardware on Blackwell data-center parts.
        return te_recipe.MXFP8BlockScaling()
    if name == "nvfp4":
        # Four-bit weights and activations with per-16-element scales. Even less precision
        # per number; the block scales are what keep it trainable.
        return te_recipe.NVFP4BlockScaling()
    raise ValueError(f"unknown fp8 recipe {name!r}")


# --------------------------------------------------------------------------------------
# Measured roofline
# --------------------------------------------------------------------------------------

def _time_gemm(n, precision, iters=50):
    """Median TFLOP/s for an n x n x n matmul at the given precision ('bf16' or 'fp8')."""
    if precision == "bf16":
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        run = lambda: torch.mm(a, b)
    elif precision == "fp32":
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        run = lambda: torch.mm(a, b)
    elif precision == "fp8":
        # torch._scaled_mm is the raw FP8 tensor-core GEMM: it multiplies two FP8 tensors
        # and applies their scale factors. This is the same hardware path TE drives, with
        # none of the scaling bookkeeping, so it measures the ceiling rather than a recipe.
        a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        # The second operand must be column-major. On Hopper/Blackwell cuBLAS accepts
        # either layout, but on Ada (sm_89, e.g. L40S) a row-major B is rejected outright
        # with CUBLAS_STATUS_NOT_SUPPORTED -- so build it transposed here rather than
        # discovering it as a crash in the first cell of notebook 01.
        b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")
        run = lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    else:
        raise ValueError(f"unknown precision {precision!r}")

    for _ in range(10):                      # warm up: let clocks boost, cache the kernel
        run()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    secs = (time.perf_counter() - start) / iters

    # An n x n by n x n matmul does n^3 multiply-adds = 2 * n^3 floating point ops.
    return 2 * n**3 / secs / 1e12

def measure_peak_tflops(precision="bf16", refresh=False, sizes=(2048, 4096, 4096*2, 4096*4)):
    """Measure this GPU's achievable dense matmul throughput, in TFLOP/s.

    `precision` selects the roofline: 'bf16' or 'fp8'. Cached to disk, because it takes a
    few seconds and never changes for a given machine.

    These are deliberately *achievable* numbers from real kernels, not marketing peaks --
    which is what makes MFU an honest "how close are we to the best this GPU does".

    Why two rooflines: MFU is a fraction, and a fraction needs the right denominator. An
    FP8 run measured against the BF16 peak looks better than it is, because FP8 tensor
    cores are roughly twice as fast. Each run should be judged against the ceiling of the
    precision it actually used.
    """
    key = f"{precision}_tflops"
    cached = {}
    if os.path.exists(_CACHE):
        with open(_CACHE) as f:
            cached = json.load(f)
        if cached.get("device") != torch.cuda.get_device_name(0):
            cached = {}
    if not refresh and key in cached:
        return cached[key]

    peak = max(_time_gemm(n, precision, iters=20) for n in sizes)
    cached["device"] = torch.cuda.get_device_name(0)
    cached[key] = peak
    with open(_CACHE, "w") as f:
        json.dump(cached, f)
    return peak

def peak_for_run(fp8):
    """The right MFU denominator for a run, and a label naming it.

    Returns (tflops, label). A BF16 run is judged against the BF16 roofline; an FP8 run
    against the FP8 one. This means FP8 runs usually show a *lower* MFU than the BF16
    baseline even while being faster in wall-clock terms -- which is the honest reading:
    they are delivering more tokens per second while using a smaller fraction of what
    their precision makes available.
    """
    precision = "fp8" if fp8 else "bf16"
    return measure_peak_tflops(precision), precision


def summary():
    """One-stop report, for the top of each notebook."""
    info = device_info()
    lines = [
        f"GPU              : {info['name']}",
        f"compute capability: sm_{info['compute_capability'].replace('.', '')}",
        f"SMs / memory     : {info['sm_count']} SMs, {info['memory_gb']} GB",
        f"measured BF16 peak: {measure_peak_tflops('bf16'):.0f} TFLOP/s (dense matmul)",
        f"measured FP8 peak : {measure_peak_tflops('fp8'):.0f} TFLOP/s (dense matmul)",
        "",
        "Transformer Engine FP8 recipe support:",
    ]
    sup = te_support()
    if "error" in sup:
        lines.append(f"  {sup['error']}")
    else:
        for name, (ok, reason) in sup.items():
            mark = "yes" if ok else "no "
            note = "" if ok else f"  ({reason})"
            lines.append(f"  {name:8s} {mark}{note}")
    return "\n".join(lines)


def supported_recipes():
    """Just the recipe names that will run here -- handy for looping in a notebook."""
    sup = te_support()
    if "error" in sup:
        return []
    return [name for name, (ok, _) in sup.items() if ok]


"""GPT-2's byte-pair tokenizer.

The model does not read text; it reads integers. This module is what turns one into the
other, using GPT-2's original byte-pair encoding: a vocabulary of 50257 pieces, where
common words are a single token and rare ones are split into several.

We use the `tokenizers` library, which ships in the container, and read GPT-2's vocabulary
from `data/tokenizer.json` (downloaded by `make data`). Nothing here touches the network,
so the workshop runs the same on a machine with no internet access.

The wrapper exists so the rest of the code can say `enc.encode(text)` and `enc.decode(ids)`
without caring which library is underneath.
"""

import os

from tokenizers import Tokenizer

# 50256 is <|endoftext|>: the marker the model learns to emit when a story is finished.
# GPT-2's vocabulary is 50257 tokens, 0..50256.
EOT_TOKEN = 50256
VOCAB_SIZE = 50257


class GPT2Tokenizer:
    """Text <-> token ids, with the small slice of the API this workshop needs."""

    def __init__(self, path):
        self._tok = Tokenizer.from_file(path)
        self.eot_token = EOT_TOKEN

    def encode(self, text):
        """One string -> a list of token ids."""
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        """Many strings at once. Much faster than a loop: the encoding happens in Rust
        across several threads, which is why this file needs no multiprocessing."""
        return [e.ids for e in self._tok.encode_batch_fast(texts, add_special_tokens=False)]

    def decode(self, ids):
        """Token ids -> the string they spell.

        `skip_special_tokens=False` keeps <|endoftext|> in the output, so callers can see
        where the model decided to stop.
        """
        return self._tok.decode(list(ids), skip_special_tokens=False)


def get_tokenizer(data_dir="data"):
    """Load the tokenizer from `data_dir`, with a message that says how to get it."""
    path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `make data` from the repository root, which "
            f"downloads GPT-2's vocabulary along with the training text."
        )
    return GPT2Tokenizer(path)

data = {}
data['train'] = np.memmap(f"data/train.bin", dtype=np.uint16, mode="r")
data['val'] = np.memmap(f"data/val.bin", dtype=np.uint16, mode="r")
def get_batch(split, cfg, batch_size, device="cuda"):
    """A random batch of (inputs, targets) from the token stream."""
    D = data[split]
    i = torch.randint(len(D) - cfg.seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(D[j:j + cfg.seq_len].astype(np.int64)) for j in i])
    y = torch.stack([torch.from_numpy(D[j + 1:j + 1 + cfg.seq_len].astype(np.int64)) for j in i])
    return x.to(device), y.to(device)

## From Notebook 01
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TorchBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.fc_1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc_2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def attention(self, qkv):
        B, T, C3 = qkv.shape
        C = C3 // 3
        q, k, v = (z.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for z in qkv.split(C, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

    def forward(self, x):
        x = x + self.attention(self.qkv(self.ln_1(x)))
        return x + self.fc_2(F.gelu(self.fc_1(self.ln_2(x)), approximate="tanh"))


class MiniGPT(nn.Module):
    RESIDUAL = ("proj.weight", "fc_2.weight", "fc2_weight")

    def __init__(self, cfg, Block=TorchBlock):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight
        self.init_weights()

    def init_weights(self):
        small = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if p.dim() < 2:
                # Biases. nn.Linear seeds them randomly while te.Linear zeroes them, so we
                # set them explicitly -- otherwise the block types start from different
                # models and the comparison is measuring two things at once. (GPT-2 uses
                # zero biases anyway.) LayerNorm gains are already ones; leave them.
                if name.endswith("bias") and "layer_norm" not in name and "ln" not in name:
                    nn.init.zeros_(p)
                continue
            nn.init.normal_(p, std=small if name.endswith(self.RESIDUAL) else 0.02)

    def forward(self, idx, targets=None):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

def warmup():
    _time_gemm(4096*8, "bf16", iters=100)

import transformer_engine.pytorch as te
class TEUnfusedBlock(nn.Module):
    """Every PyTorch module replaced one-for-one by its Transformer Engine equivalent.

    "Unfused" means the LayerNorms are still separate modules -- that is what notebook 03
    changes. The attention is a fused kernel in both versions.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.ln_1 = te.LayerNorm(cfg.n_embd)
        self.qkv  = te.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.attn = te.DotProductAttention(
            cfg.n_head, cfg.n_embd // cfg.n_head, attention_dropout=0.0,
            qkv_format="bshd", attn_mask_type="causal")
        self.proj = te.Linear(cfg.n_embd, cfg.n_embd)
        self.ln_2 = te.LayerNorm(cfg.n_embd)
        self.fc_1 = te.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc_2 = te.Linear(4 * cfg.n_embd, cfg.n_embd)

    def attention(self, qkv):
        B, T, C3 = qkv.shape
        C = C3 // 3
        q, k, v = (z.view(B, T, self.n_head, C // self.n_head) for z in qkv.split(C, dim=2))
        return self.proj(self.attn(q, k, v).view(B, T, C))

    def forward(self, x):                      # identical to TorchBlock.forward
        x = x + self.attention(self.qkv(self.ln_1(x)))
        return x + self.fc_2(F.gelu(self.fc_1(self.ln_2(x)), approximate="tanh"))

BATCH = 24
import contextlib
from transformer_engine.common import recipe as te_recipe
def fp8_context(fp8, recipe="delayed"):
    """FP8 wraps the FORWARD pass only.

    TE records the scaling factors it needs during forward and reuses them in backward,
    which runs outside this context. Wrapping `loss.backward()` too is a common mistake.
    """
    if not fp8:
        return contextlib.nullcontext()
    if recipe == "delayed":
        return te.autocast(enabled=True, recipe=te_recipe.DelayedScaling())
    return te.autocast(enabled=True, recipe=te_recipe.Float8CurrentScaling())
    
def train(model, cfg, steps=40, batch_size=BATCH, lr=6e-4, fp8=False,
          recipe="delayed", log=False, do_warmup=False):
    """The loop from notebook 01, with one addition: an optional FP8 context."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    model.train()
    if do_warmup: warmup()
    t0, tokens, last = None, 0, float("nan")
    tps = []
    for step in range(steps):
        x, y = get_batch("train", cfg, batch_size)
        # Add FP8 context
        with torch.autocast("cuda", dtype=torch.bfloat16), fp8_context(fp8, recipe):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            torch.cuda.synchronize(); t0, tokens = time.perf_counter(), 0
        elif step % 10 == 0:
            torch.cuda.synchronize();
            ttime = time.perf_counter()-t0
            tps.append(tokens/ttime)
            t0, tokens = time.perf_counter(), 0
            torch.cuda.synchronize();
        tokens += x.numel()  
        last = loss.item()
    torch.cuda.synchronize()
    time.sleep(3)
    return {"loss": last, "tok_per_s": statistics.median(tps[-4:])}