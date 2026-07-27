"""VSA utilities: DCT basis, Zeckendorf codes, Fibonacci sigmoid init, sparse codes, prefix scan."""

import math
import torch


def dct_basis(n):
    k = torch.arange(n, dtype=torch.float32)
    v = k.unsqueeze(1) * (k.unsqueeze(0) + 0.5)
    basis = torch.cos(v * math.pi / n)
    basis[0, :] = basis[0, :] / math.sqrt(2)
    return basis * math.sqrt(2.0 / n)


def zeckendorf_codes(vocab=50000):
    fib = [1, 2]
    while fib[-1] <= vocab:
        fib.append(fib[-1] + fib[-2])
    fib = fib[:-1]
    K = len(fib)
    codes = torch.zeros(vocab, K)
    for i in range(vocab):
        n = i + 1
        for j in range(K - 1, -1, -1):
            if n >= fib[j]:
                codes[i, j] = 1.0
                n -= fib[j]
    return codes


def fib_sigmoid_init(n, fib_vals=None):
    if fib_vals is None:
        f = [1, 1]
        while len(f) < n:
            f.append(f[-1] + f[-2])
        fib_vals = f[:n]
    fib = torch.tensor(fib_vals, dtype=torch.float32)
    p = fib / fib.sum()
    bias = torch.log(p / (1 - p + 1e-10))
    return bias


def sparse_block_codes(vocab=50000, K=32, S=6):
    from math import comb
    total = comb(K, S)
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(42))
    codes = torch.zeros(vocab, K)
    for v in range(vocab):
        idx = int(perm[v].item())
        n = idx
        for i in range(S, 0, -1):
            c = i - 1
            while comb(c + 1, i) <= n:
                c += 1
            codes[v, c] = 1.0
            n -= comb(c, i)
    return codes


def vsa_prefix_scan(a, b, state=None):
    B, L, D = b.shape
    if a.dim() == 2:
        a = a.unsqueeze(-1).expand(-1, -1, D)
    eps = 1e-6
    CHUNK = 32
    out = []
    s = state.clone() if state is not None else None
    for start in range(0, L, CHUNK):
        end = min(start + CHUNK, L)
        b_chunk = b[:, start:end]
        a_chunk = a[:, start:end]
        log_a_chunk = torch.log(a_chunk.clamp(min=eps))
        log_cum_chunk = torch.cumsum(log_a_chunk, dim=1)
        cum_decay_chunk = torch.exp(log_cum_chunk)
        inv_cum_decay_chunk = (1.0 / cum_decay_chunk.clamp(min=eps)).clamp(max=1e6)
        weighted = b_chunk * inv_cum_decay_chunk
        cum_weighted = torch.cumsum(weighted, dim=1)
        if s is not None:
            result_chunk = cum_decay_chunk * s.unsqueeze(1) + cum_decay_chunk * cum_weighted
        else:
            result_chunk = cum_decay_chunk * cum_weighted
        out.append(result_chunk)
        s = result_chunk[:, -1]
    result = torch.cat(out, dim=1)
    return result, result[:, -1]
