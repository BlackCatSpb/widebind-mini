"""
probe_concepts.py — offline probe of the Collective Concept Layer on WideBand Mini.

Tests three things WITHOUT touching training:
  A) Layer readiness: what does the model's OWN quality signal say about each layer?
     (residual-var EMA, delta-var stability, gate EMA, signal norms)
     -> mining should only be enabled on MATURE layers (the "child doesn't build a
        nuclear reactor" guard: don't mine on layers whose representations are random).
  B) Gap detection in K-space: tokens whose per-expert K-state is far from the layer's
     existing private-mem prototypes (uncovered regions = candidate new concepts).
  C) Decode-probe: project born candidate slots (and existing prototypes) back to D-space
     -> lm_head -> top-k tokens -> BPE pieces. Do they name real words?

Usage:
    python scripts/probe_concepts.py --ckpt checkpoints/best.pt \
        --data wb/token_stream_ADVENTUR_clean.bin --batches 120
"""
import os, sys, math, glob, argparse, gc
import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core import WideBindStack
from tokenizers import Tokenizer

TOKENIZER_PATH = r'C:\Users\black\OneDrive\Desktop\FCP\russian_tokenizer\tokenizer.json'
CYR = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')


def load_tokenizer(path=TOKENIZER_PATH):
    tok = Tokenizer.from_file(path)
    return tok


def fix_utf8(s):
    """tokenizers' id_to_token returns UTF-8 bytes as latin-1 chars. Repair."""
    try:
        b = s.encode('latin-1')
        return b.decode('utf-8', errors='replace')
    except Exception:
        return s


def is_cyrillic_word(piece):
    s = clean_piece(fix_utf8(piece))
    if not s:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.lower() in CYR) / len(letters) >= 0.9


def clean_piece(piece):
    s = piece.replace('Ġ', '')
    s = s.replace('##', '')
    s = s.replace('</s>', '').replace('<s>', '').replace('<pad>', '').replace('<unk>', '')
    return s.strip()


def decode_topk(tok, logits, k=20):
    vals, ids = torch.topk(logits, k)
    out = []
    for v, i in zip(vals.tolist(), ids.tolist()):
        piece = tok.decode([i], skip_special_tokens=True) if i < tok.get_vocab_size() else ''
        out.append((i, clean_piece(piece), v, is_cyrillic_word(piece)))
    return out


class TokenStream:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.len = len(self.data)

    def get_batch(self, seq_len, batch_size, offset):
        need = batch_size * seq_len + 1
        if offset + need > self.len:
            offset = 0
        chunk = self.data[offset:offset + need]
        x = torch.from_numpy(chunk[:batch_size * seq_len].copy()).long().view(batch_size, seq_len)
        y = torch.from_numpy(chunk[1:batch_size * seq_len + 1].copy()).long().view(batch_size, seq_len)
        return x, y, offset + batch_size * seq_len


@torch.no_grad()
def collect_signals(model, stream, seq_len, n_batches, device):
    """Feed real text, record per-layer quality signals the model computes itself."""
    model.eval()
    G = model.cfg.mlp_groups
    n_layers = len(model.layers)
    rec = {i: {'delta_var': [], 'resvar_mean': [], 'gate_ema_mean': [],
               'gate_sel': [], 'mag': [], 'signals': []} for i in range(n_layers)}

    state = gs = None
    offset = n_batches * seq_len  # start mid-stream to skip header/boom
    hp_pool = {i: [] for i in range(n_layers)}  # pooled K-states for gap detection

    for b in range(n_batches):
        x, y, offset = stream.get_batch(seq_len, 1, offset)
        x = x.to(device)
        h = model.embed_tokens(x)
        out, state, gs = model(h, state, global_state=gs, adaptive=False)
        for i, layer in enumerate(model.layers):
            m = layer.mirror
            rec[i]['delta_var'].append(m._delta_var.detach().cpu().clone())
            rec[i]['resvar_mean'].append(m._residual_var_ema.mean().item())
            rec[i]['gate_ema_mean'].append(m._gate_ema.mean().item())
            rec[i]['gate_sel'].append(m._last_gates.std().item())
            rec[i]['mag'].append(m._last_magnitude.item())
            hp = m._cached_hp.detach().cpu()
            hp_pool[i].append(hp.reshape(-1, hp.shape[-2], hp.shape[-1]))
        if b % 20 == 0:
            print(f'  batch {b}/{n_batches} done')
    return rec, hp_pool


def layer_readiness(rec, n_layers):
    """Rank layers by their own signal: mature = settled residual, stable delta_var,
    differentiated gates."""
    resvar = [np.mean(rec[i]['resvar_mean']) for i in range(n_layers)]
    dvar_stab = []
    for i in range(n_layers):
        dv = np.stack([r.numpy() for r in rec[i]['delta_var']])  # (T, G)
        stab = dv.std(axis=0).mean() / (dv.mean() + 1e-8)
        dvar_stab.append(stab)
    gate_sel = [np.mean(rec[i]['gate_sel']) for i in range(n_layers)]
    gate_mean = [np.mean(rec[i]['gate_ema_mean']) for i in range(n_layers)]

    # z-score
    def z(v):
        a = np.array(v, dtype=float)
        return (a - a.mean()) / (a.std() + 1e-8)

    score = z([-r for r in resvar]) + z([-d for d in dvar_stab]) + z(gate_sel)
    return {'resvar': resvar, 'dvar_stab': dvar_stab, 'gate_sel': gate_sel,
            'gate_mean': gate_mean, 'readiness': score.tolist()}


@torch.no_grad()
def gap_detection(model, layer_idx, hp_pool, tau_base=0.35, variance_k=2.0, S=8):
    """Find uncovered regions of K-space vs the layer's existing private-mem prototypes."""
    m = model.layers[layer_idx].mirror
    pm = m._private_mem.detach().cpu()  # (G, k)
    pm_n = F.normalize(pm, dim=-1)
    G, k = pm.shape
    pool = torch.cat(hp_pool[layer_idx], dim=0)  # (N, G, k)
    print(f'\n  Layer {layer_idx}: K-state pool {pool.shape}, prototypes {pm.shape}')

    # per-expert min cosine distance to prototypes, mean over experts
    flat = pool.reshape(-1, G, k)
    d_min = torch.zeros(flat.shape[0])
    for g in range(G):
        sim = flat[:, g] @ pm_n[g]
        d_min = d_min + (1.0 - sim) / G
    # adaptive threshold (design: tau = tau_base * (1 + variance_k * sigma_d))
    sigma_d = d_min.std().item()
    tau = tau_base * (1.0 + variance_k * sigma_d)
    gap_mask = d_min > tau
    frac = gap_mask.float().mean().item()
    print(f'  d_min: mean={d_min.mean().item():.4f} std={sigma_d:.4f} tau={tau:.4f} '
          f'gap_fraction={frac*100:.2f}%')

    if gap_mask.sum() == 0:
        return [], tau, frac

    gap_states = flat[gap_mask]  # (Ngap, G, k)
    gap_flat = gap_states.reshape(gap_states.shape[0], -1)
    n_clust = min(S, int(gap_states.shape[0] // 100) + 1)
    if gap_states.shape[0] < n_clust:
        n_clust = max(1, gap_states.shape[0])
    # simple farthest-point-ish seeding via one KMeans pass (k-means++ not needed for probe)
    centroids = gap_flat[:n_clust].clone()
    for _ in range(3):
        c_n = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
        sim = gap_flat @ c_n.T
        lab = sim.argmax(dim=1)
        for s in range(n_clust):
            pts = gap_flat[lab == s]
            if pts.shape[0] > 0:
                centroids[s] = pts.mean(dim=0)
    centroids = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
    return centroids.view(n_clust, G, k), tau, frac


def decode_concept(model, layer_idx, concept, tok, topk=20):
    """Project a K-space concept (G,k) -> D-space via mirror W_out -> lm_head -> top-k."""
    m = model.layers[layer_idx].mirror
    device = next(model.parameters()).device
    concept = concept.to(device)
    # per-expert k -> d, concat -> D
    d_vecs = torch.einsum('gk,gkd->gd', concept, m.W_out)  # (G, d)
    d_vec = d_vecs.reshape(-1)  # (D,)
    d_vec = F.rms_norm(d_vec, (d_vec.shape[-1],))
    logits = model.lm_head(d_vec.view(1, 1, -1))[0, 0]  # (vocab,)
    return decode_topk(tok, logits, topk)


def report_concept(name, rows):
    n_w = sum(1 for _, _, _, isw in rows if isw)
    print(f'  [{name}] word_rate={n_w}/{len(rows)}')
    for tid, piece, v, isw in rows[:8]:
        mark = 'W' if isw else ' '
        print(f'    #{tid:<6} {piece!r:<24} {v:8.3f} {mark}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'C:\Users\black\OneDrive\Desktop\WideBand Mini\checkpoints\best.pt')
    ap.add_argument('--data', default=r'C:\Users\black\OneDrive\Desktop\WideBand Mini\wb\token_stream_ADVENTUR_clean.bin')
    ap.add_argument('--batches', type=int, default=120)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get('cfg')
    model = WideBindStack(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    n = model.param_count()
    print(f'Model: {n:,} params | step={ckpt.get("step")} best_val={ckpt.get("best_val_loss"):.4f}')

    tok = load_tokenizer()
    stream = TokenStream(args.data)
    print(f'Data: {stream.len:,} tokens')

    print('\n[A] Collecting per-layer quality signals...')
    rec, hp_pool = collect_signals(model, stream, args.seq, args.batches, device)
    n_layers = len(model.layers)

    rd = layer_readiness(rec, n_layers)
    print(f'\n=== A. Layer readiness (model\'s own signal) ===')
    print(f'{"L":>3} {"resvar":>9} {"dvar_stab":>10} {"gate_sel":>9} {"gate_ema":>9} {"readiness":>9}')
    order = sorted(range(n_layers), key=lambda i: rd['readiness'][i], reverse=True)
    for i in order:
        print(f'{i:>3} {rd["resvar"][i]:9.4f} {rd["dvar_stab"][i]:10.4f} '
              f'{rd["gate_sel"][i]:9.4f} {rd["gate_mean"][i]:9.4f} {rd["readiness"][i]:9.3f}')

    # gate profile by depth (design prediction: sigma(g_l) grows with depth)
    print(f'\n  gate_selectivity profile by depth: ' +
          ' '.join(f'{rd["gate_sel"][i]:.3f}' for i in range(n_layers)))

    probe = [order[0], order[len(order)//2], order[-1]]
    print(f'\n  probe layers: most mature={order[0]}, mid={order[len(order)//2]}, '
          f'least mature={order[-1]}')

    print('\n[B] Gap detection + [C] decode-probe on selected layers...')
    for li in probe:
        candidates, tau, frac = gap_detection(model, li, hp_pool)
        # decode existing prototypes too
        print(f'\n=== C. Decode-probe Layer {li} ===')
        pm = model.layers[li].mirror._private_mem.detach().cpu()
        print('  -- existing private-mem prototypes (what the model already has):')
        for s in range(min(4, pm.shape[0])):
            expert_concept = torch.zeros_like(pm)
            expert_concept[s] = pm[s]
            report_concept(f'proto {s}', decode_concept(model, li, expert_concept, tok))
        print(f'  -- {len(candidates)} born candidate slot(s) from gaps:')
        for s, c in enumerate(candidates[:4]):
            report_concept(f'born {s}', decode_concept(model, li, c, tok))


if __name__ == '__main__':
    main()
