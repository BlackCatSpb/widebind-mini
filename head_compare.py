import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\black\OneDrive\Desktop\WideBind Mini')

from core.config import WideBindConfig
from core.stack import WideBindStack


def make_cfg(head, seq, batch, d=384, layers=6, vocab=65536, norm=True):
    return WideBindConfig(
        D=d,
        n_layers=layers,
        vocab=vocab,
        seq_len=seq,
        batch_size=batch,
        head_mode=head,
        head_normalize=norm,
        amp_codec=(head == 'codec'),
    )


def gen_data(seed, n_batches, cfg, noise=0.15):
    rng = torch.Generator().manual_seed(seed)
    V = cfg.vocab
    xs, ys = [], []
    for _ in range(n_batches):
        x = torch.randint(0, V, (cfg.batch_size, cfg.seq_len), generator=rng)
        y = (x + 1) % V
        noise_mask = torch.rand(x.shape, generator=rng) < noise
        y = torch.where(noise_mask, torch.randint(0, V, x.shape, generator=rng), y)
        xs.append(x)
        ys.append(y)
    return xs, ys


def make_model(cfg, seed=42):
    torch.manual_seed(seed)
    return WideBindStack(cfg)


def nll(model, h_out, y, h_emb=None):
    if getattr(model.cfg, 'amp_codec', False) and hasattr(model.lm_head, 'ce_loss'):
        # Codec: CE-нормализованный NLL (LSE), калиброванный — не факторизованный NLL.
        he = None if h_emb is None else h_emb.reshape(-1, model.cfg.D)
        ce = model.lm_head.ce_loss(h_out.reshape(-1, model.cfg.D), y.reshape(-1), he)
        return ce.mean().item()
    if hasattr(model.lm_head, 'log_probs_for_target'):
        logp = model.lm_head.log_probs_for_target(h_out, y)
        return -logp.mean().item()
    logits = model.lm_head(h_out)
    return F.cross_entropy(logits.reshape(-1, model.cfg.vocab), y.reshape(-1)).item()


@torch.no_grad()
def eval_rank(model, h_out, y, n_pos=200, n_cand=64, seed=1):
    B, L, _ = h_out.shape
    rng = torch.Generator().manual_seed(seed)
    pb = torch.randint(0, B, (n_pos,), generator=rng, device='cpu').to(h_out.device)
    pt = torch.randint(0, L, (n_pos,), generator=rng, device='cpu').to(h_out.device)
    y_sel = y[pb, pt]
    cand = torch.randint(0, model.cfg.vocab, (n_pos, n_cand - 1),
                         generator=rng, device='cpu').to(h_out.device)
    cands = torch.cat([y_sel.view(-1, 1), cand], dim=1)  # col 0 = true token
    if getattr(model.cfg, 'amp_codec', False) and hasattr(model.lm_head, 'forward'):
        # Codec: счётка linear (⟨z_v, logit⟩) — согласовано с ce_loss/argmax.
        lg = model.lm_head(h_out)[pb, pt]
        scores = lg.gather(1, cands)
    elif hasattr(model.lm_head, 'log_probs_for_target'):
        h_sel = h_out[pb, pt].unsqueeze(1)
        scores = model.lm_head.log_probs_for_target(h_sel, cands).squeeze(1)
    else:
        lg = model.lm_head(h_out)[pb, pt]
        scores = lg.gather(1, cands)
    true = scores[:, 0]
    acc = (scores.argmax(dim=1) == 0).float().mean().item()
    tie_frac = ((scores >= true.view(-1, 1)).sum(dim=1) == 1).float().mean().item()
    return acc, tie_frac


def run(cfg, xs, ys, exs, eys, steps, lr, device, log_every=200):
    model = make_model(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    rows = []
    for step in range(1, steps + 1):
        frac = min(step / 100.0, 1.0)
        for g in opt.param_groups:
            g['lr'] = lr * frac
        xi = (step - 1) % len(xs)
        xt = model.embed_tokens(xs[xi].to(device))
        yt = ys[xi].to(device)
        out, _, _ = model(xt)
        loss = model.compute_loss(out, yt, h_emb=xt)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_nll = sum(nll(model, model(hx)[0], ey.to(device), h_emb=hx)
                              for (ex, ey) in zip(exs, eys)
                              for hx in [model.embed_tokens(ex.to(device))]) / len(exs)
                acc = 0.0
                tie = 0.0
                for ex, ey in zip(exs, eys):
                    hx = model.embed_tokens(ex.to(device))
                    ho, _, _ = model(hx)
                    a, t = eval_rank(model, ho, ey.to(device))
                    acc += a
                    tie += t
                acc /= len(exs)
                tie /= len(exs)
            model.train()
            dt = time.time() - t0
            tok_s = step * cfg.batch_size * cfg.seq_len / dt
            rows.append(dict(step=step, train_ce=round(loss.item(), 4),
                             val_nll=round(val_nll, 4), rank64=round(acc, 4),
                             ties=round(tie, 4)))
            print(f'step={step:5d} train_ce={loss.item():.4f} val_nll={val_nll:.4f} '
                  f'rank@64={acc:.3f} grad={grad_norm:.2f} tok/s={tok_s:.0f}')
    return model, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='*',
                    default=['partitioned', 'sigmoid_coded', 'cognitive_coded', 'codec'])
    ap.add_argument('--steps', type=int, default=2200)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--vocab', type=int, default=65536)
    ap.add_argument('--log', type=int, default=200)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--norm', type=int, default=1, help='cognitive head: 1=normalized (calibrated), 0=strict Bernoulli')
    args = ap.parse_args()

    device = torch.device(args.device)
    device = device if device.type == 'cpu' else device
    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    print('device:', device, 'cuda avail:', torch.cuda.is_available())

    results = {}
    for head in args.heads:
        print(f'\n=== HEAD: {head} ===', flush=True)
        cfg = make_cfg(head, args.seq, args.batch, vocab=args.vocab,
                       norm=bool(args.norm))
        xs, ys = gen_data(7, 240, cfg)
        exs, eys = gen_data(999, 8, cfg)
        _, rows = run(cfg, xs, ys, exs, eys, args.steps, args.lr, device,
                      log_every=args.log)
        results[head] = rows

    out = r'C:\Users\black\OneDrive\Desktop\WideBind Mini\head_compare_results.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\n===== SUMMARY (last eval) =====')
    for h, rows in results.items():
        last = rows[-1]
        print(f'{h:14s} val_nll={last["val_nll"]:8.4f}  rank@64={last["rank64"]:.3f}')


if __name__ == '__main__':
    main()
