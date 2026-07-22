"""WideBind Mini text generation with meta-cognitive mind readout."""
import os, sys, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn.functional as F
from core import WideBandConfig, WideBindStack

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=128, temperature=1.0, top_k=50,
             context_mem=None, allow_write=None, show_mind=False, stream=False):
    model.eval()
    device = next(model.parameters()).device
    L = model.cfg.seq_len
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    state = None
    recent = set()
    mind_log = []
    for step in range(max_new_tokens):
        ctx = tokens[-L:].unsqueeze(0)
        h = model.embed_tokens(ctx)
        out, state, _ = model(h, state, adaptive=True,
                              context_mem=context_mem, allow_write=allow_write)
        logits = model.lm_head(out[:, -1:, :])[0, 0]
        logits = logits / temperature
        for rid in list(recent)[-5:]:
            logits[rid] -= 2.0
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[-1:]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        recent.add(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=0)
        if show_mind and step % 10 == 0:
            layer_minds = {}
            for i, layer in enumerate(model.layers):
                lm = layer.mirror.debug_mind()
                if lm:
                    layer_minds[f'L{i}'] = lm
            mind_log.append({'step': step, 'token': next_token.item(), 'layers': layer_minds})
        if stream:
            print(next_token.item(), end=' ', flush=True)
    return tokens.tolist(), mind_log

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--show-mind', action='store_true', help='Show meta-cognitive stats during generation')
    parser.add_argument('--continuous-learn', action='store_true', help='Allow private memory writes during generation')
    parser.add_argument('--context-mem', type=str, default=None, help='Path to .pt file with context memory (G,k)')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state_dict.get('cfg', WideBandConfig())
    model = WideBindStack(cfg).to(device)
    model.load_state_dict(state_dict['model'], strict=False)
    model.eval()
    prompt_tokens = [int(t) for t in args.prompt.split(',')] if args.prompt else [1]
    context_mem = None
    if args.context_mem:
        context_mem = torch.load(args.context_mem, map_location=device)
    tokens, mind_log = generate(model, prompt_tokens, max_new_tokens=args.tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 context_mem=context_mem,
                                 allow_write=args.continuous_learn,
                                 show_mind=args.show_mind)
    print('\nGenerated tokens:', tokens)
    if args.show_mind:
        log_file = 'mind_log.json'
        import json
        # Convert tensors to floats for JSON
        def convert(o):
            if isinstance(o, (torch.Tensor,)):
                return o.tolist() if o.numel() > 1 else o.item()
            return o
        with open(log_file, 'w') as f:
            json.dump(mind_log, f, default=convert, indent=2)
        print(f'Mind log saved to {log_file}')