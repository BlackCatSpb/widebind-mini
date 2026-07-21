"""
FCF-CPR: Fractal Cognitive Field CheckPoint Reduction.
Compresses WideBind .pt files: removes deterministic buffers, 
quantizes real weights with uniform 8-bit per tensor.
"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch


REMOVABLE_PATTERNS = {'V_dct', 'codes'}

def is_removable(k):
    return any(p in k for p in REMOVABLE_PATTERNS)

def is_scalar_gate(k, v=None):
    """True for b_i/b_d ONLY if tensor is still uniform (safe to scalar-fold)."""
    if not ('b_i' in k or 'b_d' in k):
        return False
    if v is not None and v.numel() > 1 and v.dtype.is_floating_point and v.std().item() >= 1e-8:
        return False  # non-uniform — use Tier W quantization instead
    return True


def quantize_tensor(t, n_bits=8):
    """Uniform quantization: store tensor as uint8 + min + scale.
    Returns (indices, min_val, scale)."""
    n_levels = 2 ** n_bits
    t_flat = t.float().flatten()
    t_min = t_flat.min().item()
    t_max = t_flat.max().item()
    
    if t_min == t_max:
        # Constant tensor: just store the value
        return None, t_min, 0.0
    
    scale = (t_max - t_min) / (n_levels - 1)
    indices = ((t_flat - t_min) / scale).round_().clamp_(0, n_levels - 1).to(torch.uint8)
    indices = indices.reshape(t.shape)
    return indices, t_min, scale


def dequantize_tensor(indices, t_min, scale, dtype=torch.float32):
    """Restore fp32 from uint8 + min + scale."""
    if indices is None:
        return torch.tensor(t_min, dtype=dtype)
    restored = indices.float() * scale + t_min
    return restored.to(dtype)


def quantize_tensor_channel(t, dim=0, n_bits=8):
    """Per-channel uniform quantization: each slice along dim gets own min/scale.
    For 2D weight (M, N): each row gets 256 levels instead of sharing across all M×N.
    Returns (indices, mins, scales)."""
    t_f = t.float()
    n_levels = 2 ** n_bits
    n_ch = t_f.shape[dim]
    view_shape = [t_f.ndim] * n_ch  # fancy indexing not needed
    mins = []
    scales = []
    indices_parts = []
    for i in range(n_ch):
        sl = t_f.select(dim, i)
        sl_min = sl.min().item()
        sl_max = sl.max().item()
        if sl_min == sl_max:
            mins.append(sl_min)
            scales.append(0.0)
            idx = torch.full(sl.shape, 0, dtype=torch.uint8, device=t.device)
        else:
            sc = (sl_max - sl_min) / (n_levels - 1)
            idx = ((sl - sl_min) / sc).round_().clamp_(0, n_levels - 1).to(torch.uint8)
            mins.append(sl_min)
            scales.append(sc)
        indices_parts.append(idx.unsqueeze(dim))
    indices = torch.cat(indices_parts, dim=dim) if n_ch > 0 else t.new_empty(t.shape, dtype=torch.uint8)
    return indices, torch.tensor(mins), torch.tensor(scales)


def dequantize_tensor_channel(indices, mins, scales, orig_shape, dtype=torch.float32):
    """Restore fp32 from per-channel uint8 + mins + scales."""
    restored = torch.zeros(orig_shape, dtype=dtype)
    n_ch = mins.shape[0]
    dim = 0 if orig_shape[0] == n_ch else (1 if len(orig_shape) > 1 and orig_shape[1] == n_ch else 0)
    for i in range(n_ch):
        if scales[i] == 0.0:
            sl = torch.full((orig_shape[1] if dim == 0 else orig_shape[0],), mins[i].item(), dtype=dtype)
        else:
            sl = indices.select(dim, i).float() * scales[i] + mins[i]
        restored.select(dim, i).copy_(sl)
    return restored


def analyze_sd(sd):
    """Print detailed analysis of state dict with size breakdown."""
    total_elems = sum(p.numel() for p in sd.values())
    total_bytes = sum(p.numel() * p.element_size() for p in sd.values())
    
    groups = {
        'removable': [], 'scalar_gates': [],
        'constant': [], 'weights': [], 
    }
    
    for k, v in sd.items():
        if is_removable(k):
            groups['removable'].append((k, v))
        elif is_scalar_gate(k, v):
            groups['scalar_gates'].append((k, v))
        elif v.numel() == 1 or (v.dtype.is_floating_point and v.std().item() < 1e-8 and v.numel() > 1):
            groups['constant'].append((k, v))
        elif not v.dtype.is_floating_point:
            groups['constant'].append((k, v))
        else:
            groups['weights'].append((k, v))
    
    print(f'Total: {total_elems:,} elems, {total_bytes/1e9:.2f} GB')
    print()
    
    total_compressed = 0
    total_original = 0
    for name, tensors in groups.items():
        if not tensors:
            continue
        orig = sum(v.numel() * v.element_size() for _, v in tensors)
        elems = sum(v.numel() for _, v in tensors)
        total_original += orig
        
        if name == 'removable':
            comp = 0
            print(f'  {name:15s}: {orig/1e6:>8.1f} MB  elems={elems:>9,}  -> 0 MB (recompute)')
        elif name == 'scalar_gates':
            comp = len(tensors) * 4
            print(f'  {name:15s}: {orig/1e6:>8.1f} MB  elems={elems:>9,}  -> {comp/1e6:.4f} MB (scalar)')
        elif name == 'constant':
            comp = len(tensors) * 4
            print(f'  {name:15s}: {orig/1e6:>8.1f} MB  elems={elems:>9,}  -> {comp/1e6:.4f} MB (scalar, const={tensors[0][1][0].item():.6f}...)')
        else:
            comp = elems  # uint8
            overhead = len(tensors) * 8  # min + scale per tensor
            print(f'  {name:15s}: {orig/1e6:>8.1f} MB  elems={elems:>9,}  -> {comp/1e6:.1f} MB (8-bit) + {overhead/1e3:.1f} KB meta')
            comp += overhead
        
        total_compressed += comp
    
    print(f'  {"-"*45}')
    print(f'  {"TOTAL":15s}: {total_original/1e9:.2f} GB  -> {total_compressed/1e9:.3f} GB ({total_compressed/1e6:.0f} MB)')
    print(f'  Ratio: {total_original / max(total_compressed, 1):.1f}x')
    
    return groups


class FCF_CPR:
    """FCF checkpoint compressor — removes deterministic buffers, 
    quantizes real weights with uniform 8-bit per tensor."""
    
    def compress_sd(self, sd):
        """Compress model state_dict. Returns (compressed_dict, meta_dict)."""
        result = {}
        meta = {}
        
        for k, v in sd.items():
            if is_removable(k):
                continue  # skip entirely
            
            if is_scalar_gate(k, v):
                result[k] = v[0:1].clone()
                meta[k] = ('scalar', v.shape, v.dtype)
                continue
            
            if v.numel() > 1 and v.dtype.is_floating_point and v.std().item() < 1e-8:
                val = v[0:1].clone()
                result[k] = val
                meta[k] = ('scalar', v.shape, v.dtype)
                continue
            
            if not v.dtype.is_floating_point:
                result[k] = v.clone()
                meta[k] = ('scalar', v.shape, v.dtype)
                continue
            
            # Per-channel quantization for 2D weight tensors (row-wise)
            if v.ndim == 2 and v.shape[0] >= 16 and v.shape[1] >= 8:
                indices, mins, scales = quantize_tensor_channel(v, dim=0, n_bits=8)
                result[k] = indices
                meta[k] = ('uniform8_channel', v.shape, v.dtype, mins, scales, 0)
            elif v.ndim == 3 and v.shape[0] >= 4:
                # Grouped tensors: per-group along dim=0
                indices, mins, scales = quantize_tensor_channel(v, dim=0, n_bits=8)
                result[k] = indices
                meta[k] = ('uniform8_channel', v.shape, v.dtype, mins, scales, 0)
            else:
                # Per-tensor quantization for small/single-dim tensors
                indices, t_min, scale = quantize_tensor(v, n_bits=8)
                if indices is None:
                    val = torch.tensor([t_min], dtype=torch.float32)
                    if v.ndim == 0:
                        val = val.squeeze(0)
                    result[k] = val
                    meta[k] = ('scalar', v.shape, v.dtype)
                else:
                    result[k] = indices
                    meta[k] = ('uniform8', v.shape, v.dtype, t_min, scale)
        
        return result, meta
    
    def decompress_sd(self, compressed, meta, cfg):
        """Restore full state dict from compressed format."""
        from core import dct_basis, sparse_block_codes
        
        sd = {}
        
        # Recompute deterministic buffers
        V_dct = dct_basis(cfg.D)
        codes = sparse_block_codes(cfg.vocab, K=cfg.code_dim, S=cfg.code_sparsity)
        n_layers = cfg.n_layers
        
        for k, v in compressed.items():
            info = meta.get(k)
            if info is None:
                sd[k] = v.clone()
            elif info[0] == 'scalar':
                orig_shape, orig_dtype = info[1], info[2]
                if len(orig_shape) > 0 and orig_shape[0] > 1:
                    sd[k] = v.detach().expand(orig_shape).clone().to(orig_dtype)
                else:
                    sd[k] = v.detach().clone().to(orig_dtype)
            elif info[0] == 'uniform8':
                shape, dtype, t_min, scale = info[1], info[2], info[3], info[4]
                sd[k] = dequantize_tensor(v, t_min, scale, dtype)
            elif info[0] == 'uniform8_channel':
                shape, dtype, mins, scales, dim = info[1], info[2], info[3], info[4], info[5]
                sd[k] = dequantize_tensor_channel(v, mins, scales, shape, dtype)
        
        # Re-add deterministic buffers
        for i in range(n_layers):
            sd[f'layers.{i}.V_dct'] = V_dct.clone()
        sd['embed.codes'] = codes.clone()
        sd['lm_head.codes'] = codes.clone()
        
        return sd
    
    def save_compressed(self, ckpt, save_path):
        """Save compressed checkpoint."""
        sd = ckpt['model']
        compressed, meta = self.compress_sd(sd)
        
        out = {
            'step': ckpt.get('step', 0),
            'cfg': ckpt.get('cfg'),
            'model_compressed': compressed,
            'meta': meta,
        }
        for key in ('optimizer', 'scheduler', 'best_val_loss', 'interrupted'):
            if key in ckpt:
                out[key] = ckpt[key]
        
        torch.save(out, save_path)
        size = os.path.getsize(save_path)
        print(f'\nSaved: {save_path}')
        print(f'Compressed size: {size/1e9:.2f} GB ({size/1e6:.0f} MB)')
        return size
    
    def load_compressed(self, load_path, cfg=None):
        """Load and decompress checkpoint."""
        ckpt = torch.load(load_path, map_location='cpu', weights_only=False)
        
        if 'model_compressed' not in ckpt:
            print('Not a compressed checkpoint, returning as-is')
            return ckpt
        
        print(f'Loading compressed checkpoint (step {ckpt.get("step")})')
        compressed = ckpt['model_compressed']
        meta = ckpt['meta']
        
        cfg = cfg or ckpt.get('cfg')
        if cfg is None:
            raise ValueError('Need cfg to recompute deterministic buffers')
        
        ckpt['model'] = self.decompress_sd(compressed, meta, cfg)
        del ckpt['model_compressed']
        del ckpt['meta']
        
        return ckpt


# ─── Test ───

if __name__ == '__main__':
    ckpt_path = None
    candidates = [
        r'C:\Users\black\OneDrive\Desktop\step_10000.pt',
        r'C:\Users\black\OneDrive\Desktop\best.pt',
        os.path.join('..', '..', 'best.pt'),
        os.path.join('checkpoints', 'step_10000.pt'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            ckpt_path = p
            break
    
    if not ckpt_path:
        print(f'No checkpoint found. Checked: {candidates}')
        sys.exit(1)
    
    print(f'Loading: {ckpt_path}')
    print(f'Size: {os.path.getsize(ckpt_path)/1e9:.2f} GB')
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model']
    cfg = ckpt.get('cfg')
    
    print('\n--- Analysis ---')
    analyze_sd(sd)
    
    cpr = FCF_CPR()
    
    print('\n--- Compress ---')
    compressed_path = ckpt_path.replace('.pt', '_fcf.pt')
    cpr.save_compressed(ckpt, compressed_path)
    
    print('\n--- Roundtrip test ---')
    restored = cpr.load_compressed(compressed_path, cfg=cfg)
    
    sd_orig = sd
    sd_rest = restored['model']
    
    errors = []
    for k in sd_orig:
        if k in sd_rest:
            v_orig = sd_orig[k]
            v_rest = sd_rest[k]
            if v_orig.shape == v_rest.shape:
                mse = ((v_orig.float() - v_rest.float()) ** 2).mean().item()
                max_err = (v_orig.float() - v_rest.float()).abs().max().item()
                rel_err = max_err / max(v_orig.std().item(), 1e-10)
                errors.append((k, mse, max_err, rel_err))
            else:
                print(f'  Shape mismatch: {k} {list(v_orig.shape)} vs {list(v_rest.shape)}')
    
    print(f'\nQuantization accuracy ({len(errors)} tensors):')
    mses = [e[1] for e in errors]
    print(f'  Mean MSE: {sum(mses)/len(mses):.12f}')
    print(f'  Max MSE: {max(mses):.12f}')
    print(f'  Min MSE: {min(mses):.12f}')
    
    # Check worst
    worst = sorted(errors, key=lambda x: -x[1])[:5]
    if worst:
        print(f'\n  Worst 5:')
        for k, mse, mx, rel in worst:
            print(f'    MSE={mse:.12f} max_err={mx:.6f} rel={rel:.2f}  {k}')
    
    # Clean up
    os.remove(compressed_path)
    print(f'\nTest compressed file removed.')
