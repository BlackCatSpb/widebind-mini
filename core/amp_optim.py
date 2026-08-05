"""AmpAdam — Adam-аналог для факторизованной кодечной головы (SignedAmpCodec).

Зачем свой оптимизатор: у softmax-модели цель нормирована, и обычный Adam
работает; у факторизованной (ненормированной) кодечной цели есть вырожденные
направления градиента, которые AdamW не видит:

  1. свободный сдвиг log-счёта (token_bias): градиент d(loss)/db_y = −1 —
     смещение растёт к +∞, если не ограничить. Решение: проекция градиента
     на центрированное подпространство (вычесть среднее) + ограничение |b| ≤ B.

  2. инфляция/дефляция дисперсии (log_sigma_*): σ — коробка [σ_min, 1],
     вне неё пик правдоподобия > 1 (читерский отрицательный CE). Решение:
     проекция параметра на интервал после каждого шага.

  3. амплификация чтения (log_gain): коробка [0.1, 4], за границей — насыщение
     tanh без обучения.

  4. разная «занятость» позиций кода: позиция k активна у p_k·V токенов.
     Редкие позиции получают тот же абсолютный шаг, что и частые, и шум
     их оценки выше. Решение: позиционная частотная нормализация —
     градиент параметров позиции k умножается на 1/sqrt(max(n_k/ṋ, 1)),
     т.е. шаг как у Adam, но с SNR-поправкой на частоту использования.

  5. разные масштабы групп head/embed/backbone: раздельные lr по ролям
     (по умолчанию backbone — базовая, embed — 0.5×, head — 1.0×).

По механике — это классический Adam (moments m, v + bias correction), т.е.
шум минибатча демпфируется так же, как у softmax-моделей, но поверх него —
структурные проекции кодечной головы.
"""

import math
import torch

# Границы коробок (должны совпадать с ограничениями в amp_codec.py):
#   σ = softplus(log_sigma) + 0.05  ∈ [σ_min, 1],  σ_min — из cfg
#   (маржинальная цель ограничена при любом σ_min > 0; 1/√(2π) — граница
#   допустимости likelihood для обычного CE).
LOG_SIGMA_MIN = 0.2
LOG_SIGMA_BOUNDS = (math.log(LOG_SIGMA_MIN - 0.05), math.log(0.95))
#   gain = exp(log_gain) ∈ [0.1, 4]
LOG_GAIN_BOUNDS = (math.log(0.1), math.log(4.0))
BIAS_BOUND = 4.0


def build_amp_groups(model, lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                     head_scale=1.0, embed_scale=0.5, backbone_scale=1.0):
    """Собирает param_groups для AmpAdam по ролям параметров модели.

    Возвращает плоский список групп {'params', 'lr', 'weight_decay', 'role',
    'pos_scale'|None, 'project_mean'|False, 'bounds'|None}.
    """
    cfg = model.cfg
    s_min = getattr(cfg, 'amp_sigma_min', LOG_SIGMA_MIN)
    sigma_bounds = (math.log(max(s_min - 0.05, 1e-4)), math.log(0.95))
    head = getattr(model, 'lm_head', None)
    codes = head.codes if head is not None else None
    K = codes.shape[1]
    n_k = codes.sum(0).clamp_min(1.0)          # (K,) активность позиций
    n_mean = n_k.mean()
    pos_scale = 1.0 / (n_k / n_mean).clamp_min(1.0).sqrt()  # (K,)

    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        meta = {'project_mean': False, 'bounds': None, 'pos_scale': None}
        if name.startswith('lm_head.'):
            role, rlr = 'head', lr * head_scale
            if name.endswith('token_bias'):
                meta['project_mean'] = True
                meta['bounds'] = (-BIAS_BOUND, BIAS_BOUND)
            elif name.endswith('log_sigma_on') or name.endswith('log_sigma_off') or name.endswith('log_sigma_emb'):
                meta['bounds'] = sigma_bounds
                meta['pos_scale'] = pos_scale
            elif name.endswith('log_gain') or name.endswith('log_gain_emb'):
                meta['bounds'] = LOG_GAIN_BOUNDS
                meta['pos_scale'] = pos_scale
            elif name.endswith('proto'):
                meta['pos_scale'] = pos_scale.unsqueeze(0).expand(p.shape)
            elif name.endswith('basis'):
                meta['pos_scale'] = pos_scale.unsqueeze(1).expand(p.shape)
                meta['renormalize'] = True
        elif name.startswith('embed.'):
            role, rlr = 'embed', lr * embed_scale
        else:
            role, rlr = 'backbone', lr * backbone_scale
        out.append({'params': [p], 'lr': rlr, 'weight_decay': weight_decay,
                    'role': role, **meta})
    return out


class AmpAdam(torch.optim.Optimizer):
    """Adam-аналог с проекциями, специфичными для кодечной головы.

    Использование:
        groups = build_amp_groups(model, lr=..., ...)
        opt = AmpAdam(groups)
    """

    def __init__(self, groups, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(betas=betas, eps=eps)
        for g in groups:
            g.setdefault('betas', betas)
            g.setdefault('eps', eps)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        beta1, beta2 = self.defaults['betas']
        eps = self.defaults['eps']
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            role = group['role']
            bounds = group['bounds']
            project_mean = group['project_mean']
            scale = group['pos_scale']
            renormalize = group.get('renormalize', False)
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if project_mean:
                    g = g - g.mean()
                if scale is not None:
                    g = g * scale.to(g.dtype).to(g.device)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                if wd != 0.0 and p.ndim >= 2:
                    g = g + wd * p
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                step_size = lr / bias_c1
                p.addcdiv_(exp_avg, denom, value=-step_size)
                if bounds is not None:
                    p.clamp_(min=bounds[0], max=bounds[1])
                if renormalize:
                    import torch.nn.functional as F
                    p.data = F.normalize(p.data, dim=-1)
        return loss
