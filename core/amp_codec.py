"""SignedAmpCodec — симметричный кодек «подписанной амплитуды» (без softmax).

Хранение информации — в многовекторном пространстве «вдоль оси» сегмента:
каждый из K сегментов D-пространства несёт скалярную подписанную амплитуду
a_k ∈ (−1,1): знак = сторона отклонения от оси, |a_k| = сила отклонения.

Токен v кодируется K-позициями кода (S активных из K, sparse_block_codes):
  * активные позиции хранят обучаемый прототип амплитуды α_vk (знак ±);
  * неактивные — «на оси» (ожидание около 0).

Оба конца делят один ортонормированный базис (weight tying encode/decode):
  * запись (SignedAmpEmbedding): h = scale·Σ_k α_vk·basis_k + rope;
  * чтение (SignedAmpHead) — ДВУХКАНАЛЬНОЕ, каналы аддитивны на уровне счёта:
      — контекстный канал:  a_k = tanh(gain_k·⟨h_L, basis_k⟩), σon_k/σoff_k;
      — код-канал:          p_k = gain_emb_k·⟨h_emb, basis_k⟩, σ_emb_k —
        прямое считывание собственной записи (стек превращает проекцию
        финального состояния в шум ~×30 сильнее кода, поэтому смешивать
        каналы до tanh нельзя).
    log P(v) = Σ_k [z_vk·logN(a_k; α_vk, σon_k) + (1−z_vk)·logN(a_k; o_k, σoff_k)]
             + Σ_k z_vk·logN(p_k; α_vk, σ_emb_k) + b_v.

Позиции независимы => вероятность нормирована по построению, обучение идёт
через margin_loss за O(K) на токен: −(log P(target) − E_q log P) — масштабно
инвариантная маржинальная цель (q = равномерное по словарю), где общие
сдвиги счёта гасятся, как в log-softmax, но БЕЗ exp-sum по V и БЕЗ
материализации словаря. forward() материализует полный набор лог-вероятностей
для eval (линейно по sparse-коду, без нормализации).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vsa_utils import sparse_block_codes
from .embedding import RotaryEmbedding

_LOG2PI = math.log(2.0 * math.pi)


def _amp_codes_proto(cfg, g):
    """Коды и прототипы для кодекa.

    Спаrsный режим: sparse_block_codes + случайные proto (значения ±, активность
    по хеш-позициям). Фазовый режим (amp_phasor, «корни из единицы»): плотные
    коды, proto = амплитуда cos/sin-пары: α_v,2j = A·cos(2π(j+1)v/V),
    α_v,2j+1 = A·sin(2π(j+1)v/V). Переход «следующий токен» становится
    ВРАЩЕНИЕМ пары — линейным в кодовых координатах; оператор W_pred (механика
    A) может представить его точно. proto обучаем поверх формулы.
    """
    K = cfg.code_dim
    if getattr(cfg, 'amp_phasor', False):
        assert K % 2 == 0, 'amp_phasor требует чётное code_dim'
        codes = torch.ones(cfg.vocab, K, dtype=torch.float)
        pairs = K // 2
        jj = (torch.arange(pairs, dtype=torch.float) + 1.0)
        vv = torch.arange(cfg.vocab, dtype=torch.float)
        phase = 2.0 * math.pi * torch.outer(vv, jj) / cfg.vocab
        A = math.atanh(min(getattr(cfg, 'amp_phase_amp', 0.8), 0.999))
        prot = torch.empty(cfg.vocab, K)
        prot[:, 0::2] = A * torch.cos(phase)
        prot[:, 1::2] = A * torch.sin(phase)
        return codes, prot
    codes = sparse_block_codes(cfg.vocab, K=K, S=cfg.code_sparsity)
    if getattr(cfg, 'amp_hybrid', False):
        assert K % 2 == 0, 'amp_hybrid требует чётное code_dim'
        Ks = getattr(cfg, 'amp_hybrid_s', 6)   # sparse-позиции (разделение)
        Kp = K - Ks                            # фазовые позиции (вращение)
        assert Kp % 2 == 0 and Kp > 0, 'amp_hybrid: K-amp_hybrid_s должно быть чётным > 0'
        gs = torch.Generator().manual_seed(getattr(cfg, 'amp_seed', 0) + 7)
        codes = torch.zeros(cfg.vocab, K, dtype=torch.float)
        for v in range(cfg.vocab):
            idx = torch.randperm(Ks, generator=gs)[:cfg.code_sparsity]
            codes[v, idx] = 1.0
        pairs = Kp // 2
        jj = (torch.arange(pairs, dtype=torch.float) + 1.0)
        vv = torch.arange(cfg.vocab, dtype=torch.float)
        phase = 2.0 * math.pi * torch.outer(vv, jj) / cfg.vocab
        A = math.atanh(min(getattr(cfg, 'amp_phase_amp', 0.8), 0.999))
        init = getattr(cfg, 'amp_proto_init', 0.2)
        prot = torch.empty(cfg.vocab, K)
        prot[:, :Ks] = (torch.rand(cfg.vocab, Ks, generator=g) - 0.5) * 2 * init
        prot[:, Ks::2] = A * torch.cos(phase)
        prot[:, Ks + 1::2] = A * torch.sin(phase)
        return codes, prot
    init = getattr(cfg, 'amp_proto_init', 0.2)
    prot = (torch.rand(cfg.vocab, K, generator=g) - 0.5) * 2 * init
    return codes, prot


class SignedAmpEmbedding(nn.Module):
    """Запись: токен -> D-вектор как сумма подписанных амплитуд по K осям.

    h = scale · Σ_k α_vk · basis_k + rope, α_vk = proto[v,k]·codes[v,k].
    Код маскирует амплитуду: на неактивных позициях вклад нулевой.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(getattr(cfg, 'amp_seed', 0))
        codes, prot = _amp_codes_proto(cfg, g)
        self.K = codes.shape[1]
        self.S = cfg.code_sparsity
        self.register_buffer('codes', codes)
        D = cfg.D
        assert D % self.K == 0, f'D={D} must be divisible by K={self.K}'
        d = D // self.K
        b = torch.randn(self.K, d, generator=g) * 0.5 / math.sqrt(d)
        # Единичные нормы строк: ⟨basis_k, basis_k⟩ = 1, поэтому прямое считывание
        # записанного кода равно α_k (без аттенюации на ‖basis‖²) при единичном
        # масштабе записи.
        b = b / b.norm(dim=-1, keepdim=True)
        self.basis = nn.Parameter(b)
        # proto хранит raw-логит; амплитуда α = tanh(raw) ∈ (−1,1) — ограничена,
        # поэтому квадратичный штраф (a−α)² ограничен: loss не может убежать.
        self.proto = nn.Parameter(prot)
        self._scale = getattr(cfg, 'amp_scale', 1.0)
        self.rope = RotaryEmbedding(D,
                                    theta=getattr(cfg, 'rope_theta', 1000000.0),
                                    scaling=getattr(cfg, 'rope_scaling', 1.0))

    def forward(self, tokens):
        B, L = tokens.shape
        alpha = torch.tanh(self.proto[tokens]) * self.codes[tokens]  # (B, L, K)
        out = torch.einsum('blk,kd->blkd', alpha, self.basis).reshape(B, L, -1) * self._scale
        return self.rope(out)


class SignedAmpHead(nn.Module):
    """Чтение: D-вектор -> подписанные амплитуды -> факторизованные лог-вероятности.

    Если переданы embed_basis/embed_proto (объекты nn.Parameter эмбеддинга),
    head делит их с эмбеддингом, не регистрируя у себя (избегаем двойного
    попадания в параметры оптимизатора): ссылки держим в plain-атрибутах.
    """

    def __init__(self, cfg, embed_basis=None, embed_proto=None):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(getattr(cfg, 'amp_seed', 0))
        codes, prot = _amp_codes_proto(cfg, g)
        self.K = codes.shape[1]
        self.register_buffer('codes', codes)
        if embed_basis is not None:
            self._basis_ref = [embed_basis]
        else:
            D = cfg.D
            d = D // self.K
            g = torch.Generator().manual_seed(getattr(cfg, 'amp_seed', 0) + 1)
            b = torch.randn(self.K, d, generator=g) * 0.5 / math.sqrt(d)
            b = b / b.norm(dim=-1, keepdim=True)
            self._basis_ref = [nn.Parameter(b)]
        if embed_proto is not None:
            self._proto_ref = [embed_proto]
        else:
            g = torch.Generator().manual_seed(getattr(cfg, 'amp_seed', 0) + 2)
            self._proto_ref = [nn.Parameter(prot)]
        # gain стартует в 0.5: проекции стека большие (std ~1), раннее
        # насыщение tanh убило бы градиент; модель сама заостряет gain.
        self.log_gain = nn.Parameter(torch.full((self.K,), math.log(getattr(cfg, 'amp_gain_init', 0.5))))
        # Двухканальное чтение:
        #   контекстный канал — a = tanh(gain·⟨h_L, basis⟩) со своими σon/σoff;
        #   код-канал — p_emb = gain_emb·⟨h_emb, basis⟩ против α_v, свой σ_emb.
        # Смешивать их до tanh нельзя: проекция стека (std ~2.9) в ~30 раз больше
        # кода (std ~0.1), общий tanh утопил бы код. Поэтому каналы складываются
        # НА УРОВНЕ СЧЁТА (факторизованные лог-вероятности — аддитивны).
        self.log_gain_emb = nn.Parameter(torch.full((self.K,), math.log(getattr(cfg, 'amp_gain_init', 0.5))))
        # Механизм A — оператор перехода в кодовом пространстве: контекстное
        # чтение проходит через W_pred (K,K) перед gain. Для copy W_pred → I
        # (init = I + ε); для переходов со структурой в кодах (напр. фазовые
        # сдвиги) W_pred учит линейный оператор «код(t) → код(t+1)». Включается
        # cfg.amp_pred; иначе применяется как есть (I).
        self.pred_w = nn.Parameter(torch.eye(self.K) + 0.01 * torch.randn(self.K, self.K))
        self.o = nn.Parameter(torch.zeros(self.K))
        s0 = math.log(getattr(cfg, 'amp_sigma_init', 0.3))
        self.log_sigma_on = nn.Parameter(torch.full((self.K,), s0))
        self.log_sigma_off = nn.Parameter(torch.full((self.K,), s0))
        # σ_emb = «доверие к коду»: маленькое = острое сопоставление своего кода
        # (полезно для copy), большое = канал заглушен (next-token: эхо вредит).
        self.log_sigma_emb = nn.Parameter(torch.full((self.K,), s0))
        self.token_bias = nn.Parameter(torch.zeros(cfg.vocab))

    @property
    def basis(self):
        return self._basis_ref[0]

    @property
    def proto(self):
        return self._proto_ref[0]

    def _proj(self, h_g):
        return torch.einsum('...kd,kd->...k', h_g, self.basis)

    def _amps(self, h_g):
        # Контекстный канал: gain ограничен сверху — насыщение tanh гасит
        # градиент, а не убегает. Если включён механизм A (amp_pred), чтение
        # проходит через оператор перехода W_pred в кодовом пространстве.
        z = self._proj(h_g)
        if getattr(self.cfg, 'amp_pred', False):
            z = torch.einsum('...k,kj->...j', z, self.pred_w)
        T = torch.exp(self.log_gain).clamp(0.1, 4.0)
        z = z * T
        return torch.tanh(z), z

    def _code(self, h_emb_g):
        # Код-канал: линейная проекция записанного кода со своим gain.
        T = torch.exp(self.log_gain_emb).clamp(0.1, 4.0)
        return self._proj(h_emb_g) * T

    def _sigmas(self):
        # σ ограничена [σ_min, 1.0]. Для обычного likelihood нужна допустимость
        # правдоподобия: σ ≥ 1/√(2π) ≈ 0.399, иначе пик N > 1 и logP цели может
        # стать положительным (читерский CE<0). Для маржинальной цели
        # масштабная инвариантность сама гасит инфляцию, нужен только σ_min > 0
        # (иначе margin → −∞ при σ→0). Меньший σ_min = острее ранжирование.
        s_min = getattr(self.cfg, 'amp_sigma_min', 0.2)
        return (torch.clamp(F.softplus(self.log_sigma_on) + 0.05, min=s_min, max=1.0),
                torch.clamp(F.softplus(self.log_sigma_off) + 0.05, min=s_min, max=1.0),
                torch.clamp(F.softplus(self.log_sigma_emb) + 0.05, min=s_min, max=1.0))

    def _bias(self):
        # центрирование по всем токенам (убирает свободный сдвиг log-счёта)
        # и ограничение |b| ≤ 4: иначе маржинальный loss растит b_v → +∞.
        return torch.clamp(self.token_bias - self.token_bias.mean(), min=-4.0, max=4.0)

    def _alpha_of(self, tokens):
        return torch.tanh(self.proto[tokens]) * self.codes[tokens]

    def _active_stats(self):
        """Среднее и дисперсия α по активным позициям кода + доля активности.

        μ_k = E_{v: z_vk=1}[α_vk],  ν_k = Var_{v: z_vk=1}[α_vk],  p_k = E[z_vk].
        Точные значения по словарю за O(V·K) — нужны для маржинальной цели
        (ожидание лог-счёта по равномерной q на токенах).
        """
        alpha = torch.tanh(self.proto) * self.codes  # (V, K)
        n = self.codes.sum(0).clamp_min(1)
        mu = (alpha * self.codes).sum(0) / n
        nu = ((alpha * self.codes).square().sum(0) / n - mu.square()).clamp_min(0.0)
        return mu, nu, self.codes.mean(0)

    def margin_loss(self, h, targets, h_emb=None):
        """(N,) — маржинальная цель: −(log P(target) − E_q log P), q = uniform.

        Двухканальный счёт: контекстный канал (a из преобразованного состояния,
        σon/σoff) + код-канал (p_emb из собственной записи против α_v, σ_emb).
        Каналы аддитивны на уровне счёта, поэтому масштабная инвариантность
        log-softmax сохраняется: общие сдвиги гасятся, остаётся разность
        позиционных членов. Ожидание по q точное за O(K) через μ_k, ν_k.
        """
        N, D = h.shape
        a, _ = self._amps(h.reshape(N, self.K, -1))
        pe = None if h_emb is None else self._code(h_emb.reshape(N, self.K, -1))
        son, soff, semb = self._sigmas()
        z = self.codes[targets].float()
        alpha = torch.tanh(self.proto[targets]) * z
        on = -0.5 * ((a - alpha) / son) ** 2 - torch.log(son) - 0.5 * _LOG2PI
        off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
        logp = (z * on + (1 - z) * off).sum(-1) + self._bias()[targets]
        mu, nu, p = self._active_stats()
        e_on = -0.5 * ((a - mu) ** 2 + nu) / son ** 2 - torch.log(son) - 0.5 * _LOG2PI
        e_off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
        e_q = (p * e_on + (1.0 - p) * e_off).sum(-1)
        if pe is not None:
            code = -0.5 * ((pe - alpha) / semb) ** 2 - torch.log(semb) - 0.5 * _LOG2PI
            logp = logp + (z * code).sum(-1)
            e_code = -0.5 * ((pe - mu) ** 2 + nu) / semb ** 2 - torch.log(semb) - 0.5 * _LOG2PI
            e_q = e_q + (p * e_code).sum(-1)
        return -(logp - e_q)

    def log_probs_for_target(self, h, targets, h_emb=None):
        """h: (N, D), targets: (N,) -> (N,) факторизованный лог P(target) за O(K)."""
        N, D = h.shape
        a, _ = self._amps(h.reshape(N, self.K, -1))
        pe = None if h_emb is None else self._code(h_emb.reshape(N, self.K, -1))
        son, soff, semb = self._sigmas()
        alpha = self._alpha_of(targets)
        z = self.codes[targets].float()
        on = -0.5 * ((a - alpha) / son) ** 2 - torch.log(son) - 0.5 * _LOG2PI
        off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
        logp = (z * on + (1 - z) * off).sum(-1) + self._bias()[targets]
        if pe is not None:
            code = -0.5 * ((pe - alpha) / semb) ** 2 - torch.log(semb) - 0.5 * _LOG2PI
            logp = logp + (z * code).sum(-1)
        return logp

    def code_reg(self, h, targets, h_emb=None):
        """(N,) — механизм C: регрессия контекстного чтения на код ЦЕЛИ.

        ‖a − α_y‖² по активным позициям цели. В отличие от margin, здесь нет
        компенсирующего члена E_q (который тянет a к среднему μ и ослабляет
        градиент стека): чистое плотное притяжение чтения к коду следующего
        токена. Учивает «переход»: для copy a→α_y=α_x; для counter/shift
        a→α_{x+1} (стек должен преобразовать состояние).
        """
        N, D = h.shape
        a, _ = self._amps(h.reshape(N, self.K, -1))
        z = self.codes[targets].float()
        alpha = torch.tanh(self.proto[targets]) * z
        num = ((a - alpha) ** 2 * z).sum(-1)
        den = z.sum(-1).clamp_min(1.0)
        return num / den

    def ce_loss(self, h, targets, h_emb=None):
        """(N,) — CE по факторизованному счёту: logΣ_v exp(s_v) − s_y за O(V·K).

        Математика: CE = hinge-soft ≥ argmax_hinge (доминирует борьбу с
        argmax-конкурентом) и CE = margin + [logΣexp(s) − E_q s] ≥ margin
        (добавляет требование остроты — член ≥ 0 по неравенству log-sum-exp).
        Одна цель вместо пары margin+hinge: нет hinge_weight, нет E_q-машинерии,
        один forward вместо forward+E_q.
        """
        N, D = h.shape
        logits = self.forward(h.reshape(1, N, D),
                              None if h_emb is None else h_emb.reshape(1, N, D))
        idx = torch.arange(N, device=logits.device)
        s_y = logits[0, idx, targets]
        return torch.logsumexp(logits[0], dim=-1) - s_y

    def argmax_hinge(self, h, targets, h_emb=None):
        """(N,) — шарнир против ИСТИННОГО argmax-конкурента (не среднего).

        Маржинальная цель против E_q (среднего токена) не ограничивает argmax:
        шумный контекстный канал даёт высокую дисперсию счёта, и случайные
        токены перебивают цель. Здесь: w* = argmax_{w≠y} score(w), полные
        счета за O(V·K) через forward() (без softmax и exp-sum), и
        hinge = max(0, s(w*) − s(y)) — ровно ошибка top1. Ограничен снизу 0;
        при σ_min > 0 и ограниченных a/α/bias ограничен сверху.
        """
        N, D = h.shape
        logits = self.forward(h.reshape(1, N, D),
                              None if h_emb is None else h_emb.reshape(1, N, D))
        idx = torch.arange(N, device=logits.device)
        s_y = logits[0, idx, targets]
        lm = logits[0].clone()
        lm[idx, targets] = -float('inf')
        s_w = lm.max(-1).values
        return (s_w - s_y).clamp_min(0.0)

    def forward(self, h, h_emb=None):
        """h: (B, L, D) -> (B, L, V) лог-вероятности всех токенов (eval).

        Логарифмический профиль каждой позиции разложен на v-независимую часть
        и линейную по α_vk (оба канала), поэтому материализация идёт парой
        einsum по sparse-коду, без нормализации (факторизация уже нормирует).
        """
        B, L, D = h.shape
        a, _ = self._amps(h.reshape(B, L, self.K, -1))
        son, soff, semb = self._sigmas()
        son_sq, soff_sq = son ** 2, soff ** 2
        off = -0.5 * ((a - self.o) / soff) ** 2 - torch.log(soff) - 0.5 * _LOG2PI
        A = -0.5 * (a / son) ** 2 + 0.5 * ((a - self.o) / soff) ** 2 \
            + torch.log(soff) - torch.log(son)
        alpha = torch.tanh(self.proto) * self.codes  # (V, K), амплитуды ∈ (−1,1)
        logits = off.sum(-1).unsqueeze(-1) + self._bias().unsqueeze(0).unsqueeze(0)
        logits = logits + torch.einsum('vk,blk->blv', alpha, a / son_sq)
        logits = logits - 0.5 * torch.einsum('vk,k->v', alpha.square(), 1.0 / son_sq).unsqueeze(0).unsqueeze(0)
        logits = logits + torch.einsum('vk,blk->blv', self.codes, A)
        if h_emb is not None:
            pe = self._code(h_emb.reshape(B, L, self.K, -1))
            semb_sq = semb ** 2
            A_code = -0.5 * pe.square() / semb_sq - torch.log(semb) - 0.5 * _LOG2PI
            logits = logits + torch.einsum('vk,blk->blv', self.codes, A_code)
            logits = logits + torch.einsum('vk,blk->blv', alpha / semb_sq, pe)
            logits = logits - 0.5 * torch.einsum('vk,k->v', alpha.square(), 1.0 / semb_sq).unsqueeze(0).unsqueeze(0)
        return logits
