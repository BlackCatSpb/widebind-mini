"""TrajectoryCrossBind: gradient transfer BETWEEN layers via cross-mixing.

Key idea: instead of crossing u⊙v within one layer, cross between
different layers' trajectories (hp_t ⊙ hp_{t-k}), or between hp and VSA state.

This creates direct gradient paths between layers, not just through residual.
"""
import sys, os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TrajectoryCrossBind(nn.Module):
    """Cross-mixing between current hp and trajectory memory.

    Maintains a trajectory buffer of past hp states.
    At each step: hp_t is crossed with hp_{t-1}, hp_{t-2}, ... (multi-scale)
    Gradient flows from layer t through bind back to layer t-1, t-2, ...
    """

    def __init__(self, D, K, S=4, traj_len=4):
        super().__init__()
        self.D, self.K, self.S, self.traj_len = D, K, S, traj_len
        self.W_proj = nn.Linear(D, K, bias=True)
        self.hp_norm = nn.RMSNorm(K)

        # Cross-weights per (spiral, trajectory_offset)
        self.w_cross_u = nn.Parameter(torch.randn(S, traj_len, K) * 0.3)
        self.w_cross_v = nn.Parameter(torch.randn(S, traj_len, K) * 0.3)
        nn.init.normal_(self.w_cross_u, 0, 1.0)
        nn.init.normal_(self.w_cross_v, 0, 1.0)

        # Per-spiral output projections
        self.W_out = nn.Parameter(torch.empty(S, K, D))
        nn.init.xavier_uniform_(self.W_out, gain=0.5)

        # Trajectory buffer (registered as buffer for proper device handling)
        self.register_buffer('_traj_buf', torch.zeros(traj_len, 1, 1, K), persistent=False)

    def forward(self, h, traj_state=None):
        """h: (B, L, D), traj_state: (traj_len, B, L, K) or None."""
        hp = self.hp_norm(self.W_proj(h))
        B, L, K = hp.shape

        # Build trajectory: current hp + past states
        if traj_state is None:
            traj = [hp] + [torch.zeros_like(hp) for _ in range(self.traj_len - 1)]
        else:
            # Shift: drop oldest, add current
            traj = [hp] + [traj_state[i] for i in range(self.traj_len - 1)]

        new_traj = torch.stack(traj, dim=0)

        out = None
        for s in range(self.S):
            acc = None
            for t in range(self.traj_len):
                u = hp * self.w_cross_u[s, t]
                v = traj[t] * self.w_cross_v[s, t]
                # Cross-mixing between current hp and trajectory[t]
                cross = u * torch.roll(v, shifts=t + 1, dims=-1)
                acc = cross if acc is None else acc + cross
            term = acc @ self.W_out[s]
            out = term if out is None else out + term

        return out, new_traj

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def test_gradient_flow():
    """Test that gradients flow from layer N back to layer 0 through bind."""
    D, K, S = 128, 64, 4
    N_LAYERS = 8
    torch.manual_seed(42)

    print("=== Gradient Flow Test ===")
    print(f"D={D}, K={K}, S={S}, layers={N_LAYERS}")

    # Stack of N bind layers, each shares the trajectory buffer
    binds = nn.ModuleList([TrajectoryCrossBind(D, K, S, traj_len=min(4, i + 1)) for i in range(N_LAYERS)])

    h_input = torch.randn(2, 16, D, requires_grad=True)
    h = h_input
    traj = None
    losses_per_layer = []

    for i in range(N_LAYERS):
        out, traj = binds[i](h, traj)
        h = h + out  # residual
        loss = out.norm()
        losses_per_layer.append(loss.item())

    # Backprop from final layer
    loss_final = h.sum()
    loss_final.backward()

    print(f"Input grad norm: {h_input.grad.norm().item():.6f}")
    print(f"Gradient flows through all layers: {h_input.grad.abs().sum().item() > 0}")

    # Check per-bind parameter gradients
    for i, bind in enumerate(binds):
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in bind.parameters())
        print(f"  Bind L{i}: has grad = {has_grad}")
    print()


def test_traj_vs_isolated():
    """Compare trajectory cross-bind vs isolated on long-range dependency task.

    Task: sequence of length 64 where token[t] = token[t-8] (long dependency).
    Trajectory bind has direct gradient path through 8-step history.
    Isolated bind must propagate through residual (weaker signal).
    """
    D, K, S = 128, 64, 4
    SEQ_LEN = 64
    N_TOKENS = 256
    STEPS = 150
    LR = 2e-3

    print("=== Trajectory vs Isolated (long-range dependency) ===")

    torch.manual_seed(42)
    seq = torch.randint(0, N_TOKENS, (SEQ_LEN,))
    for t in range(8, SEQ_LEN):
        if torch.rand(1) < 0.8:
            seq[t] = seq[t - 8]  # 8-step dependency

    codes = torch.zeros(N_TOKENS, K)
    for v in range(N_TOKENS):
        active = torch.randperm(K)[:6]
        codes[v, active] = 1.0

    class CodeBook(nn.Module):
        def __init__(self):
            super().__init__()
            self.basis = nn.Parameter(F.normalize(torch.randn(K, D // K), dim=-1))
            self.proto = nn.Parameter((torch.rand(N_TOKENS, K) - 0.5) * 0.4)
            self.register_buffer('codes', codes)
        def forward(self, tokens):
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            alpha = torch.tanh(self.proto[tokens]) * self.codes[tokens]
            return torch.einsum('blk,kd->blkd', alpha, self.basis).reshape(tokens.shape[0], tokens.shape[1], -1)

    results = {}
    for name, traj_len in [("trajectory(8)", 8), ("isolated(1)", 1)]:
        book = CodeBook()
        bind = TrajectoryCrossBind(D, K, S, traj_len=traj_len)
        opt = torch.optim.Adam(list(book.parameters()) + list(bind.parameters()), lr=LR)

        # Train on short sequences
        losses = []
        for step in range(STEPS):
            start = torch.randint(0, SEQ_LEN - 16, (1,)).item()
            chunk = seq[start:start + 16]
            h = book(chunk).squeeze(0)
            traj = None
            out, traj = bind(h.unsqueeze(0), traj)
            out = out.squeeze(0)
            target = book(chunk).squeeze(0).detach()
            loss = F.mse_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Eval on FULL sequence (longer than training chunks)
        with torch.no_grad():
            h_full = book(seq).squeeze(0)
            traj = None
            out_full, traj = bind(h_full.unsqueeze(0), traj)
            out_full = out_full.squeeze(0)
            target_full = book(seq).squeeze(0)
            eval_loss = F.mse_loss(out_full, target_full).item()

        results[name] = (losses[-1], eval_loss)
        print(f"{name:18s}: train_loss={losses[-1]:.4f}  eval_loss={eval_loss:.4f}")

    train_winner = min(results, key=lambda k: results[k][0])
    eval_winner = min(results, key=lambda k: results[k][1])
    print(f"Train winner: {train_winner}  Eval winner: {eval_winner}")
    print()


def test_multidim_cross():
    """Test multi-dimensional crossing: multiple trajectory dimensions.

    Instead of 1D trajectory (past hp), use multi-dim:
    - dim 0: previous layer hp
    - dim 1: VSA memory state
    - dim 2: Mirror correction signal
    Each dimension gets its own cross-weights.
    """
    D, K, S = 128, 64, 4
    N_DIMS = 3

    print("=== Multi-Dimensional Cross Test ===")

    bind = TrajectoryCrossBind(D, K, S, traj_len=N_DIMS)

    h = torch.randn(2, 16, D)
    traj = torch.randn(N_DIMS, 2, 16, K)

    out, new_traj = bind(h, traj)
    print(f"Input: {tuple(h.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"New trajectory: {tuple(new_traj.shape)}")
    print(f"Params: {bind.param_count() / 1e3:.1f}K")
    print()


if __name__ == '__main__':
    test_gradient_flow()
    test_traj_vs_isolated()
    test_multidim_cross()
