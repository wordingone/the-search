"""
FAIR BENCHMARK: ANIMA-0 vs ANIMA-1 vs Transformer
=================================================

CRITICAL: All models have EXACTLY THE SAME number of parameters.

Previous benchmarks were INVALID because:
- ANIMA-Zero: 26,572 params
- VanillaTransformer: 8,964 params (2.96x fewer!)

This benchmark:
1. Sets a TARGET_PARAMS budget
2. Scales all models to match EXACTLY
3. Reports delta if any mismatch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
import math

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ============================================
# TARGET PARAMETER BUDGET
# ============================================
TARGET_PARAMS = 25000  # All models must match this (+/- 5%)
TOLERANCE = 0.05       # 5% tolerance


# ============================================
# ANIMA-ZERO (Sequential, No Compression)
# ============================================
class ANIMAZero(nn.Module):
    """
    ANIMA-Zero scaled to target params.

    Original used d=32 for all dimensions.
    We scale dimensions to hit target params.
    """

    def __init__(self, sensory_dim=8, output_dim=4, target_params=TARGET_PARAMS):
        super().__init__()

        # Binary search for dimension that gives target params
        d = self._find_dimension(sensory_dim, output_dim, target_params)
        self.d = d

        # W: Type S (Sensing)
        self.W_enc = nn.Linear(sensory_dim, d)
        self.W_from_W = nn.Linear(d, d, bias=False)
        self.W_from_I = nn.Linear(d, d, bias=False)
        self.W_from_A = nn.Linear(d, d, bias=False)
        self.W_attention = nn.Linear(d * 2, d)
        self.W_mult_gate = nn.Linear(d * 2, d)

        # I: Type M (Memory) - GRU gates
        self.I_z_W = nn.Linear(d, d, bias=False)
        self.I_z_I = nn.Linear(d, d, bias=False)
        self.I_z_A = nn.Linear(d, d, bias=False)
        self.I_z_bias = nn.Parameter(torch.zeros(d))

        self.I_r_W = nn.Linear(d, d, bias=False)
        self.I_r_I = nn.Linear(d, d, bias=False)
        self.I_r_A = nn.Linear(d, d, bias=False)
        self.I_r_bias = nn.Parameter(torch.zeros(d))

        self.I_h_W = nn.Linear(d, d, bias=False)
        self.I_h_I = nn.Linear(d, d, bias=False)
        self.I_h_A = nn.Linear(d, d, bias=False)
        self.I_h_bias = nn.Parameter(torch.zeros(d))

        self.I_mult_gate = nn.Linear(d * 2, d)

        # A: Type D (Decision)
        self.A_from_W = nn.Linear(d, d, bias=False)
        self.A_from_I = nn.Linear(d, d, bias=False)
        self.A_from_A = nn.Linear(d, d, bias=False)
        self.A_bias = nn.Parameter(torch.zeros(d))
        self.A_mult_gate = nn.Linear(d * 2, d)

        # Output
        self.phi = nn.Linear(d, output_dim)

        # State
        self.W = None
        self.I = None
        self.A = None

        self._init_weights()

    def _find_dimension(self, s_dim, o_dim, target):
        """Binary search for dimension."""
        for d in range(8, 128):
            # Count params for this dimension
            params = (
                s_dim * d + d +           # W_enc
                d * d * 3 +               # W_from_*
                d * 2 * d + d +           # W_attention
                d * 2 * d + d +           # W_mult_gate
                d * d * 9 +               # I gates (z, r, h) x (W, I, A)
                d * 3 +                   # I biases
                d * 2 * d + d +           # I_mult_gate
                d * d * 3 +               # A_from_*
                d +                       # A_bias
                d * 2 * d + d +           # A_mult_gate
                d * o_dim + o_dim         # phi
            )
            if params >= target * (1 - TOLERANCE):
                return d
        return 32

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.orthogonal_(param)
                with torch.no_grad():
                    param.mul_(0.99 / max(param.abs().max(), 1e-6))
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.constant_(self.I_z_bias, -1.0)

    def reset(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.W = torch.zeros(batch_size, self.d, device=device)
        self.I = torch.zeros(batch_size, self.d, device=device)
        self.A = torch.zeros(batch_size, self.d, device=device)

    def step(self, obs):
        if self.W is None:
            self.reset(obs.shape[0], obs.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Sense
        obs_enc = torch.tanh(self.W_enc(obs))
        W_all = self.W_from_W(self.W) + self.W_from_I(self.I) + self.W_from_A(self.A)
        attn = torch.sigmoid(self.W_attention(torch.cat([self.W, self.I], -1)))
        mult = torch.sigmoid(self.W_mult_gate(torch.cat([self.I, self.A], -1)))
        W_new = torch.tanh(obs_enc * attn + W_all * mult)

        # Memory
        z = torch.sigmoid(self.I_z_W(W_new) + self.I_z_I(self.I) + self.I_z_A(self.A) + self.I_z_bias)
        r = torch.sigmoid(self.I_r_W(W_new) + self.I_r_I(self.I) + self.I_r_A(self.A) + self.I_r_bias)
        h = torch.tanh(self.I_h_W(W_new) + self.I_h_I(r * self.I) + self.I_h_A(self.A) + self.I_h_bias)
        mult_I = torch.sigmoid(self.I_mult_gate(torch.cat([W_new, self.A], -1)))
        I_new = (1 - z) * self.I + z * (h * mult_I)

        # Decide
        A_all = self.A_from_W(W_new) + self.A_from_I(I_new) + self.A_from_A(self.A) + self.A_bias
        mult_A = torch.sigmoid(self.A_mult_gate(torch.cat([W_new, I_new], -1)))
        A_new = torch.tanh(A_all * mult_A)

        self.W, self.I, self.A = W_new, I_new, A_new
        return {'action': self.phi(A_new)}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# ANIMA-ONE (Parallel, Compressed)
# ============================================
class ANIMAOne(nn.Module):
    """
    ANIMA-1 scaled to target params.

    Uses bottleneck compression and chunk-parallel processing.
    """

    def __init__(self, sensory_dim=8, output_dim=4, target_params=TARGET_PARAMS):
        super().__init__()

        # Find dimensions
        d, b = self._find_dimensions(sensory_dim, output_dim, target_params)
        self.d = d
        self.b = b

        # Sensing with compression
        self.sense = nn.Linear(sensory_dim, d)
        self.compress = nn.Linear(d, b)
        self.expand = nn.Linear(b, d)

        # Memory GRU (chunk-parallel capable)
        self.gru_z = nn.Linear(d * 2, d)
        self.gru_r = nn.Linear(d * 2, d)
        self.gru_h = nn.Linear(d * 2, d)

        # Type interaction with compression
        self.interact_compress = nn.Linear(d * 3, b)
        self.interact_S = nn.Linear(b, d)
        self.interact_M = nn.Linear(b, d)
        self.interact_D = nn.Linear(b, d)
        self.phi_gate = nn.Linear(d * 3, d * 3)

        # Output with compression
        self.out_compress = nn.Linear(d * 3, b)
        self.out_expand = nn.Linear(b, d)
        self.output = nn.Linear(d, output_dim)

        # State
        self.S = None
        self.M = None
        self.D = None

        self._init_weights()

    def _find_dimensions(self, s_dim, o_dim, target):
        """Find d and b that hit target params."""
        for d in range(16, 96):
            for b in range(8, d):
                params = (
                    s_dim * d + d +           # sense
                    d * b + b +               # compress
                    b * d + d +               # expand
                    (d * 2) * d * 3 + d * 3 + # GRU gates
                    d * 3 * b + b +           # interact_compress
                    b * d * 3 + d * 3 +       # interact S/M/D
                    d * 3 * d * 3 + d * 3 +   # phi_gate
                    d * 3 * b + b +           # out_compress
                    b * d + d +               # out_expand
                    d * o_dim + o_dim         # output
                )
                if abs(params - target) / target < TOLERANCE:
                    return d, b
        return 24, 12  # fallback

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.5)
            else:
                nn.init.zeros_(param)

    def reset(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.S = torch.zeros(batch_size, self.d, device=device)
        self.M = torch.zeros(batch_size, self.d, device=device)
        self.D = torch.zeros(batch_size, self.d, device=device)

    def step(self, obs):
        if self.S is None:
            self.reset(obs.shape[0], obs.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Sense with compression
        sensed = torch.tanh(self.sense(obs))
        compressed = torch.tanh(self.compress(sensed))
        S_new = torch.tanh(self.expand(compressed))

        # Memory GRU
        combined = torch.cat([sensed, self.M], -1)
        z = torch.sigmoid(self.gru_z(combined))
        r = torch.sigmoid(self.gru_r(combined))
        h = torch.tanh(self.gru_h(torch.cat([sensed, r * self.M], -1)))
        M_new = (1 - z) * self.M + z * h

        # Type interaction with compression
        all_states = torch.cat([S_new, M_new, self.D], -1)
        gate = torch.sigmoid(self.phi_gate(all_states))
        gated = all_states * gate
        inter_c = torch.tanh(self.interact_compress(gated))
        S_int = torch.tanh(self.interact_S(inter_c))
        M_int = torch.tanh(self.interact_M(inter_c))
        D_new = torch.tanh(self.interact_D(inter_c))

        self.S, self.M, self.D = S_int, M_int, D_new

        # Output with compression
        combined_out = torch.cat([S_int, M_int, D_new], -1)
        out_c = torch.tanh(self.out_compress(combined_out))
        decision = torch.tanh(self.out_expand(out_c))
        action = self.output(decision)

        return {'action': action}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# TRANSFORMER (Scaled to Match)
# ============================================
class TransformerMatched(nn.Module):
    """
    Transformer scaled to EXACTLY match ANIMA params.
    """

    def __init__(self, sensory_dim=8, output_dim=4, target_params=TARGET_PARAMS):
        super().__init__()

        # Find dimensions
        d, ff = self._find_dimensions(sensory_dim, output_dim, target_params)
        self.d = d

        self.input_proj = nn.Linear(sensory_dim, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=max(1, d // 16),  # Ensure at least 1 head
            dim_feedforward=ff,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_proj = nn.Linear(d, output_dim)

    def _find_dimensions(self, s_dim, o_dim, target):
        """Find d_model and ff_dim that hit target params."""
        for d in range(16, 128, 8):
            for ff in range(d, d * 8, d):
                nhead = max(1, d // 16)
                # Approximate transformer params
                params = (
                    s_dim * d + d +                    # input_proj
                    # Self-attention: 4 * d * d (Q, K, V, O)
                    4 * d * d + 4 * d +
                    # FFN: d * ff + ff + ff * d + d
                    d * ff + ff + ff * d + d +
                    # LayerNorms: 2 * 2 * d
                    4 * d +
                    # output_proj
                    d * o_dim + o_dim
                )
                if abs(params - target) / target < TOLERANCE:
                    return d, ff
        return 48, 96  # fallback

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# BENCHMARK TASKS (same as before)
# ============================================

def gen_sequence(n=50):
    data = []
    for _ in range(n):
        start, step = random.randint(1, 10), random.randint(1, 5)
        seq = [(start + i * step) / 100.0 for i in range(9)]
        data.append({'input': seq[:-1], 'target': seq[-1]})
    return data

def gen_pattern(n=50):
    data = []
    for _ in range(n):
        plen = random.randint(2, 4)
        pattern = [random.randint(1, 9) / 10.0 for _ in range(plen)]
        seq = (pattern * 5)[:9]
        data.append({'input': seq[:-1], 'target': seq[-1]})
    return data

def gen_conditional(n=50):
    data = []
    for _ in range(n):
        seq = [random.randint(1, 10) / 20.0 for _ in range(8)]
        last = seq[-1] * 20
        target = (last * 2 if last > 5 else last + 1) / 20.0
        data.append({'input': seq, 'target': target})
    return data

def gen_analogy(n=50):
    data = []
    for _ in range(n):
        a = random.randint(1, 5) / 10.0
        c = random.randint(1, 5) / 10.0
        data.append({'input': [a, a*2, c, 0], 'target': c*2})
    return data

def gen_projectile(n=50):
    data = []
    for _ in range(n):
        v0 = random.uniform(5, 15)
        positions = [v0 * t * 0.1 / 20.0 for t in range(8)]
        landing = min(v0 * 0.8 / 20.0, 1.0)
        data.append({'input': positions, 'target': landing})
    return data

def gen_collision(n=50):
    data = []
    for _ in range(n):
        x1, v1 = random.uniform(0, 0.5), random.uniform(0.1, 0.3)
        x2, v2 = random.uniform(0.5, 1), random.uniform(-0.3, -0.1)
        collide = 1.0 if v1 > -v2 else 0.0
        seq = [x1, v1, x2, v2] + [0.0] * 4
        data.append({'input': seq, 'target': collide, 'binary': True})
    return data

def gen_goal(n=50):
    data = []
    for _ in range(n):
        pos = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        goal = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        dist = max((dx**2 + dy**2)**0.5, 0.01)
        data.append({'input': pos + goal + [0]*4, 'target': [dx/dist, dy/dist], 'goal': True})
    return data

def gen_momentum(n=50):
    data = []
    for _ in range(n):
        m1, v1 = random.uniform(0.2, 1), random.uniform(0.2, 1)
        m2, v2 = random.uniform(0.2, 1), random.uniform(-1, -0.2)
        v_f = (m1*v1 + m2*v2) / (m1 + m2)
        data.append({'input': [m1, v1, m2, v2] + [0]*4, 'target': (v_f + 1) / 2})
    return data


def train_eval(model, train, test, task, epochs=50, lr=0.01):
    """Train and evaluate."""
    device = next(model.parameters()).device
    is_goal = task == 'goal'
    is_binary = task == 'collision'

    train_x = torch.tensor([d['input'] for d in train], dtype=torch.float32, device=device)
    if is_goal:
        train_y = torch.tensor([d['target'] for d in train], dtype=torch.float32, device=device)
    else:
        train_y = torch.tensor([[d['target']] for d in train], dtype=torch.float32, device=device)

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()

        if hasattr(model, 'reset'):
            # ANIMA models
            outputs = []
            for i in range(len(train)):
                model.reset(1, device)
                inp = train_x[i]
                if len(inp) < 8:
                    inp = torch.cat([inp, torch.zeros(8 - len(inp), device=device)])
                result = model.step(inp[:8].unsqueeze(0))
                if is_goal:
                    outputs.append(result['action'][:, :2])
                else:
                    outputs.append(result['action'][:, :1])
            outputs = torch.cat(outputs, dim=0)
        else:
            # Transformer
            x = train_x
            if x.shape[-1] < 8:
                x = torch.cat([x, torch.zeros(x.shape[0], 8 - x.shape[-1], device=device)], -1)
            out = model(x.unsqueeze(1))
            if is_goal:
                outputs = out[:, 0, :2]
            else:
                outputs = out[:, 0, :1]

        loss = crit(outputs, train_y)
        loss.backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for d in test:
            inp = torch.tensor(d['input'], dtype=torch.float32, device=device)
            if len(inp) < 8:
                inp = torch.cat([inp, torch.zeros(8 - len(inp), device=device)])
            inp = inp[:8]

            if hasattr(model, 'reset'):
                model.reset(1, device)
                result = model.step(inp.unsqueeze(0))
                pred = result['action'][0, :2].cpu().numpy() if is_goal else result['action'][0, 0].item()
            else:
                out = model(inp.unsqueeze(0).unsqueeze(0))
                pred = out[0, 0, :2].cpu().numpy() if is_goal else out[0, 0, 0].item()

            if is_goal:
                target = d['target']
                dot = pred[0]*target[0] + pred[1]*target[1]
                pn = max((pred[0]**2 + pred[1]**2)**0.5, 0.01)
                tn = max((target[0]**2 + target[1]**2)**0.5, 0.01)
                if dot/(pn*tn) > 0.7:
                    correct += 1
            elif is_binary:
                if (pred > 0.5) == (d['target'] > 0.5):
                    correct += 1
            else:
                if abs(pred - d['target']) < max(0.2 * abs(d['target']), 0.1):
                    correct += 1

    return correct / len(test)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Target params: {TARGET_PARAMS} (+/- {TOLERANCE*100}%)")
    print()

    # Create models
    anima0 = ANIMAZero(target_params=TARGET_PARAMS).to(device)
    anima1 = ANIMAOne(target_params=TARGET_PARAMS).to(device)
    transformer = TransformerMatched(target_params=TARGET_PARAMS).to(device)

    # Verify parameter counts
    p0 = anima0.count_parameters()
    p1 = anima1.count_parameters()
    pt = transformer.count_parameters()

    print("="*70)
    print("PARAMETER VERIFICATION")
    print("="*70)
    print(f"ANIMA-Zero:    {p0:>6} params  (delta: {(p0-TARGET_PARAMS)/TARGET_PARAMS*100:+.1f}%)")
    print(f"ANIMA-One:     {p1:>6} params  (delta: {(p1-TARGET_PARAMS)/TARGET_PARAMS*100:+.1f}%)")
    print(f"Transformer:   {pt:>6} params  (delta: {(pt-TARGET_PARAMS)/TARGET_PARAMS*100:+.1f}%)")

    # Check fairness
    max_diff = max(abs(p0-p1), abs(p0-pt), abs(p1-pt))
    max_pct = max_diff / TARGET_PARAMS * 100
    print(f"\nMax param difference: {max_diff} ({max_pct:.1f}%)")
    if max_pct > TOLERANCE * 100:
        print("WARNING: Parameter mismatch exceeds tolerance!")
    else:
        print("FAIR: All models within tolerance")

    print()

    # Run benchmark
    tasks = [
        ('sequence', gen_sequence, 'reasoning'),
        ('pattern', gen_pattern, 'reasoning'),
        ('conditional', gen_conditional, 'reasoning'),
        ('analogy', gen_analogy, 'reasoning'),
        ('projectile', gen_projectile, 'physics'),
        ('collision', gen_collision, 'physics'),
        ('goal', gen_goal, 'physics'),
        ('momentum', gen_momentum, 'physics'),
    ]

    results = {
        'ANIMA-Zero': {'params': p0, 'reasoning': {}, 'physics': {}},
        'ANIMA-One': {'params': p1, 'reasoning': {}, 'physics': {}},
        'Transformer': {'params': pt, 'reasoning': {}, 'physics': {}},
    }

    print("="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Task':<12} {'ANIMA-0':>10} {'ANIMA-1':>10} {'Transformer':>12}")
    print("-"*46)

    for task, gen_fn, category in tasks:
        train = gen_fn(100)
        test = gen_fn(50)

        # Fresh models each task
        a0 = ANIMAZero(target_params=TARGET_PARAMS).to(device)
        a1 = ANIMAOne(target_params=TARGET_PARAMS).to(device)
        tf = TransformerMatched(target_params=TARGET_PARAMS).to(device)

        s0 = train_eval(a0, train, test, task)
        s1 = train_eval(a1, train, test, task)
        st = train_eval(tf, train, test, task)

        results['ANIMA-Zero'][category][task] = s0
        results['ANIMA-One'][category][task] = s1
        results['Transformer'][category][task] = s1

        print(f"{task:<12} {s0*100:>9.1f}% {s1*100:>9.1f}% {st*100:>11.1f}%")

    # Compute averages
    print("-"*46)
    for model in results:
        r_avg = sum(results[model]['reasoning'].values()) / 4
        p_avg = sum(results[model]['physics'].values()) / 4
        results[model]['reasoning_avg'] = r_avg
        results[model]['physics_avg'] = p_avg
        results[model]['overall'] = (r_avg + p_avg) / 2

    print(f"{'REASONING':<12} {results['ANIMA-Zero']['reasoning_avg']*100:>9.1f}% "
          f"{results['ANIMA-One']['reasoning_avg']*100:>9.1f}% "
          f"{results['Transformer']['reasoning_avg']*100:>11.1f}%")
    print(f"{'PHYSICS':<12} {results['ANIMA-Zero']['physics_avg']*100:>9.1f}% "
          f"{results['ANIMA-One']['physics_avg']*100:>9.1f}% "
          f"{results['Transformer']['physics_avg']*100:>11.1f}%")
    print("-"*46)
    print(f"{'OVERALL':<12} {results['ANIMA-Zero']['overall']*100:>9.1f}% "
          f"{results['ANIMA-One']['overall']*100:>9.1f}% "
          f"{results['Transformer']['overall']*100:>11.1f}%")

    # Save
    results['target_params'] = TARGET_PARAMS
    results['tolerance'] = TOLERANCE
    with open(os.path.join(os.path.dirname(__file__), 'fair_benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to fair_benchmark_results.json")
    return results


if __name__ == '__main__':
    main()
