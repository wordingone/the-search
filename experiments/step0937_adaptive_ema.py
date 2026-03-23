"""
step0937_adaptive_ema.py -- Adaptive EMA decay rate (chemotaxis-inspired).

R3 hypothesis: Adaptive memory timescale from observation statistics enables
position-sensitive tracking without per-state memory. The substrate auto-tunes to
the environment's temporal structure. High obs-change variance (FT09, frequent resets)
→ shorter memory → faster adaptation. Low variance (LS20, gradual navigation)
→ longer memory → stable tracking.

Architecture: Same as 916 (recurrent h + clamped alpha + 800b). ONLY change:
fixed EMA decay lambda=0.9 → adaptive lambda_t derived from obs_change variance.

Formula (Leo spec, mail 2707):
  obs_change = ||enc_t - enc_{t-1}||
  change_var = 0.99 * change_var + 0.01 * (obs_change - change_mean)**2
  change_mean = 0.99 * change_mean + 0.01 * obs_change
  lambda_t = clip(1.0 - 1.0 / (1.0 + change_var), 0.5, 0.99)
  delta_per_action[a] = lambda_t * delta[a] + (1 - lambda_t) * obs_change

NOTE: Formula is implemented verbatim. Lambda_t actual values logged per game
as diagnostic — verify whether high-var games get short vs long memory as predicted.

Kill criteria: LS20 < 200 → KILL. Chain kill from judge.
Run: PRISM-light Mode C (randomized), 10K steps/phase, 5 seeds.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320
ETA_W = 0.01
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00
LAMBDA_LO = 0.50
LAMBDA_HI = 0.99
TEST_SEEDS = list(range(1, 6))
PHASE_STEPS = 10_000
DIAG_STEPS = {1_000, 5_000, 10_000}


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_action(vals, temp, rng):
    x = vals / temp; x -= np.max(x); e = np.exp(x)
    return int(rng.choice(len(vals), p=e / (e.sum() + 1e-12)))


class Chain937:
    """916 recurrent h + clamped alpha + 800b with adaptive EMA decay lambda_t.

    Alpha + running_mean persist across games. W_pred + h + delta + change stats reset.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)

        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1

        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)

        self._n = None
        self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None

        # Adaptive lambda state
        self._change_mean = 0.0
        self._change_var = 0.0
        self._lambda_t = 0.9   # starts at fixed rate, adapts

        self._prev_ext = None
        self._prev_enc = None
        self._prev_action = None

        self._phase_step = 0
        self._diag_log = {}

    def set_game(self, n_actions):
        self._n = n_actions
        self.W_pred = np.zeros((EXT_DIM, EXT_DIM + n_actions), dtype=np.float32)
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._change_mean = 0.0
        self._change_var = 0.0
        self._lambda_t = 0.9
        self._prev_ext = None
        self._prev_enc = None
        self._prev_action = None
        self._phase_step = 0
        self._diag_log = {}

    def reset_seed(self, seed):
        self._rng = np.random.RandomState(seed)
        rs = np.random.RandomState(seed + 10000)
        self.W_h = rs.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_x = rs.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.alpha = np.ones(EXT_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._pred_errors.clear()
        self._n = None; self.W_pred = None
        self.h = np.zeros(H_DIM, dtype=np.float32)
        self.delta_per_action = None
        self._change_mean = 0.0; self._change_var = 0.0; self._lambda_t = 0.9
        self._prev_ext = None; self._prev_enc = None; self._prev_action = None
        self._phase_step = 0; self._diag_log = {}

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        enc = enc_raw - self._running_mean
        self.h = np.tanh(self.W_h @ self.h + self.W_x @ enc)
        ext_enc = np.concatenate([enc, self.h]).astype(np.float32)
        return enc, ext_enc

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        me = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(me)) or np.any(np.isinf(me)):
            return
        ra = np.sqrt(np.clip(me, 0, 1e6) + 1e-8)
        mr = np.mean(ra)
        if mr < 1e-8 or np.isnan(mr):
            return
        self.alpha = np.clip(ra / mr, ALPHA_LO, ALPHA_HI)

    def _update_lambda(self, obs_change):
        """Adaptive EMA decay from observation change variance (Leo spec, mail 2707)."""
        self._change_var = 0.99 * self._change_var + 0.01 * (obs_change - self._change_mean) ** 2
        self._change_mean = 0.99 * self._change_mean + 0.01 * obs_change
        self._lambda_t = float(np.clip(1.0 - 1.0 / (1.0 + self._change_var),
                                       LAMBDA_LO, LAMBDA_HI))

    def process(self, obs):
        enc, ext_enc = self._encode(obs)
        self._phase_step += 1

        if self._prev_ext is not None and self._prev_action is not None:
            # W_pred training (same as 916)
            inp = np.concatenate([self._prev_ext * self.alpha,
                                   one_hot(self._prev_action, self._n)])
            pred = self.W_pred @ inp
            error = (ext_enc * self.alpha) - pred
            en = float(np.linalg.norm(error))
            if en > 10.0:
                error = error * (10.0 / en)
            if not np.any(np.isnan(error)):
                self.W_pred -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            # obs_change = ||enc_t - enc_{t-1}|| (raw enc, no alpha, no h)
            obs_change = float(np.linalg.norm(enc - self._prev_enc))

            # Update adaptive lambda from obs_change variance
            self._update_lambda(obs_change)

            # 800b with adaptive lambda (THE ONLY CHANGE from 916)
            a = self._prev_action
            self.delta_per_action[a] = (self._lambda_t * self.delta_per_action[a]
                                         + (1 - self._lambda_t) * obs_change)

        # Diagnostic snapshots
        if self._phase_step in DIAG_STEPS:
            self._diag_log[self._phase_step] = {
                "lambda_t": self._lambda_t,
                "change_var": self._change_var,
                "change_mean": self._change_mean,
                "delta_min": float(self.delta_per_action.min()),
                "delta_max": float(self.delta_per_action.max()),
            }

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = softmax_action(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_ext = ext_enc.copy()
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_ext = None
        self._prev_enc = None
        self._prev_action = None

    def alpha_conc(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))

    def diag_str(self):
        parts = []
        for step in sorted(self._diag_log):
            d = self._diag_log[step]
            parts.append(f"@{step//1000}K: λ={d['lambda_t']:.3f}"
                         f" var={d['change_var']:.3f} mean={d['change_mean']:.3f}"
                         f" delta=[{d['delta_min']:.3f},{d['delta_max']:.3f}]")
        return "  ".join(parts) if parts else "n/a"


def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def load_cifar():
    try:
        import torchvision, torchvision.transforms as T
        ds = torchvision.datasets.CIFAR100('B:/M/the-search/data', train=False,
                                            download=True, transform=T.ToTensor())
        imgs = np.array([np.array(ds[i][0]).transpose(1, 2, 0)
                         for i in range(len(ds))], dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"CIFAR load failed: {e}"); return None, None


def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None: return 0.0
    sub.set_game(100)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions)
    env = make_env(game)
    obs = env.reset(seed=seed); step = 0; completions = 0; level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            completions += cl - level; level = cl; sub.on_level_transition()
        if done:
            obs = env.reset(seed=seed); level = 0; sub.on_level_transition()
    return completions


print("=" * 70)
print("STEP 937 — ADAPTIVE EMA DECAY (chemotaxis-inspired, λ_t from obs_change variance)")
print("=" * 70)
print("R3: lambda_t = clip(1 - 1/(1+change_var), 0.5, 0.99). ONE change from 916.")
print("High obs-change var → adaptive lambda. Log actual lambda per game for verification.")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()
cifar1 = []; ls20 = []; ft09 = []; vc33 = []; cifar2 = []

for seed in TEST_SEEDS:
    sub = Chain937(seed=seed)
    sub.reset_seed(seed)
    print(f"\n  Seed {seed}:")

    c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, PHASE_STEPS)
    cifar1.append(c1)
    print(f"    CIFAR-1: acc={c1:.4f}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    l = run_arc(sub, "LS20", 4, seed * 1000, PHASE_STEPS)
    ls20.append(l)
    print(f"    LS20:    L1={l:4d}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    f = run_arc(sub, "FT09", 68, seed * 1000, PHASE_STEPS)
    ft09.append(f)
    print(f"    FT09:    L1={f:4d}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    v = run_arc(sub, "VC33", 68, seed * 1000, PHASE_STEPS)
    vc33.append(v)
    print(f"    VC33:    L1={v:4d}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

    c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, PHASE_STEPS)
    cifar2.append(c2)
    print(f"    CIFAR-2: acc={c2:.4f}  alpha_conc={sub.alpha_conc():.2f}")
    print(f"             diag: {sub.diag_str()}")

print(f"\n{'=' * 70}")
print(f"STEP 937 RESULTS (adaptive lambda, PRISM-light, 10K/phase, 5 seeds):")
print(f"  CIFAR-1 acc: {np.mean(cifar1):.4f}  {[f'{x:.3f}' for x in cifar1]}")
print(f"  LS20  L1:    {np.mean(ls20):.1f}/seed  std={np.std(ls20):.1f}  zero={sum(1 for x in ls20 if x == 0)}/5  {ls20}")
print(f"  FT09  L1:    {np.mean(ft09):.1f}/seed  zero={sum(1 for x in ft09 if x == 0)}/5  {ft09}")
print(f"  VC33  L1:    {np.mean(vc33):.1f}/seed  zero={sum(1 for x in vc33 if x == 0)}/5  {vc33}")
print(f"  CIFAR-2 acc: {np.mean(cifar2):.4f}  {[f'{x:.3f}' for x in cifar2]}")
print(f"\nComparison (chain, 5 seeds):")
print(f"  916 (fixed lambda=0.9, 25K): LS20=290.7  FT09=0  VC33=0")
print(f"  914 (fixed lambda=0.9, 10K): LS20=248.6  FT09=0  VC33=0")
print(f"  937 (adaptive lambda, 10K):  LS20={np.mean(ls20):.1f}  FT09={np.mean(ft09):.1f}  VC33={np.mean(vc33):.1f}")
print(f"\nKill criterion: LS20 < 200 → KILL.")
print(f"Total elapsed: {time.time() - t0:.1f}s")
print("STEP 937 DONE")
