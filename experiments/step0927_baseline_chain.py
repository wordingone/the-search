"""
step0927_baseline_chain.py -- Published baselines on the FULL CHAIN.

R3 hypothesis: Cross-game transfer context (via persistent state) is what
separates 916 from published methods. Baselines that reset per game can't
carry trajectory context.

Per Leo mail 2675: run ICM/RND/Count-based/Random on make_default_chain()
(CIFAR→LS20→FT09→VC33→CIFAR). Same substrate persists across games per seed.
Compare standalone results (917-920) vs chain.

Runtime cap: 10K steps per game (reduced from 25K to stay under 5-min cap).
5 seeds. 4 baselines.

Baselines:
  Random:      pure uniform random action selector (compute floor)
  Count-based: obs_hash visit count + argmin (step 919)
  RND:         Random Network Distillation novelty + 800b (step 918)
  ICM:         Intrinsic Curiosity Module + 800b (step 917)

Comparison chain: Step 914 (895h, 5 seeds, 25K), Step 926 (916h, 5 seeds, 25K).
"""
import sys, time, hashlib
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque, defaultdict
from substrates.step0674 import _enc_frame

ENC_DIM = 256
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
EPSILON = 0.20
SOFTMAX_TEMP = 0.10

TEST_SEEDS = list(range(1, 6))
PHASE_STEPS = 10_000   # 10K per game (5-min cap compliance)

# Chain: (game_name, n_actions). None = CIFAR special handling.
GAME_SEQUENCE = [
    ("CIFAR", 100),
    ("LS20",  4),
    ("FT09",  68),
    ("VC33",  68),
    ("CIFAR", 100),
]


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v


def softmax_sel(delta, temp, rng):
    x = np.array(delta) / temp; x -= np.max(x); e = np.exp(x)
    probs = e / (e.sum() + 1e-12)
    return int(rng.choice(len(delta), p=probs))


# ─────────────────────────────────────────────
# BASELINE 1: Random (compute floor)
# ─────────────────────────────────────────────
class RandomSubstrate:
    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self._n = None

    def set_game(self, n_actions):
        self._n = n_actions

    def process(self, obs):
        return int(self._rng.randint(0, self._n))

    def on_level_transition(self): pass


# ─────────────────────────────────────────────
# BASELINE 2: Count-based (step 919)
# ─────────────────────────────────────────────
class CountBasedSubstrate:
    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        # Persistent across games
        self._visit_count = defaultdict(lambda: defaultdict(int))

    def set_game(self, n_actions):
        self._n = n_actions
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _hash(self, enc):
        rounded = np.round(enc, 1).astype(np.float16)
        return hashlib.md5(rounded.tobytes()).hexdigest()[:12]

    def process(self, obs):
        enc = self._encode(obs)
        h = self._hash(enc)
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            counts = np.array([self._visit_count[h][a] for a in range(self._n)])
            candidates = np.where(counts == counts.min())[0]
            action = int(self._rng.choice(candidates))
        self._visit_count[h][action] += 1
        return action

    def on_level_transition(self): pass


# ─────────────────────────────────────────────
# BASELINE 3: RND (step 918)
# ─────────────────────────────────────────────
class RNDSubstrate:
    def __init__(self, seed):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._n = None
        # RND networks (persistent — accumulate across games)
        rs = np.random.RandomState(seed + 99999)
        self.W_target = rs.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.W_pred = np.zeros((ENC_DIM, ENC_DIM), dtype=np.float32)
        self.delta_per_action = None
        self._prev_enc = None; self._prev_action = None

    def set_game(self, n_actions):
        self._n = n_actions
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def process(self, obs):
        enc = self._encode(obs)
        # RND novelty signal
        target = np.tanh(self.W_target @ enc)
        pred = np.tanh(self.W_pred @ enc)
        novelty = float(np.linalg.norm(target - pred))
        # Update predictor
        err = target - pred; en = float(np.linalg.norm(err))
        if en > 1.0: err *= 1.0 / en
        self.W_pred += ETA_W * np.outer(err, enc)

        if self._prev_enc is not None and self._prev_action is not None:
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * novelty)

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None


# ─────────────────────────────────────────────
# BASELINE 4: ICM (step 917)
# ─────────────────────────────────────────────
class ICMSubstrate:
    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._n = None
        self.W_f = None  # forward model (reset per game — n_actions changes)
        self.delta_per_action = None
        self._prev_enc = None; self._prev_action = None

    def set_game(self, n_actions):
        self._n = n_actions
        self.W_f = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def process(self, obs):
        enc = self._encode(obs)

        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc, one_hot(self._prev_action, self._n)])
            pred = self.W_f @ inp
            error = enc - pred
            icm_reward = float(np.linalg.norm(error))
            en = float(np.linalg.norm(error))
            if en > 5.0: error *= 5.0 / en
            self.W_f += ETA_W * np.outer(error, inp)
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * icm_reward)

        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n))
        else:
            action = softmax_sel(self.delta_per_action, SOFTMAX_TEMP, self._rng)

        self._prev_enc = enc.copy(); self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None


# ─────────────────────────────────────────────
# Chain runner
# ─────────────────────────────────────────────
def load_cifar():
    try:
        import torchvision, torchvision.transforms as T
        ds = torchvision.datasets.CIFAR100('B:/M/the-search/data', train=False,
                                            download=True, transform=T.ToTensor())
        imgs = np.array([np.array(ds[i][0]).transpose(1,2,0) for i in range(len(ds))],
                        dtype=np.float32)
        lbls = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return imgs, lbls
    except Exception as e:
        print(f"  CIFAR load failed: {e}"); return None, None


def make_env(name):
    try:
        import arcagi3; return arcagi3.make(name)
    except:
        import util_arcagi3; return util_arcagi3.make(name)


def run_cifar(sub, imgs, lbls, seed, n_steps):
    if imgs is None: return 0.0
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(imgs))[:n_steps]
    correct = sum(1 for i in idx if sub.process(imgs[i]) % 100 == lbls[i])
    return correct / len(idx)


def run_arc(sub, game, n_actions, seed, n_steps):
    sub.set_game(n_actions)
    env = make_env(game)
    obs = env.reset(seed=seed)
    step = 0; completions = 0; level = 0
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


def run_chain(SubClass, name, seeds, n_steps, cifar_imgs, cifar_lbls):
    print(f"\n--- {name} ---")
    cifar1_list = []; ls20_list = []; ft09_list = []; vc33_list = []; cifar2_list = []
    for seed in seeds:
        sub = SubClass(seed=seed)
        # CIFAR-1
        sub.set_game(100)
        c1 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000, n_steps)
        cifar1_list.append(c1)
        # LS20
        l = run_arc(sub, "LS20", 4, seed * 1000, n_steps)
        ls20_list.append(l)
        # FT09
        f = run_arc(sub, "FT09", 68, seed * 1000, n_steps)
        ft09_list.append(f)
        # VC33
        v = run_arc(sub, "VC33", 68, seed * 1000, n_steps)
        vc33_list.append(v)
        # CIFAR-2
        sub.set_game(100)
        c2 = run_cifar(sub, cifar_imgs, cifar_lbls, seed * 1000 + 1, n_steps)
        cifar2_list.append(c2)
        print(f"  seed={seed}: CIFAR1={c1:.3f} LS20={l:3d} FT09={f:3d} VC33={v:3d} CIFAR2={c2:.3f}")
    return cifar1_list, ls20_list, ft09_list, vc33_list, cifar2_list


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
print("=" * 70)
print("STEP 927 — BASELINE CHAIN (ICM/RND/Count/Random on CIFAR→LS20→FT09→VC33→CIFAR)")
print("=" * 70)
print(f"10K steps/game, 5 seeds. Outer=seeds, inner=games (same substrate persists).")
t0 = time.time()

cifar_imgs, cifar_lbls = load_cifar()

BASELINES = [
    ("Random",       RandomSubstrate),
    ("Count-based",  CountBasedSubstrate),
    ("RND",          RNDSubstrate),
    ("ICM",          ICMSubstrate),
]

summary = {}
for bname, bclass in BASELINES:
    c1, ls, ft, vc, c2 = run_chain(bclass, bname, TEST_SEEDS, PHASE_STEPS,
                                     cifar_imgs, cifar_lbls)
    summary[bname] = dict(
        cifar1=np.mean(c1), ls20=np.mean(ls), ft09=np.mean(ft),
        vc33=np.mean(vc), cifar2=np.mean(c2),
        ls20_list=ls, ft09_list=ft, vc33_list=vc
    )
    print(f"  {bname}: CIFAR1={np.mean(c1):.3f} LS20={np.mean(ls):.1f} "
          f"FT09={np.mean(ft):.1f} VC33={np.mean(vc):.1f} CIFAR2={np.mean(c2):.3f}")

print(f"\n{'='*70}")
print(f"STEP 927 RESULTS — BASELINE CHAIN (10K/game, 5 seeds):")
print(f"{'Substrate':<15} {'CIFAR-1':>8} {'LS20':>8} {'FT09':>8} {'VC33':>8} {'CIFAR-2':>8}")
print("-" * 57)
for bname, s in summary.items():
    print(f"{bname:<15} {s['cifar1']:>8.3f} {s['ls20']:>8.1f} {s['ft09']:>8.1f} {s['vc33']:>8.1f} {s['cifar2']:>8.3f}")
print("-" * 57)
print(f"{'916 chain*':<15} {'?':>8} {'?':>8} {'0.0':>8} {'0.0':>8} {'?':>8}")
print(f"{'895h chain*':<15} {'~0.01':>8} {'248.6':>8} {'0.0':>8} {'0.0':>8} {'~0.01':>8}")
print(f"\n* 916/895h chain used 25K/game, 5 seeds — not directly comparable to 10K.")
print(f"\nKill criterion: any baseline with LS20 > 916 chain → 916 is NOT adding value.")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("STEP 927 DONE")
