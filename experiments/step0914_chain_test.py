"""
step0914_chain_test.py -- Full chain test: CIFAR→LS20→FT09→VC33→CIFAR.

R3 hypothesis: alpha R3 encoding generalizes across game domains.
Alpha (attention weights) persists across all game phases within a seed.
W and delta reset per game (n_actions changes between games).

Architecture: 895h cold (clamped alpha + 800b change-tracking).
- CIFAR-100: 100 class actions. L1 = first step with >10% rolling acc.
- LS20: 4 navigation actions.
- FT09: 68 click positions.
- VC33: 68 actions (4 dirs + 64 grid clicks).
- CIFAR-100 again: same 100-class setup.

Protocol: 25K per game phase, 5 seeds, cold start (no pretrain).
This is the honest baseline — where do we ACTUALLY stand?

Leo mail 2647. Priority: HIGHEST.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

TEST_SEEDS = list(range(1, 6))
PHASE_STEPS = 25_000
ETA_W = 0.01
ALPHA_EMA = 0.10
INIT_DELTA = 1.0
ALPHA_UPDATE_DELAY = 50
ENC_DIM = 256
EPSILON = 0.20
SOFTMAX_TEMP = 0.10
ALPHA_LO = 0.10
ALPHA_HI = 5.00

# Game sequence: (name, n_actions)
GAME_SEQUENCE = [
    ("CIFAR", 100),
    ("LS20",  4),
    ("FT09",  68),
    ("VC33",  68),
    ("CIFAR", 100),
]


def one_hot(a, n):
    v = np.zeros(n, dtype=np.float32); v[a] = 1.0
    return v


def softmax_action(delta, temp):
    x = delta / temp
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class Chain895h:
    """895h (clamped alpha + 800b) for multi-game chain.
    Alpha and running_mean persist across all game phases.
    W and delta_per_action reset when switching games.
    """

    def __init__(self, seed=0, epsilon=EPSILON):
        self._seed = seed
        self._epsilon = epsilon
        self._rng = np.random.RandomState(seed)
        # Persistent across games:
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._pred_errors = deque(maxlen=200)
        # Game-specific (reset per game):
        self._n_actions = None
        self.W = None
        self.delta_per_action = None
        self._prev_enc = None
        self._prev_action = None

    def set_game(self, n_actions):
        """Call when transitioning to a new game. Resets W+delta, preserves alpha."""
        self._n_actions = n_actions
        self.W = np.zeros((ENC_DIM, ENC_DIM + n_actions), dtype=np.float32)
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._pred_errors.clear()
        self._prev_enc = None
        self._prev_action = None

    def reset_seed(self, seed):
        """Full reset including alpha (new seed)."""
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.alpha = np.ones(ENC_DIM, dtype=np.float32)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._pred_errors.clear()
        self._n_actions = None
        self.W = None
        self.delta_per_action = None
        self._prev_enc = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return enc_raw - self._running_mean

    def _update_alpha(self):
        if len(self._pred_errors) < ALPHA_UPDATE_DELAY:
            return
        mean_errors = np.mean(self._pred_errors, axis=0)
        if np.any(np.isnan(mean_errors)) or np.any(np.isinf(mean_errors)):
            return
        raw_alpha = np.sqrt(np.clip(mean_errors, 0, 1e6) + 1e-8)
        mean_raw = np.mean(raw_alpha)
        if mean_raw < 1e-8 or np.isnan(mean_raw):
            return
        self.alpha = raw_alpha / mean_raw
        self.alpha = np.clip(self.alpha, ALPHA_LO, ALPHA_HI)

    def process(self, observation):
        enc = self._encode(observation)

        if self._prev_enc is not None and self._prev_action is not None:
            inp = np.concatenate([self._prev_enc * self.alpha,
                                   one_hot(self._prev_action, self._n_actions)])
            pred = self.W @ inp
            error = (enc * self.alpha) - pred

            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self.W -= ETA_W * np.outer(error, inp)
                self._pred_errors.append(np.abs(error))
                self._update_alpha()

            weighted_delta = (enc - self._prev_enc) * self.alpha
            change = float(np.linalg.norm(weighted_delta))
            a = self._prev_action
            self.delta_per_action[a] = ((1 - ALPHA_EMA) * self.delta_per_action[a]
                                         + ALPHA_EMA * change)

        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            probs = softmax_action(self.delta_per_action, SOFTMAX_TEMP)
            action = int(self._rng.choice(self._n_actions, p=probs))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def alpha_concentration(self):
        return float(np.max(self.alpha) / (np.min(self.alpha) + 1e-8))


def make_env(game_name):
    try:
        import arcagi3; return arcagi3.make(game_name)
    except:
        import util_arcagi3; return util_arcagi3.make(game_name)


def load_cifar():
    """Load CIFAR-100 test set. Returns (images float32 HWC 0-1, labels)."""
    try:
        import torchvision
        import torchvision.transforms as transforms
        import os
        data_dir = 'B:/M/the-search/data'
        ds = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True,
            transform=transforms.ToTensor()
        )
        images = np.array([np.array(ds[i][0]).transpose(1, 2, 0)
                           for i in range(len(ds))], dtype=np.float32)
        labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int32)
        return images, labels
    except Exception as e:
        print(f"  CIFAR-100 load failed: {e}")
        return None, None


def run_arc_phase(sub, game_name, n_actions, env_seed, n_steps):
    """Run one ARC game phase. Returns level completions."""
    sub.set_game(n_actions)
    env = make_env(game_name)
    obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); sub.on_level_transition(); continue
        action = sub.process(np.asarray(obs, dtype=np.float32)) % n_actions
        obs, _, done, info = env.step(action); step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            sub.on_level_transition()
        if done:
            obs = env.reset(seed=env_seed); current_level = 0
            sub.on_level_transition()
    return completions


def run_cifar_phase(sub, n_actions, images, labels, rng_seed, n_steps):
    """Run one CIFAR-100 phase. Returns (correct, total, accuracy)."""
    if images is None:
        return 0, 0, 0.0
    sub.set_game(n_actions)
    rng = np.random.RandomState(rng_seed)
    idx = rng.permutation(len(images))[:n_steps]
    correct = 0; steps = 0
    for img_idx in idx:
        obs = images[img_idx]  # float32 HWC 0-1
        label = int(labels[img_idx])
        action = sub.process(obs) % n_actions
        if action == label:
            correct += 1
        steps += 1
    accuracy = correct / max(steps, 1)
    return correct, steps, accuracy


print("=" * 70)
print("STEP 914 — FULL CHAIN TEST: CIFAR→LS20→FT09→VC33→CIFAR")
print("=" * 70)
print(f"Architecture: 895h cold (clamped alpha + 800b). Alpha persists across games.")
print(f"W + delta reset per game (n_actions: CIFAR=100, LS20=4, FT09=68, VC33=68).")
print(f"25K per phase, 5 seeds, cold start. Honest baseline.")

t0 = time.time()
cifar_images, cifar_labels = load_cifar()
cifar_ok = cifar_images is not None
print(f"CIFAR-100: {'loaded' if cifar_ok else 'NOT AVAILABLE'}")

all_results = {g: [] for g, _ in GAME_SEQUENCE}
# CIFAR appears twice — track separately
cifar1_results = []
cifar2_results = []

for ts in TEST_SEEDS:
    sub = Chain895h(seed=ts)
    sub.reset_seed(ts)
    print(f"\n  Seed {ts}:")

    # Phase 1: CIFAR
    c1, s1, acc1 = run_cifar_phase(sub, 100, cifar_images, cifar_labels, ts * 1000, PHASE_STEPS)
    cifar1_results.append(acc1)
    print(f"    CIFAR-1: acc={acc1:.4f} ({c1}/{s1})  alpha_conc={sub.alpha_concentration():.2f}")

    # Phase 2: LS20
    ls20_l1 = run_arc_phase(sub, "LS20", 4, ts * 1000, PHASE_STEPS)
    all_results["LS20"].append(ls20_l1)
    print(f"    LS20:    L1={ls20_l1:4d}  alpha_conc={sub.alpha_concentration():.2f}")

    # Phase 3: FT09
    ft09_l1 = run_arc_phase(sub, "FT09", 68, ts * 1000, PHASE_STEPS)
    all_results["FT09"].append(ft09_l1)
    print(f"    FT09:    L1={ft09_l1:4d}  alpha_conc={sub.alpha_concentration():.2f}")

    # Phase 4: VC33
    vc33_l1 = run_arc_phase(sub, "VC33", 68, ts * 1000, PHASE_STEPS)
    all_results["VC33"].append(vc33_l1)
    print(f"    VC33:    L1={vc33_l1:4d}  alpha_conc={sub.alpha_concentration():.2f}")

    # Phase 5: CIFAR again
    c2, s2, acc2 = run_cifar_phase(sub, 100, cifar_images, cifar_labels, ts * 1000 + 1, PHASE_STEPS)
    cifar2_results.append(acc2)
    print(f"    CIFAR-2: acc={acc2:.4f} ({c2}/{s2})  alpha_conc={sub.alpha_concentration():.2f}")

print(f"\n{'='*70}")
print(f"CHAIN RESULTS (5 seeds):")
print(f"  CIFAR-1 acc:  {np.mean(cifar1_results):.4f}  {[f'{x:.3f}' for x in cifar1_results]}")
print(f"  LS20  L1:     {np.mean(all_results['LS20']):.1f}/seed  std={np.std(all_results['LS20']):.1f}  zero={sum(1 for x in all_results['LS20'] if x==0)}/5  {all_results['LS20']}")
print(f"  FT09  L1:     {np.mean(all_results['FT09']):.1f}/seed  std={np.std(all_results['FT09']):.1f}  zero={sum(1 for x in all_results['FT09'] if x==0)}/5  {all_results['FT09']}")
print(f"  VC33  L1:     {np.mean(all_results['VC33']):.1f}/seed  std={np.std(all_results['VC33']):.1f}  zero={sum(1 for x in all_results['VC33'] if x==0)}/5  {all_results['VC33']}")
print(f"  CIFAR-2 acc:  {np.mean(cifar2_results):.4f}  {[f'{x:.3f}' for x in cifar2_results]}")
print(f"\nComparison:")
print(f"  LS20 895h cold (standalone, 25K, 10 seeds):  268.0/seed  0/10 zeros")
print(f"  FT09 895f cold (standalone, 25K, 10 seeds):    0.0/seed  10/10 zeros")
print(f"  LS20 chain (post-CIFAR, 5 seeds):   {np.mean(all_results['LS20']):.1f}/seed")
print(f"  FT09 chain (post-LS20+CIFAR, 5 seeds): {np.mean(all_results['FT09']):.1f}/seed")
print(f"  VC33 chain (post-FT09, 5 seeds):    {np.mean(all_results['VC33']):.1f}/seed  (first VC33 measurement post-ban)")
cifar_delta = np.mean(cifar2_results) - np.mean(cifar1_results)
print(f"  CIFAR transfer: CIFAR-2 - CIFAR-1 = {cifar_delta:+.4f}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 914 DONE")
