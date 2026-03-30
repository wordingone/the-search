"""
Step 1324 — CNN SELFSUP-ENT probe: all 10 ARC games + MBPP.
Leo mail 3747, 2026-03-29.

## Question

Step 1323 showed speedup=inf on 1 game (n=1). Ambiguous. Run all 10 solved
ARC games + MBPP to resolve: does CNN show speedup on 3+ games?

If speedup > 1 on 3+ games → CNN learns from experience (not noise).
If speedup inf/null on most → n=1 was noise.

## Architecture

Same CNN SELFSUP-ENT as step 1323 (1305/1306 template):
- 4 conv layers (16-channel input → 32 → 64 → 128 → 256)
- Forward prediction head (avg_features + action_type_vec → avg_features)
- Entropy-driven action selection (sigmoid probs, multinomial sampling)
- Adam optimizer, self-supervised prediction loss + entropy bonus
- MBPP: random fallback (CNN obs format incompatible with text)

## Constitutional audit
R2 VIOLATION: Adam is external optimizer. THIS IS A PROBE, NOT A CANDIDATE.
Explicit acknowledgement: testing capability ceiling, not R2 compliance.

## Protocol changes from spec
- 1K steps/try instead of 5K: runtime cap compliance (Jun directive, 5 min max).
  Eli notified Leo before running (mail 3749). Leo's 5K at ~40s/episode × 22
  episodes = ~15 min >> cap. At 1K: ~12-13s/episode × 22 = ~290s ≈ 4.8 min.
- MAX_SECONDS=12 per episode (hard cap).
- LOSS_CHECKPOINTS: [250, 500, 1000] (adjusted for 1K steps).
- Pred_loss tracking bug fix (from 1323): step counter reset between tries so
  try2 checkpoints correctly capture within-try2 steps.

## All 10 ARC games + MBPP (no select_games — full sweep, no selection bias)
Games: ft09, ls20, vc33, tr87, sp80, sb26, tu93, cn04, cd82, lp85 + MBPP

## Kill criteria
- speedup ≤ 1 on all games where L1 reached → CNN doesn't learn from experience
- speedup > 1 on 3+ games → CNN learns → find R2-compliant alternative
"""
import sys, os, time, json, logging, hashlib

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_masked import seal_mapping, label_filename, masked_run_log, format_speedup, write_experiment_results

STEP = 1324
MAX_STEPS   = 1_000   # per try — reduced from spec's 5K for 5-min cap compliance
MAX_SECONDS = 12      # per episode (hard cap)

CONDITIONS = ['cnn_ent']
LABELS_DISPLAY = {'cnn_ent': 'CNN-ENT'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1324')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

# All 10 solved ARC games (full sweep — no random selection)
ARC_GAMES = ['cd82', 'cn04', 'ft09', 'lp85', 'ls20', 'sb26', 'sp80', 'tr87', 'tu93', 'vc33']
ALL_GAMES  = ['mbpp'] + ARC_GAMES

# Deterministic labels: MBPP unmasked; ARC games labeled 01-10 alphabetically
# (alphabetical = no ordering reveals game identity)
GAME_LABELS = {'mbpp': 'MBPP'}
for _i, _g in enumerate(ARC_GAMES, 1):
    GAME_LABELS[_g] = f'Game {_i:02d}'

SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'vc33':  ('vc33_fullchain.json',  'all_actions_encoded'),
    'tr87':  ('tr87_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'all_actions'),
    'sb26':  ('sb26_fullchain.json',  'all_actions'),
    'tu93':  ('tu93_fullchain.json',  'all_actions'),
    'cn04':  ('cn04_fullchain.json',  'sequence'),
    'cd82':  ('cd82_fullchain.json',  'all_actions'),
    'lp85':  ('lp85_fullchain.json',  'all_actions'),
}
ACTION_OFFSET = {'ls20': -1, 'vc33': 7}

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN hyperparameters (same as step 1306/1323)
_BUFFER_MAXLEN = 200_000
_TRAIN_FREQ    = 5
_BATCH_SIZE    = 64
_LR            = 0.0001
_ACTION_ENTROPY_COEF = 0.0001
_COORD_ENTROPY_COEF  = 0.00001
_SG_OUTPUT_DIM = 4101
N_KEYBOARD     = 7

LOSS_CHECKPOINTS = [250, 500, 1000]   # adjusted for 1K steps

SEED_A = 0
SEED_B = 1


# ---------------------------------------------------------------------------
# CNN model (same as step 1306/1323)
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.action_pool = nn.MaxPool2d(4, 4)
        action_flattened_size = 256 * 16 * 16
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)
        self.pred_head = nn.Linear(256 + 6, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))
        avg_features = conv_features.mean(dim=[2, 3])

        af = self.action_pool(conv_features)
        af = af.view(af.size(0), -1)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)

        cf = F.relu(self.coord_conv1(conv_features))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        cf = self.coord_conv4(cf)
        coord_logits = cf.view(cf.size(0), -1)

        combined = torch.cat([action_logits, coord_logits], dim=1)
        return combined, avg_features

    def predict_next(self, avg_features, action_type_vec):
        inp = torch.cat([avg_features, action_type_vec], dim=1)
        return self.pred_head(inp)


def _sg_to_action_type_vec(sg_idx):
    vec = np.zeros(6, dtype=np.float32)
    if sg_idx < 5:
        vec[sg_idx] = 1.0
    else:
        vec[5] = 1.0
    return vec


def _obs_to_one_hot(obs_arr):
    frame = np.round(obs_arr).astype(np.int32).squeeze(0)
    frame = np.clip(frame, 0, 15)
    one_hot = np.zeros((16, 64, 64), dtype=np.bool_)
    for c in range(16):
        one_hot[c] = (frame == c)
    return one_hot


def _is_arc_obs(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


# ---------------------------------------------------------------------------
# CNN SELFSUP-ENT substrate
# ---------------------------------------------------------------------------

class CnnEntSubstrate:
    """CNN with entropy-driven action selection. Same as step 1306 SELFSUP-ENT.

    R2-VIOLATING PROBE. Adam optimizer. Explicit flag.
    MBPP fallback: random actions (CNN format incompatible with text obs).
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(42)   # fixed internal RNG for buffer sampling
        self._use_random = not (n_actions > 128)  # True for MBPP (n_actions=128)

        self._model = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=_LR)
        self._buffer = deque(maxlen=_BUFFER_MAXLEN)
        self._buffer_hashes = set()

        self._prev_one_hot = None
        self._prev_sg_idx = None
        self._train_counter = 0
        self._step = 0          # RELATIVE step — reset via reset_loss_tracking()
        self._action_counts = np.zeros(n_actions, np.float32)

        # Episode-start snapshot: stored at creation, NEVER reset on level transitions
        self._episode_start_state_dict = {k: v.clone().cpu() for k, v in
                                           self._model.state_dict().items()}

        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def reset_loss_tracking(self):
        """Reset step counter and loss checkpoints for next try.
        Model weights and buffer are NOT reset — learning transfers.
        Called between try1 and try2.
        """
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def process(self, obs_arr):
        self._step += 1
        obs_arr = np.asarray(obs_arr, dtype=np.float32)

        if self._use_random or not _is_arc_obs(obs_arr):
            self._prev_one_hot = None
            return int(self._rng.randint(self.n_actions))

        one_hot = _obs_to_one_hot(obs_arr)
        self._prev_one_hot = one_hot

        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, avg_features = self._model(tensor)
            logits = logits.squeeze(0).cpu()

        # Entropy-driven selection
        action_logits = logits[:5]
        coord_logits = logits[5:]
        action_probs = torch.sigmoid(action_logits)
        coord_probs_scaled = torch.sigmoid(coord_logits) / 4096.0
        combined = torch.cat([action_probs, coord_probs_scaled])
        total = combined.sum().item()
        if total <= 1e-12:
            combined = torch.ones(_SG_OUTPUT_DIM) / _SG_OUTPUT_DIM
        else:
            combined = combined / total

        sg_idx = int(torch.multinomial(combined, 1).item())
        self._prev_sg_idx = sg_idx
        action = self._sg_to_prism(sg_idx)
        self._action_counts[action % self.n_actions] += 1
        return action

    def update_after_step(self, obs_next, action, reward_env):
        if self._use_random or self._prev_one_hot is None:
            return
        obs_next_arr = np.asarray(obs_next, dtype=np.float32)
        if not _is_arc_obs(obs_next_arr):
            return

        one_hot_next = _obs_to_one_hot(obs_next_arr)
        sg_idx = self._prev_sg_idx if self._prev_sg_idx is not None else (action % 5)
        action_type_vec = _sg_to_action_type_vec(sg_idx)

        self._add_to_buffer(self._prev_one_hot, sg_idx, action_type_vec, one_hot_next)

        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            self._train_step()

    def on_level_transition(self):
        self._prev_one_hot = None
        self._prev_sg_idx = None

    def compute_weight_drift(self):
        drift = 0.0
        for name, param in self._model.named_parameters():
            init_val = self._episode_start_state_dict[name]
            drift += (param.data.cpu() - init_val.cpu()).norm().item()
        return float(drift)

    def get_compression_ratio(self):
        """Compression ratio: pred_loss at step 1000 / pred_loss at step 250."""
        l_early = self._pred_loss_at.get(250)
        l_late  = self._pred_loss_at.get(1000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def _sg_to_prism(self, sg_idx):
        if sg_idx < 5:
            return sg_idx
        click_idx = sg_idx - 5
        prism_action = N_KEYBOARD + click_idx
        if prism_action < self.n_actions:
            return prism_action
        return sg_idx % 5

    def _add_to_buffer(self, one_hot, sg_idx, action_type_vec, one_hot_next):
        state_bytes = one_hot.tobytes()
        action_bytes = np.array([sg_idx], dtype=np.int32).tobytes()
        h = hashlib.md5(state_bytes + action_bytes).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state': one_hot.copy(),
                'action_type_vec': action_type_vec.copy(),
                'next_state': one_hot_next.copy(),
            })

    def _train_step(self):
        n = len(self._buffer)
        buf_list = list(self._buffer)
        indices = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf_list[i] for i in indices]

        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])
        ).to(_DEVICE)
        next_states = torch.from_numpy(
            np.stack([b['next_state'].astype(np.float32) for b in batch])
        ).to(_DEVICE)
        action_vecs = torch.from_numpy(
            np.stack([b['action_type_vec'] for b in batch])
        ).to(_DEVICE)

        logits, avg_features_t = self._model(states)
        with torch.no_grad():
            _, avg_features_next = self._model(next_states)

        predicted_next = self._model.predict_next(avg_features_t, action_vecs)
        pred_loss = F.mse_loss(predicted_next, avg_features_next)

        al = logits[:, :5]
        cl = logits[:, 5:]
        ap = torch.sigmoid(al)
        cp = torch.sigmoid(cl)
        action_entropy = -(ap * (ap + 1e-8).log()).sum(dim=1).mean()
        coord_entropy = -(cp * (cp + 1e-8).log()).sum(dim=1).mean()

        total_loss = pred_loss - _ACTION_ENTROPY_COEF * action_entropy - _COORD_ENTROPY_COEF * coord_entropy

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        loss_val = float(pred_loss.item())
        self._recent_losses.append(loss_val)
        for ck in LOSS_CHECKPOINTS:
            if self._pred_loss_at[ck] is None and self._step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(list(self._recent_losses))), 6)


# ---------------------------------------------------------------------------
# Game factory and solvers
# ---------------------------------------------------------------------------

def make_game(game_name):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.make(gn)
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def load_prescription(game_name):
    if game_name.lower() not in SOLVER_PRESCRIPTIONS:
        return None
    fname, field = SOLVER_PRESCRIPTIONS[game_name.lower()]
    path = os.path.join(PDIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(field)


def compute_solver_level_steps(game_name):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.compute_solver_steps(0)
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    env.reset(seed=1)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    offset = ACTION_OFFSET.get(game_name.lower(), 0)
    level, level_first_step, step, fresh_episode = 0, {}, 0, True
    for action in prescription:
        action_int = (int(action) + offset) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        step += 1
        if fresh_episode:
            fresh_episode = False
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step
            level = cl
        if done:
            env.reset(seed=1)
            fresh_episode = True
    return level_first_step


def compute_arc_score(level_first_step, solver_level_steps):
    if not level_first_step or not solver_level_steps:
        return 0.0
    scores = []
    for lvl, s_step in solver_level_steps.items():
        a_step = level_first_step.get(lvl)
        if a_step is not None and s_step > 0:
            scores.append((s_step / a_step) ** 2)
    if not scores:
        return 0.0
    return round(float(np.mean(scores)), 6)


def compute_action_kl(action_log, n_actions):
    if len(action_log) < 400:
        return None
    early_c = np.zeros(n_actions, np.float32)
    late_c = np.zeros(n_actions, np.float32)
    for a in action_log[:200]:
        if 0 <= a < n_actions:
            early_c[a] += 1
    for a in action_log[-200:]:
        if 0 <= a < n_actions:
            late_c[a] += 1
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p  = (late_c + 1e-8) / (late_c.sum() + 1e-8 * n_actions)
    return round(float(np.sum(early_p * np.log(early_p / late_p + 1e-12))), 4)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, solver_level_steps, seed):
    obs = env.reset(seed=seed)
    action_log = []
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

    steps = 0
    level = 0
    max_level = 0
    level_first_step = {}
    t_start = time.time()
    fresh_episode = True

    while steps < MAX_STEPS:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()

        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

        if obs_next is not None:
            substrate.update_after_step(obs_next, action, reward)

        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl not in level_first_step:
                level_first_step[cl] = steps
            if cl > max_level:
                max_level = cl
            level = cl
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
        else:
            obs = obs_next

    elapsed = time.time() - t_start

    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    action_kl = compute_action_kl(action_log, n_actions)

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'wdrift': round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj': substrate.get_pred_loss_trajectory(),
        'compression_ratio': substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner — try1 then try2 on SAME substrate (weights persist)
# ---------------------------------------------------------------------------

def run_game(game_name, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    substrate = CnnEntSubstrate(n_actions=n_actions)

    result_try1 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_A)

    # Reset loss tracking only (not weights) so try2 loss trajectory is meaningful
    substrate.reset_loss_tracking()

    result_try2 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_B)

    l1_try1 = result_try1['level_first_step'].get(1)
    l1_try2 = result_try2['level_first_step'].get(1)

    if l1_try1 is not None and l1_try2 is not None and l1_try2 > 0:
        speedup = round(l1_try1 / l1_try2, 4)
    elif l1_try1 is None and l1_try2 is not None:
        speedup = float('inf')
    elif l1_try1 is not None and l1_try2 is None:
        speedup = 0.0
    else:
        speedup = None

    return {
        'game': game_name,
        'condition': 'cnn_ent',
        'try1': result_try1,
        'try2': result_try2,
        'second_exposure_speedup': speedup,
        'compression_ratio': result_try1['compression_ratio'],
        'n_actions': n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — CNN SELFSUP-ENT probe expanded (R2-violating, explicit flag)")
    print(f"Games: all 10 ARC + MBPP ({len(ALL_GAMES)} total)")
    print(f"Device: {_DEVICE}")
    print(f"CNN SELFSUP-ENT: 4 conv layers, Adam, entropy-driven action, forward prediction")
    print(f"R2 VIOLATION FLAGGED — capability probe, not a candidate.")
    print(f"Primary metric: second_exposure_speedup per game, chain mean")
    print(f"{MAX_STEPS} steps/try, MAX_SECONDS={MAX_SECONDS}/episode. Estimated runtime: ~5 min.")
    print()

    seal_mapping(RESULTS_DIR, ALL_GAMES, GAME_LABELS)

    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in ALL_GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
        except Exception:
            solver_steps_cache[game] = {}
    print()

    all_results = []
    speedup_by_label = {}

    for game in ALL_GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        t0 = time.time()
        result = run_game(game, solver_steps)
        all_results.append(result)
        speedup_by_label[label] = result['second_exposure_speedup']
        elapsed = time.time() - t0
        print(masked_run_log(label, elapsed))

        out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
        with open(out_path, 'w') as f:
            masked_r = {k: v for k, v in result.items() if k != 'game'}
            masked_r['label'] = label
            f.write(json.dumps(masked_r, default=str) + '\n')

    # Chain aggregate: mean of finite speedup > 0 across all games
    finite_speedups = [v for v in speedup_by_label.values()
                       if v is not None and v != float('inf') and v > 0]
    inf_count  = sum(1 for v in speedup_by_label.values() if v == float('inf'))
    null_count = sum(1 for v in speedup_by_label.values() if v is None)
    zero_count = sum(1 for v in speedup_by_label.values() if v == 0.0)

    if finite_speedups:
        chain_speedup = round(float(np.mean(finite_speedups)), 4)
    elif inf_count > 0:
        chain_speedup = float('inf')
    else:
        chain_speedup = None

    speedup_by_condition = {'cnn_ent': chain_speedup}
    write_experiment_results(RESULTS_DIR, STEP, speedup_by_condition, all_results, CONDITIONS)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT (R2-violating probe, n=11)\n")
    print(f"  CNN-ENT chain speedup = {format_speedup(chain_speedup)}")
    print(f"  (finite>0: {len(finite_speedups)} games, inf: {inf_count}, null: {null_count}, zero: {zero_count})")
    print()
    print("  Per-game breakdown:")
    for label in sorted(speedup_by_label):
        sp = speedup_by_label[label]
        cr = next((r['compression_ratio'] for r in all_results
                   if GAME_LABELS.get(r['game']) == label), None)
        cr_str = f"{cr:.4f}" if cr is not None else "N/A"
        print(f"    {label}: speedup={format_speedup(sp)}  cr={cr_str}")
    print()

    print("KILL ASSESSMENT:")
    n_positive = len(finite_speedups) + inf_count
    if n_positive == 0:
        print("  >>> KILL: speedup ≤ 1 on all games where L1 reached")
        print("  >>> CNN does NOT learn from experience. Problem deeper than R2.")
    elif n_positive >= 3:
        print(f"  >>> SIGNAL: speedup > 1 on {n_positive} games (>= 3 threshold)")
        print("  >>> NO KILL — CNN learns from experience. Find R2-compliant alternative.")
    else:
        print(f"  >>> WEAK SIGNAL: speedup > 1 on {n_positive} game(s) (< 3 threshold)")
        print("  >>> Ambiguous. Chain speedup may be noise.")
    print(sep)
    print(f"\nResults: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
