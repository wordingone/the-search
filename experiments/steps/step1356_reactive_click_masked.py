"""
Step 1356 — Reactive spatial exploration: click near observed changes.
Leo mail 3864, 2026-03-30. Queue after 1355.

1355 tests position LEARNING (info-gain trained position_head).
1356 tests position FOLLOWING — no training, pure reactive.

Hypothesis: 250 random clicks over 4096 positions is too sparse to LEARN.
But the substrate doesn't need to learn — it needs to REACT.
When a click produces a visible change, click NEAR that spot again.
Follow the change. Like human exploration: click, see what happens, click nearby.

Architecture:
  change_map = EMA(|obs_t - obs_{t-1}|, per pixel, alpha=0.1)  shape: (64, 64)
  click_probs = softmax(change_map.flatten() / T=1.0)
  click_pos = multinomial(click_probs)

  Early steps: change_map ≈ 0 → softmax ≈ uniform → random clicks.
  After click on interactive object: change_map concentrates → next click nearby.
  No training. No gradient. No learned head.

For KB games (n_actions=7): no change map for position, same as HIER.
type_head: UNTRAINED (random init, ~12.5% click rate — no type suppression).

Conditions:
  REACT: HIER + reactive change_map position selection. Seeds 13500-13504.
  HIER (1352 baseline): Random position selection. Same seeds.

Kill criteria:
  REACT Game B RHAE > 0 → LANDMARK. Reactive exploration reaches click game.
  REACT chain RHAE > HIER → reactive helps overall. SIGNAL.
  REACT = HIER → frame differencing doesn't concentrate clicks usefully. KILL.
"""
import sys, os, time, json, logging, hashlib, math
from collections import deque

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_masked import (
    select_games, seal_mapping, label_filename,
    masked_game_list,
    compute_progress_speedup, format_speedup,
    speedup_for_chain, compute_rhae_try2,
    ARC_OPTIMAL_STEPS_PROXY,
)

STEP        = 1356
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

N_DRAWS     = 5
DRAW_SEEDS  = [13500 + i for i in range(N_DRAWS)]

RESULTS_DIR       = os.path.join('B:/M/the-search/experiments/compositions', 'results_1356')
HIER_BASELINE_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1352')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]

ARC_N_ACTIONS   = 4103
N_TYPES         = 8
N_KB_ACTIONS    = 7
N_CLICK_POS     = 4096   # 64 × 64

ENT_CHECKPOINTS   = [100, 500, 1000, 2000]
MAX_TYPE_ENTROPY  = round(math.log(N_TYPES), 4)
MAX_POS_ENTROPY   = round(math.log(N_CLICK_POS), 4)  # ~8.3178
CHANGE_MAP_EMA    = 0.1   # EMA alpha for per-pixel change
CLICK_TEMP        = 1.0   # softmax temperature for change_map → click probs


# ---------------------------------------------------------------------------
# Load HIER baseline
# ---------------------------------------------------------------------------

def load_baseline(results_dir, step_num, n_draws=5):
    rhae_list = []
    for d in range(n_draws):
        draw_dir = os.path.join(results_dir, f'draw{d}')
        game_eff = []
        for label in ['MBPP', 'Game A', 'Game B']:
            fn = os.path.join(draw_dir, label_filename(label, step_num))
            if os.path.exists(fn):
                with open(fn) as f:
                    row = json.loads(f.readline())
                game_eff.append(row.get('eff_sq', 0.0))
        rhae = sum(game_eff) / 3 if game_eff else 0.0
        rhae_list.append(round(rhae, 7))
    return rhae_list


# ---------------------------------------------------------------------------
# Obs encoding
# ---------------------------------------------------------------------------

def _is_arc_obs(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


def _obs_to_one_hot(obs_arr):
    frame = np.round(obs_arr).astype(np.int32).squeeze(0)
    frame = np.clip(frame, 0, 15)
    one_hot = np.zeros((16, 64, 64), dtype=np.bool_)
    for c in range(16):
        one_hot[c] = (frame == c)
    return one_hot


def _encode_obs_mlp(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    if _is_arc_obs(arr):
        return _obs_to_one_hot(arr).astype(np.float32).flatten()
    return arr.flatten()


def _action_type_vec(action, n_types=6):
    vec = np.zeros(6, dtype=np.float32)
    vec[int(action) % n_types] = 1.0
    return vec


def _entropy(probs):
    eps = 1e-10
    return float(-np.sum(probs * np.log(probs + eps)))


def _change_map_to_probs(change_map, temperature=CLICK_TEMP):
    """Convert per-pixel change EMA to click probability distribution."""
    flat = change_map.flatten().astype(np.float64)
    logits = flat / temperature
    logits -= logits.max()   # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()
    return probs


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------

class MlpModel(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=_HIDDEN):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden)
        self.fc1         = nn.Linear(hidden, hidden)
        self.fc2         = nn.Linear(hidden, hidden)
        self.fc3         = nn.Linear(hidden, hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        self.pred_head   = nn.Linear(hidden + 6, hidden)
        self.dropout     = nn.Dropout(0.2)

    def forward(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2))
        return self.action_head(h3), h3

    def forward_all_layers(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3, h2, h1, h0


# ---------------------------------------------------------------------------
# Reactive hierarchical substrate
# ---------------------------------------------------------------------------

class ReactiveHierSubstrate:
    """
    MLP+TP with hierarchical action selection.
    type_head: UNTRAINED (random init, ~12.5% click rate).
    Position selection for clicks: reactive change_map (EMA of |obs_t - obs_{t-1}|).
    No trained position_head. Pure observation → reaction.
    """

    def __init__(self, n_actions):
        self.n_actions    = n_actions
        self._is_hier     = (n_actions == ARC_N_ACTIONS)
        self._rng         = np.random.RandomState(42)
        self._model       = None
        self._type_head   = None
        self._g           = None
        self._opt_pred    = None
        self._opt_g       = None
        self._opt_f       = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._prev_obs_raw  = None   # raw obs for change_map computation
        self._train_counter = 0
        self._step          = 0
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

        # Action selection stats
        self._type_counts = np.zeros(N_TYPES, dtype=np.int64)
        self._click_count = 0

        # Reactive change map: per-pixel EMA of |obs_t - obs_{t-1}|, shape (64, 64)
        self._change_map  = None   # initialized on first obs transition

        # Type entropy and change_map entropy at checkpoints
        self._type_entropy_at    = {ck: None for ck in ENT_CHECKPOINTS}
        self._change_ent_at      = {ck: None for ck in ENT_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._model = MlpModel(input_dim, self.n_actions).to(_DEVICE)
        if self._is_hier:
            self._type_head = nn.Linear(_HIDDEN, N_TYPES).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g = {k: torch.optim.Adam(v.parameters(), lr=_LR) for k, v in self._g.items()}
        self._opt_f = {
            'f_proj': torch.optim.Adam(self._model.input_proj.parameters(), lr=_LR),
            'f1':     torch.optim.Adam(self._model.fc1.parameters(), lr=_LR),
            'f2':     torch.optim.Adam(self._model.fc2.parameters(), lr=_LR),
            'f3':     torch.optim.Adam(self._model.fc3.parameters(), lr=_LR),
        }

    def get_try_stats(self):
        if not self._is_hier or self._step == 0:
            return None, {ck: None for ck in ENT_CHECKPOINTS}, {ck: None for ck in ENT_CHECKPOINTS}
        total = self._type_counts.sum()
        cf = round(float(self._type_counts[N_TYPES - 1]) / total, 4) if total > 0 else 0.0
        return cf, dict(self._type_entropy_at), dict(self._change_ent_at)

    def get_change_map_stats(self):
        """Return change_map diagnostics: n_nonzero pixels and max change value."""
        if self._change_map is None:
            return {'n_nonzero': 0, 'max_change': 0.0}
        nonzero = int((self._change_map > 1e-6).sum())
        max_val = round(float(self._change_map.max()), 4)
        return {'n_nonzero': nonzero, 'max_change': max_val}

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at   = {ck: None for ck in LOSS_CHECKPOINTS}
        self._type_counts    = np.zeros(N_TYPES, dtype=np.int64)
        self._click_count    = 0
        self._type_entropy_at = {ck: None for ck in ENT_CHECKPOINTS}
        self._change_ent_at  = {ck: None for ck in ENT_CHECKPOINTS}
        self._prev_obs_raw   = None
        # NOTE: change_map persists across tries (accumulated spatial experience)

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc
        self._prev_obs_raw = np.asarray(obs_arr, dtype=np.float32).copy()
        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            _, h3 = self._model(tensor)

            if self._is_hier:
                # Type selection: UNTRAINED random type_head (~12.5% click rate)
                type_logits = self._type_head(h3).squeeze(0).cpu()
                type_probs  = torch.softmax(type_logits, dim=-1).numpy()
                action_type = int(np.random.choice(N_TYPES, p=type_probs))
                self._type_counts[action_type] += 1

                # Record type entropy at checkpoint steps
                if self._step in self._type_entropy_at and self._type_entropy_at[self._step] is None:
                    self._type_entropy_at[self._step] = round(_entropy(type_probs), 4)

                # Record change_map entropy at checkpoint steps
                if self._step in self._change_ent_at and self._change_ent_at[self._step] is None:
                    if self._change_map is not None:
                        cm_probs = _change_map_to_probs(self._change_map)
                        self._change_ent_at[self._step] = round(_entropy(cm_probs), 4)
                    else:
                        self._change_ent_at[self._step] = MAX_POS_ENTROPY

                if action_type == N_TYPES - 1:  # click
                    if self._change_map is not None:
                        # Reactive: sample click position from change_map distribution
                        cm_probs = _change_map_to_probs(self._change_map)
                        pos = int(np.random.choice(N_CLICK_POS, p=cm_probs))
                    else:
                        # No change observed yet: uniform random
                        pos = int(self._rng.randint(0, N_CLICK_POS))
                    action = N_KB_ACTIONS + pos
                    self._click_count += 1
                else:
                    action = action_type
            else:
                logits, _ = self._model(tensor)
                probs = torch.softmax(logits.squeeze(0).cpu(), dim=-1).numpy()
                action = int(np.random.choice(self.n_actions, p=probs))

        return action % self.n_actions

    def _update_change_map(self, obs_next_arr):
        """Update per-pixel change EMA from raw obs transition."""
        if self._prev_obs_raw is None:
            return
        obs_t = self._prev_obs_raw
        obs_n = np.asarray(obs_next_arr, dtype=np.float32)
        diff = np.abs(obs_n - obs_t)  # (C, 64, 64) or (64, 64)
        if diff.ndim == 3:
            change_2d = diff.sum(axis=0)  # sum over channels → (64, 64)
        else:
            change_2d = diff
        if self._change_map is None:
            self._change_map = change_2d.astype(np.float32)
        else:
            self._change_map = ((1 - CHANGE_MAP_EMA) * self._change_map +
                                CHANGE_MAP_EMA * change_2d)

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)

        # Update change_map from raw obs transition
        if self._is_hier:
            self._update_change_map(obs_next)

        action_type_8 = int(action) if int(action) < N_KB_ACTIONS else (N_TYPES - 1)
        at_vec = _action_type_vec(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state': self._prev_enc.copy(),
                'action_type_vec': at_vec,
                'next_state': enc_next.copy(),
                'action_type_8': action_type_8,
            })
        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_obs_raw = None

    def _tp_train_step(self):
        n = len(self._buffer); buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]
        states = torch.from_numpy(np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)

        with torch.no_grad():
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_in = torch.cat([h3, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err = pred_next - target_h3_next
            pred_loss = float((pred_err ** 2).mean())

            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)

        with torch.enable_grad():
            pred_n2 = self._model.pred_head(torch.cat([h3.detach(), action_vecs.detach()], dim=1))
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()
        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                hr = self._g[gk](h_in.detach())
                lg = F.mse_loss(hr, h_out.detach())
                self._opt_g[gk].zero_grad(); lg.backward(); self._opt_g[gk].step()
        for fk, layer, x_in, target in [
            ('f3', self._model.fc3, h2, target_h3),
            ('f2', self._model.fc2, h1, target_h2),
            ('f1', self._model.fc1, h0, target_h1),
            ('f_proj', self._model.input_proj, states, target_h0),
        ]:
            with torch.enable_grad():
                hf = F.relu(layer(x_in.detach()))
                lf = F.mse_loss(hf, target.detach())
                self._opt_f[fk].zero_grad(); lf.backward(); self._opt_f[fk].step()

        return pred_loss

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None


# ---------------------------------------------------------------------------
# Game factory
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


def get_optimal_steps(game_name, seed):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        problem_idx = int(seed) % mbpp_game.N_EVAL_PROBLEMS
        solver = mbpp_game.compute_solver_steps(problem_idx)
        return solver.get(1)
    return ARC_OPTIMAL_STEPS_PROXY


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs            = env.reset(seed=seed)
    steps          = 0
    level          = 0
    steps_to_first_progress = None
    t_start        = time.time()
    fresh_episode  = True

    while steps < max_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action  = substrate.process(obs_arr) % n_actions
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
            if steps_to_first_progress is None:
                steps_to_first_progress = steps
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
    return steps_to_first_progress, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Single draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    print(f"  Draw {draw_idx} (seed={draw_seed}): {masked_game_list(game_labels)}")
    draw_results = []
    try2_progress = {}
    optimal_steps_d = {}

    for game_name, label in zip(games, game_labels.values()):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = 4103

        substrate = ReactiveHierSubstrate(n_actions=n_actions)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0, max_steps=TRY1_STEPS)
        try1_click_frac, try1_type_ent, try1_change_ent = substrate.get_try_stats()
        change_stats = substrate.get_change_map_stats() if substrate._is_hier else {}

        substrate.reset_for_try2()
        p2, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=TRY2_STEPS)
        try2_click_frac, try2_type_ent, try2_change_ent = substrate.get_try_stats()

        speedup = compute_progress_speedup(p1, p2)
        opt = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff = min(1.0, opt / p2)
            eff_sq = round(eff ** 2, 6)

        cr = substrate.get_compression_ratio()
        hier_str = f"click={try2_click_frac}" if substrate._is_hier else "flat"

        try2_progress[label]   = p2
        optimal_steps_d[label] = opt

        result = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt, 'cr': cr,
            'is_hier': substrate._is_hier,
            'try1_click_frac': try1_click_frac,
            'try2_click_frac': try2_click_frac,
            'try1_type_entropy_at_100':   try1_type_ent.get(100),
            'try1_type_entropy_at_500':   try1_type_ent.get(500),
            'try1_type_entropy_at_1000':  try1_type_ent.get(1000),
            'try1_type_entropy_at_2000':  try1_type_ent.get(2000),
            'try2_type_entropy_at_100':   try2_type_ent.get(100),
            'try2_type_entropy_at_500':   try2_type_ent.get(500),
            'try2_type_entropy_at_1000':  try2_type_ent.get(1000),
            'try2_type_entropy_at_2000':  try2_type_ent.get(2000),
            'try2_change_ent_at_100':     try2_change_ent.get(100),
            'try2_change_ent_at_500':     try2_change_ent.get(500),
            'try2_change_ent_at_1000':    try2_change_ent.get(1000),
            'try2_change_ent_at_2000':    try2_change_ent.get(2000),
            'max_type_entropy': MAX_TYPE_ENTROPY,
            'max_pos_entropy':  MAX_POS_ENTROPY,
            'change_map_stats': change_stats,
            't1': t1, 't2': t2,
        }
        draw_results.append(result)

        print(f"    {label}: speedup={format_speedup(speedup)}  eff²={eff_sq}  cr={cr}  {hier_str}  ({t1+t2:.1f}s)")
        if substrate._is_hier:
            te_traj  = [try2_type_ent.get(ck) for ck in ENT_CHECKPOINTS]
            ce_traj  = [try2_change_ent.get(ck) for ck in ENT_CHECKPOINTS]
            print(f"      type_entropy(try2)   at {ENT_CHECKPOINTS}: {te_traj}")
            print(f"      change_map_ent(try2) at {ENT_CHECKPOINTS}: {ce_traj}  (max={MAX_POS_ENTROPY:.4f})")
            click_shift = None
            if try1_click_frac is not None and try2_click_frac is not None:
                click_shift = round(try2_click_frac - try1_click_frac, 4)
            print(f"      click_frac: try1={try1_click_frac} try2={try2_click_frac} shift={click_shift}")
            if change_stats:
                print(f"      change_map: n_nonzero={change_stats['n_nonzero']}/4096  max={change_stats['max_change']}")

        out_path = os.path.join(draw_dir, label_filename(label, STEP))
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    chain_sp = sum(speedup_for_chain(r['speedup']) for r in draw_results) / len(draw_results)
    print(f"    → RHAE(try2) = {rhae:.6f}  chain_speedup={chain_sp:.4f}")
    print()
    return draw_results, rhae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    hier_rhae = load_baseline(HIER_BASELINE_DIR, 1352, 5)
    hier_mean = sum(hier_rhae) / len(hier_rhae)
    hier_nz   = sum(1 for r in hier_rhae if r > 0)

    print(f"Step {STEP} — Reactive spatial exploration (change_map position selection).")
    print(f"Device: {_DEVICE}")
    print(f"Protocol: {N_DRAWS} draws × 3 games × 2 tries = {N_DRAWS*3*2} episodes")
    print(f"REACT: click_pos ~ softmax(change_map / T={CLICK_TEMP}), T={CLICK_TEMP}, EMA_alpha={CHANGE_MAP_EMA}")
    print(f"type_head: UNTRAINED (random). Max type entropy: {MAX_TYPE_ENTROPY}")
    print(f"Max pos entropy (change_map): {MAX_POS_ENTROPY:.4f}")
    print(f"Draw seeds: {DRAW_SEEDS}")
    print(f"HIER baseline (1352 d0-4): {hier_rhae}  mean={hier_mean:.7f}  nz={hier_nz}/5")
    print()

    # Tier 1: timing
    games0, labels0 = select_games(seed=DRAW_SEEDS[0])
    first_arc = next((g for g in games0 if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = labels0[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, draw 0)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = ReactiveHierSubstrate(n_actions=na_t)
        t0 = time.time()
        obs_t = env_t.reset(seed=0)
        for _ in range(100):
            a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
            obs_t, _, done_t, _ = env_t.step(a)
            if done_t or obs_t is None:
                obs_t = env_t.reset(seed=0)
        elapsed_100 = time.time() - t0
        est_ep  = elapsed_100 / 100 * 2000
        est_tot = est_ep * N_DRAWS * 3 * 2
        print(f"  100 steps: {elapsed_100:.1f}s → est per episode: {est_ep:.0f}s → est total: {est_tot:.0f}s")
        print()

    all_results   = []
    rhae_per_draw = []

    for draw_idx in range(N_DRAWS):
        draw_results, draw_rhae = run_draw(draw_idx, DRAW_SEEDS[draw_idx])
        all_results.extend(draw_results)
        rhae_per_draw.append(draw_rhae)

    chain_mean    = sum(rhae_per_draw) / len(rhae_per_draw)
    nonzero_draws = sum(1 for r in rhae_per_draw if r is not None and r > 0)
    all_eff_sq    = [r['eff_sq'] for r in all_results]

    hier_results = [r for r in all_results if r['is_hier']]

    # Type entropy trajectory
    ent_traj_mean = {}
    for ck in ENT_CHECKPOINTS:
        vals = [r[f'try2_type_entropy_at_{ck}'] for r in hier_results if r[f'try2_type_entropy_at_{ck}'] is not None]
        ent_traj_mean[ck] = round(sum(vals) / len(vals), 4) if vals else None

    # Change_map entropy trajectory
    cm_ent_traj_mean = {}
    for ck in ENT_CHECKPOINTS:
        vals = [r[f'try2_change_ent_at_{ck}'] for r in hier_results if r[f'try2_change_ent_at_{ck}'] is not None]
        cm_ent_traj_mean[ck] = round(sum(vals) / len(vals), 4) if vals else None

    try2_cfs = [r['try2_click_frac'] for r in hier_results if r['try2_click_frac'] is not None]
    mean_click_frac = round(sum(try2_cfs) / len(try2_cfs), 4) if try2_cfs else None
    gameb_progress  = sum(1 for r in all_results if r['label'] == 'Game B' and r['eff_sq'] > 0)

    print("=" * 80)
    print(f"STEP {STEP} — RESULT (Reactive change_map click selection)")
    print()
    print(f"  REACT RHAE per draw: {[f'{r:.6f}' for r in rhae_per_draw]}")
    print(f"  Chain mean: {chain_mean:.7f}  Non-zero: {nonzero_draws}/{N_DRAWS}")
    print(f"  eff² max={max(all_eff_sq):.6f}")
    print()
    print(f"  Type entropy (try2 mean across ARC):")
    print(f"    At {ENT_CHECKPOINTS}: {[ent_traj_mean[ck] for ck in ENT_CHECKPOINTS]}")
    print()
    print(f"  Change_map entropy (try2 mean across ARC):")
    print(f"    At {ENT_CHECKPOINTS}: {[cm_ent_traj_mean[ck] for ck in ENT_CHECKPOINTS]}")
    ce_vals = [cm_ent_traj_mean[ck] for ck in ENT_CHECKPOINTS if cm_ent_traj_mean[ck] is not None]
    ce_drop = round(ce_vals[0] - ce_vals[-1], 4) if len(ce_vals) >= 2 else 0
    print(f"    Drop CE100→CE2000: {ce_drop}  (max={MAX_POS_ENTROPY:.4f})")
    print(f"  Mean click_frac: {mean_click_frac}  (1352 HIER: 0.1232 — should stay ~12.5%)")
    print()
    print("  Games with progress:")
    for r in all_results:
        if r['eff_sq'] > 0:
            hier_str = f"click={r['try2_click_frac']}" if r['is_hier'] else "flat"
            print(f"    Draw {r['draw']} {r['label']} ({r['game']}): eff²={r['eff_sq']}  p2={r['p2']}  {hier_str}")
            if r.get('change_map_stats', {}).get('n_nonzero', 0) > 0:
                print(f"      change_map: n_nonzero={r['change_map_stats']['n_nonzero']}  max={r['change_map_stats']['max_change']}")
    print()

    print(f"  Baselines:")
    print(f"    HIER (1352 d0-4): {hier_rhae}  mean={hier_mean:.7f}  nz={hier_nz}/5")
    print()

    print("ASSESSMENT:")
    if gameb_progress > 0:
        print(f"  >>> LANDMARK: Game B (click) progress! {gameb_progress} episodes.")
    if ce_drop > 0.5:
        print(f"  >>> change_map entropy drops {ce_drop} — reactive focusing IS concentrating clicks.")
    else:
        print(f"  >>> change_map entropy flat (drop={ce_drop}). Little spatial concentration.")
    if mean_click_frac is not None:
        if abs(mean_click_frac - 0.125) < 0.03:
            print(f"  >>> click_frac={mean_click_frac} stable (~12.5%). No type bias.")
        else:
            print(f"  >>> click_frac={mean_click_frac} (expected ~12.5%). Check type_head.")
    if hier_mean > 0 and chain_mean > hier_mean:
        ratio = chain_mean / hier_mean
        print(f"  >>> REACT RHAE={chain_mean:.7f} > HIER {hier_mean:.7f} ({ratio:.1f}×). SIGNAL.")
    elif chain_mean == 0:
        print(f"  >>> KILL: REACT RHAE=0. Reactive position selection ineffective.")
    else:
        print(f"  >>> REACT RHAE={chain_mean:.7f} vs HIER {hier_mean:.7f}.")
    print("=" * 80)

    summary = {
        'step': STEP, 'n_draws': N_DRAWS, 'draw_seeds': DRAW_SEEDS,
        'rhae_per_draw': rhae_per_draw, 'chain_mean_rhae': chain_mean,
        'nonzero_draws': nonzero_draws, 'gameb_progress': gameb_progress,
        'mean_click_frac': mean_click_frac,
        'type_entropy_traj_mean':     {str(k): v for k, v in ent_traj_mean.items()},
        'change_map_ent_traj_mean':   {str(k): v for k, v in cm_ent_traj_mean.items()},
        'max_type_entropy': MAX_TYPE_ENTROPY,
        'max_pos_entropy':  MAX_POS_ENTROPY,
        'hier_baseline_rhae': hier_rhae, 'hier_baseline_mean': hier_mean,
    }
    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == '__main__':
    main()
