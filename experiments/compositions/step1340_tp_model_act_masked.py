"""
Step 1340 — World model action selection: substrate uses TP predictions to choose actions.
Leo mail 3810, 2026-03-29.

Substrate has 92% prediction compression (MLP+TP from 1337). It acts randomly anyway.
This step makes it USE the model: for each step, sample K=32 candidate actions, predict
the next latent state for each, select the action with maximum predicted change (novelty).

Why novelty (predicted change in latent space):
- Actions that change state = interactive actions (pressing buttons, clicking objects)
- Actions that don't change state = null actions (clicking empty space)
- Novelty naturally selects interactive actions over null actions
- This is curiosity-driven deliberation: simulate before acting

Why this is different from Step 1306 (argmax on raw predicted delta, CNN):
- 1306: noisy CNN features, argmax on noise → concentration on noise
- 1340: MLP with 92% compression, argmax on meaningful prediction → signal-driven selection

Constitutional audit:
  R0: Deterministic init, seed-free ✓
  R1: Self-supervised. Action selected by internal prediction, no external reward ✓
  R2: Action selection IS the computation — forward model predicts, same comp selects ✓
  R3: As prediction improves, action selection changes ✓
  R4: Implicit in RHAE(try2) ✓
  R5: Game is ground truth ✓
  R6: Deletion test: remove model-based selection → entropy baseline ✓

NEW component: model-based action selection via predicted novelty.
Not in catalog C1-C33 — first test of deliberation in the search.

Conditions:
  model_act: MLP + TP + model-based action selection (novelty in latent space)
  mlp_tp:    MLP + TP + entropy (random) selection — same as 1337 control

Protocol: 2K steps per try, seed-free, level-masked. 1 draw × 3 games × 2 conditions.

Kill:     MODEL-ACT RHAE(try2) = 0, no progress → model-based doesn't help. KILL.
Signal:   Any try reaches progress → SIGNAL. Report action entropy diagnostic.
Landmark: MODEL-ACT RHAE(try2) > 0 → first non-zero RHAE. LANDMARK.
"""
import sys, os, time, json, logging, hashlib
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
    masked_game_list, masked_run_log,
    compute_progress_speedup, format_speedup,
    speedup_for_chain, compute_rhae_try2, write_experiment_results,
    ARC_OPTIMAL_STEPS_PROXY, get_arc_optimal_steps,
)

GAMES, GAME_LABELS = select_games(seed=1340)

STEP        = 1340
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280   # 4.7 min per game (5-min cap)

CONDITIONS  = ['model_act', 'mlp_tp']
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1340')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]

# Model-based action selection constants
_MODEL_ACT_K       = 32     # candidate actions sampled per step
_MODEL_ACT_WARMUP  = 64     # env steps before using model (buffer fill time)


# ---------------------------------------------------------------------------
# Obs encoding (same as 1337-1339)
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


# ---------------------------------------------------------------------------
# MLP model (same as 1337)
# ---------------------------------------------------------------------------

class MlpSelfSupModel(nn.Module):
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
        logits = self.action_head(h3)
        return logits, h3

    def forward_all_layers(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3, h2, h1, h0


# ---------------------------------------------------------------------------
# Model-based action selection substrate (MODEL-ACT)
# ---------------------------------------------------------------------------

class ModelActMlpTpSubstrate:
    """MLP + TP + model-based action selection via predicted novelty.

    Action selection:
      For each step after warmup:
        1. Sample K=32 random candidate actions from full action space
        2. For each candidate: predict next latent state h3 using pred_head
        3. Novelty = ||predicted_h3_next - current_h3||_2
        4. Select action with maximum novelty

    Novelty in latent space = predicted change in internal representation.
    Actions that change state (interactive) → higher novelty than null actions.

    During warmup (first 64 steps, buffer filling): random action selection.
    """

    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(42)
        self._model      = None
        self._g          = None
        self._opt_pred   = None
        self._opt_g      = None
        self._opt_f      = None
        self._input_dim  = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._prev_action   = None
        self._train_counter = 0
        self._step          = 0
        self._ep_start_sd   = None
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}
        self._action_log_all = []  # full action log for entropy

    def _init_model(self, input_dim):
        self._input_dim = input_dim
        self._model = MlpSelfSupModel(input_dim, self.n_actions).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g = {k: torch.optim.Adam(v.parameters(), lr=_LR)
                       for k, v in self._g.items()}
        self._opt_f = {
            'f_proj': torch.optim.Adam(self._model.input_proj.parameters(), lr=_LR),
            'f1':     torch.optim.Adam(self._model.fc1.parameters(), lr=_LR),
            'f2':     torch.optim.Adam(self._model.fc2.parameters(), lr=_LR),
            'f3':     torch.optim.Adam(self._model.fc3.parameters(), lr=_LR),
        }
        self._ep_start_sd = {k: v.clone().cpu()
                             for k, v in self._model.state_dict().items()}

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}
        self._action_log_all = []

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc

        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)

        # During warmup or before buffer trained: random selection
        if self._step <= _MODEL_ACT_WARMUP or len(self._buffer) < _BATCH_SIZE:
            action = int(self._rng.randint(self.n_actions))
            self._prev_action = action
            self._action_log_all.append(action)
            return action % self.n_actions

        # Model-based: select action by predicted latent novelty
        with torch.no_grad():
            h3, _, _, _ = self._model.forward_all_layers(tensor)  # (1, 512)

            # Sample K candidate actions
            k = min(_MODEL_ACT_K, self.n_actions)
            candidates = self._rng.choice(self.n_actions, k, replace=False)

            # Batch: predict next h3 for all candidates at once
            action_vecs = torch.from_numpy(
                np.stack([_action_type_vec(int(a)) for a in candidates])
            ).to(_DEVICE)  # (K, 6)

            h3_rep  = h3.expand(k, -1)          # (K, 512)
            pred_in = torch.cat([h3_rep, action_vecs], dim=1)   # (K, 518)
            pred_next = self._model.pred_head(pred_in)           # (K, 512)

            # Novelty = L2 change in latent space
            novelty = (pred_next - h3_rep).norm(dim=1)  # (K,)
            best_idx = int(novelty.argmax().item())
            action   = int(candidates[best_idx])

        self._prev_action = action
        self._action_log_all.append(action)
        return action % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)
        at_vec = _action_type_vec(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state':           self._prev_enc.copy(),
                'action_type_vec': at_vec,
                'next_state':      enc_next.copy(),
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
        self._prev_enc    = None
        self._prev_action = None

    def _tp_train_step(self):
        n   = len(self._buffer)
        buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]

        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(
            np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(
            np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)

        with torch.no_grad():
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_in   = torch.cat([h3, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err  = pred_next - target_h3_next
            pred_loss = float((pred_err ** 2).mean())

            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)

        with torch.enable_grad():
            pred_in2 = torch.cat([h3.detach(), action_vecs.detach()], dim=1)
            pred_n2  = self._model.pred_head(pred_in2)
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()

        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                h_recon = self._g[gk](h_in.detach())
                loss_g  = F.mse_loss(h_recon, h_out.detach())
                self._opt_g[gk].zero_grad(); loss_g.backward(); self._opt_g[gk].step()

        for fk, layer, x_in, target in [
            ('f3',     self._model.fc3,        h2,     target_h3),
            ('f2',     self._model.fc2,        h1,     target_h2),
            ('f1',     self._model.fc1,        h0,     target_h1),
            ('f_proj', self._model.input_proj, states, target_h0),
        ]:
            with torch.enable_grad():
                h_fresh = F.relu(layer(x_in.detach()))
                loss_local = F.mse_loss(h_fresh, target.detach())
                self._opt_f[fk].zero_grad(); loss_local.backward(); self._opt_f[fk].step()

        return pred_loss

    def compute_weight_drift(self):
        if self._model is None or self._ep_start_sd is None:
            return 0.0
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._ep_start_sd:
                drift += (param.data.cpu() - self._ep_start_sd[name]).norm().item()
        return float(drift)

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_action_entropy(self):
        """Compute entropy of action distribution (diagnostic for concentration)."""
        if not self._action_log_all:
            return None
        counts = np.zeros(self.n_actions, dtype=np.float32)
        for a in self._action_log_all:
            if 0 <= a < self.n_actions:
                counts[a] += 1
        total = counts.sum()
        if total < 1:
            return None
        probs = counts / total
        probs += 1e-10
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = float(np.log(self.n_actions))
        return round(entropy / max_entropy, 4)  # normalized 0-1


# ---------------------------------------------------------------------------
# MLP-TP control substrate (same as 1337, random action selection)
# ---------------------------------------------------------------------------

class MlpTpSubstrate:
    """MLP + TP control. Random action selection. Identical to step 1337."""

    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(42)
        self._model      = None
        self._g          = None
        self._opt_pred   = None
        self._opt_g      = None
        self._opt_f      = None
        self._input_dim  = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._prev_action   = None
        self._train_counter = 0
        self._step          = 0
        self._ep_start_sd   = None
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}
        self._action_log_all = []

    def _init_model(self, input_dim):
        self._input_dim = input_dim
        self._model = MlpSelfSupModel(input_dim, self.n_actions).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g = {k: torch.optim.Adam(v.parameters(), lr=_LR)
                       for k, v in self._g.items()}
        self._opt_f = {
            'f_proj': torch.optim.Adam(self._model.input_proj.parameters(), lr=_LR),
            'f1':     torch.optim.Adam(self._model.fc1.parameters(), lr=_LR),
            'f2':     torch.optim.Adam(self._model.fc2.parameters(), lr=_LR),
            'f3':     torch.optim.Adam(self._model.fc3.parameters(), lr=_LR),
        }
        self._ep_start_sd = {k: v.clone().cpu()
                             for k, v in self._model.state_dict().items()}

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}
        self._action_log_all = []

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc
        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = self._model(tensor)
            logits = logits.squeeze(0).cpu()
        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        self._prev_action = action
        self._action_log_all.append(action)
        return action % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)
        at_vec = _action_type_vec(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state':           self._prev_enc.copy(),
                'action_type_vec': at_vec,
                'next_state':      enc_next.copy(),
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
        self._prev_enc    = None
        self._prev_action = None

    def _tp_train_step(self):
        n   = len(self._buffer)
        buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]

        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(
            np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(
            np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)

        with torch.no_grad():
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_in   = torch.cat([h3, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err  = pred_next - target_h3_next
            pred_loss = float((pred_err ** 2).mean())

            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)

        with torch.enable_grad():
            pred_in2 = torch.cat([h3.detach(), action_vecs.detach()], dim=1)
            pred_n2  = self._model.pred_head(pred_in2)
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()

        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                h_recon = self._g[gk](h_in.detach())
                loss_g  = F.mse_loss(h_recon, h_out.detach())
                self._opt_g[gk].zero_grad(); loss_g.backward(); self._opt_g[gk].step()

        for fk, layer, x_in, target in [
            ('f3',     self._model.fc3,        h2,     target_h3),
            ('f2',     self._model.fc2,        h1,     target_h2),
            ('f1',     self._model.fc1,        h0,     target_h1),
            ('f_proj', self._model.input_proj, states, target_h0),
        ]:
            with torch.enable_grad():
                h_fresh = F.relu(layer(x_in.detach()))
                loss_local = F.mse_loss(h_fresh, target.detach())
                self._opt_f[fk].zero_grad(); loss_local.backward(); self._opt_f[fk].step()

        return pred_loss

    def compute_weight_drift(self):
        if self._model is None or self._ep_start_sd is None:
            return 0.0
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._ep_start_sd:
                drift += (param.data.cpu() - self._ep_start_sd[name]).norm().item()
        return float(drift)

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_action_entropy(self):
        if not self._action_log_all:
            return None
        counts = np.zeros(self.n_actions, dtype=np.float32)
        for a in self._action_log_all:
            if 0 <= a < self.n_actions:
                counts[a] += 1
        total = counts.sum()
        if total < 1:
            return None
        probs = counts / total
        probs += 1e-10
        entropy = -float(np.sum(probs * np.log(probs)))
        return round(entropy / float(np.log(self.n_actions)), 4)


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

def compute_action_kl(action_log, n_actions):
    if len(action_log) < 400:
        return None
    early_c = np.zeros(n_actions, np.float32)
    late_c  = np.zeros(n_actions, np.float32)
    for a in action_log[:200]:
        if 0 <= a < n_actions: early_c[a] += 1
    for a in action_log[-200:]:
        if 0 <= a < n_actions: late_c[a] += 1
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p  = (late_c  + 1e-8) / (late_c.sum()  + 1e-8 * n_actions)
    return round(float(np.sum(early_p * np.log(early_p / late_p + 1e-12))), 4)


def run_episode(env, substrate, n_actions, seed, max_steps):
    obs            = env.reset(seed=seed)
    action_log     = []
    action_counts  = np.zeros(n_actions, np.float32)
    steps          = 0
    level          = 0
    progress_count = 0
    steps_to_first_progress = None
    t_start        = time.time()
    fresh_episode  = True
    i3_counts_at_200 = None

    while steps < max_steps:
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
        action  = substrate.process(obs_arr) % n_actions
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
            progress_count += 1
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

    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    return steps_to_first_progress, progress_count, {
        'steps_taken':             steps,
        'elapsed_seconds':         round(elapsed, 2),
        'steps_to_first_progress': steps_to_first_progress,
        'progress_count':          progress_count,
        'I3_cv':                   i3_cv,
        'action_kl':               compute_action_kl(action_log, n_actions),
        'action_entropy_norm':     substrate.get_action_entropy(),
        'wdrift':                  round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj':          substrate.get_pred_loss_trajectory(),
        'compression_ratio':       substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(game_name, label, condition):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    if condition == 'model_act':
        substrate = ModelActMlpTpSubstrate(n_actions=n_actions)
    else:
        substrate = MlpTpSubstrate(n_actions=n_actions)

    p1, _, result_try1 = run_episode(env, substrate, n_actions,
                                     seed=0, max_steps=TRY1_STEPS)
    substrate.reset_for_try2()

    p2, _, result_try2 = run_episode(env, substrate, n_actions,
                                     seed=TRY2_SEED, max_steps=TRY2_STEPS)

    speedup = compute_progress_speedup(p1, p2)
    opt = get_optimal_steps(game_name, TRY2_SEED)
    efficiency_sq = 0.0
    if p2 is not None and opt is not None and opt > 0:
        eff = min(1.0, opt / p2)
        efficiency_sq = round(eff ** 2, 6)

    return {
        'label':                   label,
        'condition':               condition,
        'n_actions':               n_actions,
        'try1':                    result_try1,
        'try2':                    result_try2,
        'second_exposure_speedup': speedup,
        'efficiency_sq_try2':      efficiency_sq,
        'optimal_steps':           opt,
        'compression_ratio':       result_try1.get('compression_ratio'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — World model action selection (MODEL-ACT vs MLP-TP)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"K={_MODEL_ACT_K} candidate actions sampled per step, warmup={_MODEL_ACT_WARMUP} steps")
    print(f"Conditions: {CONDITIONS}")
    print(f"Try1: {TRY1_STEPS} steps (seed 0), Try2: {TRY2_STEPS} steps (seed {TRY2_SEED})")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    # Tier 1: timing check
    first_arc = next((g for g in GAMES if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = GAME_LABELS[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, MODEL-ACT)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = ModelActMlpTpSubstrate(n_actions=na_t)
        t0 = time.time()
        obs_t = env_t.reset(seed=0)
        for _ in range(100):
            a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
            obs_t, _, done_t, _ = env_t.step(a)
            if done_t or obs_t is None:
                obs_t = env_t.reset(seed=0)
        elapsed_100 = time.time() - t0
        est_2k = elapsed_100 / 100 * 2000
        est_total = est_2k * len(GAMES) * len(CONDITIONS) * 2
        print(f"  100 steps: {elapsed_100:.1f}s → est 2K: {est_2k:.0f}s → est total: {est_total:.0f}s")
        if est_total > 300:
            print(f"  WARNING: est {est_total:.0f}s > 300s cap. Consider reducing K or steps.")
        print()

    all_results = []
    rhae_by_condition    = {c: None for c in CONDITIONS}
    speedup_by_condition = {c: [] for c in CONDITIONS}
    eff_sq_by_game_cond  = {c: {} for c in CONDITIONS}
    entropy_by_game_cond = {c: {} for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"=== Condition: {condition.upper()} ===")
        try2_progress = {}
        optimal_steps = {}

        for game_name, label in zip(GAMES, GAME_LABELS.values()):
            t_game = time.time()
            result = run_game(game_name, label, condition)
            elapsed = time.time() - t_game

            all_results.append(result)
            p2      = result['try2']['steps_to_first_progress']
            speedup = result['second_exposure_speedup']
            cr      = result['compression_ratio']
            eff     = result['efficiency_sq_try2']
            ent1    = result['try1'].get('action_entropy_norm')
            ent2    = result['try2'].get('action_entropy_norm')

            try2_progress[label] = p2
            optimal_steps[label] = result['optimal_steps']
            eff_sq_by_game_cond[condition][label] = eff
            entropy_by_game_cond[condition][label] = (ent1, ent2)

            print(f"  {label}: speedup={format_speedup(speedup)}  cr={cr}  eff²={eff}"
                  f"  ent1={ent1}  ent2={ent2}  ({elapsed:.1f}s)")
            speedup_by_condition[condition].append(speedup_for_chain(speedup))

            out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
            with open(out_path, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

        rhae = compute_rhae_try2(try2_progress, optimal_steps)
        rhae_by_condition[condition] = rhae
        chain_speedup = sum(speedup_by_condition[condition]) / len(speedup_by_condition[condition])
        print(f"  → RHAE(try2) = {rhae:.4f}  (chain_speedup={chain_speedup:.4f})")
        print()

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(f"STEP {STEP} — RESULT (World model action selection vs MLP-TP entropy)")
    print()
    print(f"  MODEL-ACT  RHAE(try2) = {rhae_by_condition['model_act']:.4f}")
    print(f"  MLP-TP     RHAE(try2) = {rhae_by_condition['mlp_tp']:.4f}")
    print()
    print("Per-game efficiency²:")
    for label in GAME_LABELS.values():
        m_eff = eff_sq_by_game_cond['model_act'].get(label, 0.0)
        e_eff = eff_sq_by_game_cond['mlp_tp'].get(label, 0.0)
        print(f"  {label}: MODEL={m_eff:.6f}  ENT={e_eff:.6f}")
    print()
    print("Action entropy diagnostic (normalized, 1.0=uniform, <1.0=concentrated):")
    for label in GAME_LABELS.values():
        m_ent = entropy_by_game_cond['model_act'].get(label, (None, None))
        e_ent = entropy_by_game_cond['mlp_tp'].get(label, (None, None))
        print(f"  {label}: MODEL try1={m_ent[0]}  try2={m_ent[1]}  |  "
              f"ENT try1={e_ent[0]}  try2={e_ent[1]}")
    print()

    # Kill/Signal assessment
    model_rhae = rhae_by_condition['model_act']
    mlp_rhae   = rhae_by_condition['mlp_tp']
    model_any_progress = any(
        r['try1'].get('steps_to_first_progress') is not None or
        r['try2'].get('steps_to_first_progress') is not None
        for r in all_results if r['condition'] == 'model_act'
    )
    print("KILL/SIGNAL ASSESSMENT:")
    if model_rhae > 0:
        print(f"  >>> LANDMARK: MODEL-ACT RHAE(try2) = {model_rhae:.4f}. First non-zero RHAE.")
    elif model_any_progress:
        print(f"  >>> SIGNAL: MODEL-ACT reached progress (but try2 efficiency=0). "
              f"Model-based selection produces level advancement.")
    elif model_rhae > mlp_rhae:
        print(f"  >>> SIGNAL: MODEL-ACT RHAE={model_rhae:.4f} > MLP-TP RHAE={mlp_rhae:.4f}.")
    else:
        # Check action concentration
        m_ent_vals = [v[1] for v in entropy_by_game_cond['model_act'].values() if v[1] is not None]
        e_ent_vals = [v[1] for v in entropy_by_game_cond['mlp_tp'].values() if v[1] is not None]
        avg_m = sum(m_ent_vals)/len(m_ent_vals) if m_ent_vals else None
        avg_e = sum(e_ent_vals)/len(e_ent_vals) if e_ent_vals else None
        if avg_m is not None and avg_e is not None and avg_m < avg_e * 0.9:
            print(f"  >>> CONCENTRATION: MODEL-ACT entropy={avg_m:.3f} < MLP-TP entropy={avg_e:.3f}. "
                  f"Model concentrating on subset of actions (same as 1306). KILL.")
        else:
            print(f"  >>> KILL: MODEL-ACT RHAE={model_rhae:.4f} ≤ MLP-TP RHAE={mlp_rhae:.4f}. "
                  f"Model-based selection doesn't help reach progress.")
    print("=" * 80)

    chain_speedups = {c: round(sum(speedup_by_condition[c]) / len(speedup_by_condition[c]), 4)
                      for c in CONDITIONS}
    write_experiment_results(
        RESULTS_DIR, STEP,
        rhae_by_condition=rhae_by_condition,
        all_results=all_results,
        conditions=CONDITIONS,
        game_labels=GAME_LABELS,
        speedup_by_condition=chain_speedups,
    )


if __name__ == '__main__':
    main()
