"""
Step 1343 — Action-conditional model with rich action encoding.
Leo mail 3821, 2026-03-29.

Same as 1342 but fixes the root cause: action_type_vec (6-dim, action%6) collapses
spatial information. novelty_var ≈ 0 in 1342 because the model sees identical
encodings for different-position clicks.

Action encoding fix:
  ARC:  (type_onehot[7], x_norm, y_norm, zeros[7]) = 16-dim.
        Click at (32,10) ≠ click at (32,50) — position preserved.
  MBPP: char_embed(128→16) = 16-dim. Learned. 'a' ≠ 'g'.

Everything else identical to 1342:
  - 2-layer nonlinear pred_head (512+16 → 512 → ReLU → 512)
  - Noop-relative novelty: ||pred_act - pred_noop||
  - Warmup = 500, K=32 candidates, 2K steps, seed-free
  - Control: MLP-TP entropy

Constitutional audit: identical to 1342. R1-R6 pass. Only input encoding changed.

Kill:   novelty_var ≈ 0 → even rich encoding can't differentiate. KILL.
Signal: novelty_var > 0.01 → model IS differentiating. Mechanism exists.
        RHAE > 0 → differentiation guides progress. LANDMARK.
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
    masked_game_list,
    compute_progress_speedup, format_speedup,
    speedup_for_chain, compute_rhae_try2, write_experiment_results,
    ARC_OPTIMAL_STEPS_PROXY,
)

GAMES, GAME_LABELS = select_games(seed=1343)

STEP        = 1343
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

CONDITIONS  = ['ac_enc', 'mlp_tp']
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1343')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]

_AC_WARMUP     = 500
_AC_K          = 32
_ACTION_ENC_DIM = 16   # unified action encoding dimension for pred_head
_MBPP_THRESHOLD = 200  # n_actions <= this → MBPP (use char embed)
_ARC_N_TYPES    = 7    # ARC action types


# ---------------------------------------------------------------------------
# Obs encoding (same as 1337-1342)
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


# ---------------------------------------------------------------------------
# Rich action encoding (the fix)
# ---------------------------------------------------------------------------

def _encode_arc_action(action):
    """ARC: (type_onehot[7], x_norm, y_norm) padded to 16-dim."""
    enc = np.zeros(_ACTION_ENC_DIM, dtype=np.float32)
    a = int(action)
    type_idx = a % _ARC_N_TYPES
    enc[type_idx] = 1.0                    # type onehot [0:7]
    x = (a // _ARC_N_TYPES) % 64
    y = (a // (_ARC_N_TYPES * 64)) % 64
    enc[7] = x / 63.0                      # x_norm [7]
    enc[8] = y / 63.0                      # y_norm [8]
    # enc[9:16] = 0 (padding)
    return enc


_NOOP_ENC = np.zeros(_ACTION_ENC_DIM, dtype=np.float32)  # null action = zeros


# ---------------------------------------------------------------------------
# MLP model with 2-layer pred_head (16-dim action input)
# ---------------------------------------------------------------------------

class AcEncMlpSelfSupModel(nn.Module):
    """MLP + 2-layer nonlinear pred_head with 16-dim action encoding.

    For MBPP: has char_embed(128, 16). For ARC: char_embed unused.
    pred_head: Linear(512+16, 512) → ReLU → Linear(512, 512).
    """
    def __init__(self, input_dim, n_actions, hidden=_HIDDEN, is_mbpp=False):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden)
        self.fc1         = nn.Linear(hidden, hidden)
        self.fc2         = nn.Linear(hidden, hidden)
        self.fc3         = nn.Linear(hidden, hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        self.pred_head1  = nn.Linear(hidden + _ACTION_ENC_DIM, hidden)
        self.pred_head2  = nn.Linear(hidden, hidden)
        self.dropout     = nn.Dropout(0.2)
        self.is_mbpp     = is_mbpp
        if is_mbpp:
            self.char_embed = nn.Embedding(n_actions, _ACTION_ENC_DIM)

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

    def pred_next(self, h3, action_enc):
        """Predict next h3. action_enc: (batch, 16) float tensor."""
        pred_in = torch.cat([h3, action_enc], dim=1)
        return self.pred_head2(F.relu(self.pred_head1(pred_in)))

    def encode_action_batch(self, action_encs_np):
        """Convert numpy (K, 16) action encodings to tensor for ARC,
        or embed integer actions for MBPP."""
        return torch.from_numpy(action_encs_np).to(next(self.parameters()).device)

    def embed_actions(self, action_ints):
        """MBPP: embed integer action indices → (batch, 16) float."""
        assert self.is_mbpp, "embed_actions only for MBPP"
        idx = torch.tensor(action_ints, dtype=torch.long,
                           device=next(self.parameters()).device)
        return self.char_embed(idx)


# ---------------------------------------------------------------------------
# Action-conditional substrate with rich encoding (AC-ENC)
# ---------------------------------------------------------------------------

class AcEncMlpTpSubstrate:
    """MLP + TP + action-conditional model with rich action encoding.

    Same as 1342 but action encoding fixes action identity:
    - ARC: (type_onehot[7], x_norm, y_norm) = 9-dim padded to 16
    - MBPP: char_embed(n_actions, 16) = 16-dim learned

    Noop: zeros(16) for both games.
    """

    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._is_mbpp    = n_actions <= _MBPP_THRESHOLD
        self._rng        = np.random.RandomState(42)
        self._model      = None
        self._g          = None
        self._opt_pred   = None
        self._opt_g      = None
        self._opt_f      = None
        self._opt_embed  = None
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
        self._action_log_all   = []
        self._novelty_var_log  = []

    def _encode_action_np(self, action):
        """Return 16-dim numpy encoding for a single action."""
        if self._is_mbpp:
            # Will use model embedding at inference; return placeholder for buffer
            enc = np.zeros(_ACTION_ENC_DIM, dtype=np.float32)
            enc[0] = float(int(action))  # store action index as a scalar (for buffer)
            return enc
        return _encode_arc_action(action)

    def _action_enc_tensor(self, actions):
        """Return (len(actions), 16) tensor for action encoding.

        For ARC: pure numpy → tensor.
        For MBPP: use char_embed.
        """
        if self._is_mbpp:
            return self._model.embed_actions(actions)  # (K, 16)
        encs = np.stack([_encode_arc_action(a) for a in actions])
        return torch.from_numpy(encs).to(_DEVICE)

    def _noop_tensor(self, batch_size=1):
        """Return (batch_size, 16) noop encoding tensor."""
        return torch.zeros(batch_size, _ACTION_ENC_DIM, dtype=torch.float32,
                           device=_DEVICE)

    def _init_model(self, input_dim):
        self._input_dim = input_dim
        self._model = AcEncMlpSelfSupModel(
            input_dim, self.n_actions, is_mbpp=self._is_mbpp).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        pred_params = (list(self._model.pred_head1.parameters()) +
                       list(self._model.pred_head2.parameters()))
        if self._is_mbpp:
            pred_params += list(self._model.char_embed.parameters())
        self._opt_pred = torch.optim.Adam(pred_params, lr=_LR)
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
        self._action_log_all  = []
        self._novelty_var_log = []

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc

        if self._step <= _AC_WARMUP or len(self._buffer) < _BATCH_SIZE:
            action = int(self._rng.randint(self.n_actions))
            self._prev_action = action
            self._action_log_all.append(action)
            return action % self.n_actions

        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            h3, _, _, _ = self._model.forward_all_layers(tensor)

            # Noop prediction
            noop_t = self._noop_tensor(1)
            pred_noop = self._model.pred_next(h3, noop_t)

            # K candidates
            k = min(_AC_K, self.n_actions)
            candidates = self._rng.choice(self.n_actions, k, replace=False)
            act_encs   = self._action_enc_tensor(candidates)  # (K, 16)

            h3_rep   = h3.expand(k, -1)
            noop_rep = pred_noop.expand(k, -1)
            pred_acts = self._model.pred_next(h3_rep, act_encs)

            novelty  = (pred_acts - noop_rep).norm(dim=1)
            nov_var  = float(novelty.var().item())
            self._novelty_var_log.append(nov_var)

            best_idx = int(novelty.argmax().item())
            action   = int(candidates[best_idx])

        self._prev_action = action
        self._action_log_all.append(action)
        return action % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)
        # Store action index in buffer (decoded in train step)
        at_enc_np = self._encode_action_np(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state':      self._prev_enc.copy(),
                'action_int': int(action),      # store int for MBPP embed
                'action_enc': at_enc_np,        # ARC: real encoding; MBPP: placeholder
                'next_state': enc_next.copy(),
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

        # Build action encodings for batch
        if self._is_mbpp:
            action_ints = [b['action_int'] for b in batch]
            with torch.no_grad():
                action_encs = self._model.embed_actions(action_ints)
        else:
            action_encs = torch.from_numpy(
                np.stack([b['action_enc'] for b in batch])).to(_DEVICE)

        with torch.no_grad():
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_next = self._model.pred_next(h3, action_encs)
            pred_err  = pred_next - target_h3_next
            pred_loss = float((pred_err ** 2).mean())

            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)

        with torch.enable_grad():
            if self._is_mbpp:
                action_encs_g = self._model.embed_actions(action_ints)
            else:
                action_encs_g = action_encs.detach()
            pred_n2 = self._model.pred_next(h3.detach(), action_encs_g)
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

    def get_novelty_var_mean(self):
        if not self._novelty_var_log:
            return None
        return round(float(np.mean(self._novelty_var_log)), 6)


# ---------------------------------------------------------------------------
# MLP-TP control (same as 1337-1342)
# ---------------------------------------------------------------------------

class MlpTpSubstrate:
    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(42)
        self._model      = None
        self._g          = None
        self._opt_pred   = None
        self._opt_g      = None
        self._opt_f      = None
        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._train_counter = 0
        self._step          = 0
        self._ep_start_sd   = None
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}
        self._action_log_all = []

    def _init_model(self, input_dim):
        class _MlpModel(nn.Module):
            def __init__(self, inp, n_act, hid=_HIDDEN):
                super().__init__()
                self.input_proj  = nn.Linear(inp, hid)
                self.fc1         = nn.Linear(hid, hid)
                self.fc2         = nn.Linear(hid, hid)
                self.fc3         = nn.Linear(hid, hid)
                self.action_head = nn.Linear(hid, n_act)
                self.pred_head   = nn.Linear(hid + 6, hid)
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
        self._model = _MlpModel(input_dim, self.n_actions).to(_DEVICE)
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
        self._ep_start_sd = {k: v.clone().cpu() for k, v in self._model.state_dict().items()}

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
        probs = torch.softmax(logits.squeeze(0).cpu(), dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        self._action_log_all.append(action)
        return action % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)
        at_vec = np.zeros(6, dtype=np.float32)
        at_vec[int(action) % 6] = 1.0
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state': self._prev_enc.copy(),
                                 'action_type_vec': at_vec,
                                 'next_state': enc_next.copy()})
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
                h_r = self._g[gk](h_in.detach())
                lg = F.mse_loss(h_r, h_out.detach())
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

    def compute_weight_drift(self):
        if self._model is None or self._ep_start_sd is None:
            return 0.0
        return float(sum((p.data.cpu() - self._ep_start_sd[n]).norm().item()
                         for n, p in self._model.named_parameters()
                         if n in self._ep_start_sd))

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
            if 0 <= a < self.n_actions: counts[a] += 1
        total = counts.sum()
        if total < 1: return None
        probs = (counts / total) + 1e-10
        return round(-float(np.sum(probs * np.log(probs))) / float(np.log(self.n_actions)), 4)

    def get_novelty_var_mean(self):
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
        'novelty_var_mean':        substrate.get_novelty_var_mean(),
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

    substrate = (AcEncMlpTpSubstrate(n_actions=n_actions)
                 if condition == 'ac_enc'
                 else MlpTpSubstrate(n_actions=n_actions))

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
    print(f"Step {STEP} — Action-conditional model with rich encoding (AC-ENC vs MLP-TP)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"Action encoding: ARC=(type_1hot[7],x_norm,y_norm,pad)=16-dim | MBPP=char_embed(128→16)")
    print(f"pred_head: 2-layer (512+16→512→ReLU→512). Noop-relative novelty.")
    print(f"Warmup={_AC_WARMUP}, K={_AC_K}. Conditions: {CONDITIONS}")
    print(f"Try1: {TRY1_STEPS} steps (seed 0), Try2: {TRY2_STEPS} steps (seed {TRY2_SEED})")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    first_arc = next((g for g in GAMES if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = GAME_LABELS[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, AC-ENC)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = AcEncMlpTpSubstrate(n_actions=na_t)
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
            print(f"  WARNING: est {est_total:.0f}s > 300s cap.")
        print()

    all_results = []
    rhae_by_condition    = {c: None for c in CONDITIONS}
    speedup_by_condition = {c: [] for c in CONDITIONS}
    eff_sq_by_game_cond  = {c: {} for c in CONDITIONS}

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
            nv1     = result['try1'].get('novelty_var_mean')
            nv2     = result['try2'].get('novelty_var_mean')

            try2_progress[label] = p2
            optimal_steps[label] = result['optimal_steps']
            eff_sq_by_game_cond[condition][label] = eff

            nv_str = f'  nov_var={nv1}/{nv2}' if nv1 is not None else ''
            print(f"  {label}: speedup={format_speedup(speedup)}  cr={cr}  eff²={eff}"
                  f"  ent={ent1}/{ent2}{nv_str}  ({elapsed:.1f}s)")
            speedup_by_condition[condition].append(speedup_for_chain(speedup))

            out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
            with open(out_path, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

        rhae = compute_rhae_try2(try2_progress, optimal_steps)
        rhae_by_condition[condition] = rhae
        chain_speedup = sum(speedup_by_condition[condition]) / len(speedup_by_condition[condition])
        print(f"  → RHAE(try2) = {rhae:.4f}  (chain_speedup={chain_speedup:.4f})")
        print()

    print("=" * 80)
    print(f"STEP {STEP} — RESULT (Rich action encoding vs MLP-TP)")
    print()
    print(f"  AC-ENC   RHAE(try2) = {rhae_by_condition['ac_enc']:.4f}")
    print(f"  MLP-TP   RHAE(try2) = {rhae_by_condition['mlp_tp']:.4f}")
    print(f"  AC-1L    RHAE(try2) = 0.0000  (1340 — 1-layer linear head)")
    print(f"  AC-2L    RHAE(try2) = 0.0000  (1342 — 2-layer, 6-dim encoding)")
    print()
    print("Per-game efficiency²:")
    for label in GAME_LABELS.values():
        ac_val  = eff_sq_by_game_cond['ac_enc'].get(label, 0.0)
        ctl_val = eff_sq_by_game_cond['mlp_tp'].get(label, 0.0)
        print(f"  {label}: AC-ENC={ac_val:.6f}  ENT={ctl_val:.6f}")
    print()
    print("Novelty variance (key diagnostic, >0.01 = model differentiates):")
    for r in all_results:
        if r['condition'] == 'ac_enc':
            nv1 = r['try1'].get('novelty_var_mean')
            nv2 = r['try2'].get('novelty_var_mean')
            sig = 'DIFFERENTIATES' if ((nv1 or 0) > 0.01 or (nv2 or 0) > 0.01) else 'flat'
            print(f"  {r['label']}: try1={nv1}  try2={nv2}  {sig}")
    print()

    ac_rhae = rhae_by_condition['ac_enc']
    any_differentiate = any(
        r['condition'] == 'ac_enc' and
        ((r['try1'].get('novelty_var_mean') or 0) > 0.01 or
         (r['try2'].get('novelty_var_mean') or 0) > 0.01)
        for r in all_results
    )
    print("KILL/SIGNAL ASSESSMENT:")
    if ac_rhae is not None and ac_rhae > 0:
        print(f"  >>> SIGNAL: AC-ENC RHAE={ac_rhae:.4f} > 0. Rich encoding guides progress!")
        print(f"  >>> LANDMARK: First model-based action selection with RHAE > 0.")
    elif any_differentiate:
        print(f"  >>> PARTIAL: AC-ENC differentiates actions (nov_var > 0.01) but RHAE=0.")
        print(f"  >>> Model learned action-conditional dynamics. Selection doesn't reach progress.")
        print(f"  >>> Continue: mechanism exists, need better novelty target or more steps.")
    else:
        print(f"  >>> KILL: AC-ENC RHAE=0 AND novelty_var≈0. Even rich encoding can't differentiate.")
        print(f"  >>> Deeper architecture needed (full action-conditional MLP core).")
    print("=" * 80)

    write_experiment_results(
        RESULTS_DIR, STEP,
        rhae_by_condition, all_results, CONDITIONS,
        game_labels=GAME_LABELS,
        speedup_by_condition={
            c: sum(v) / len(v) for c, v in speedup_by_condition.items() if v
        }
    )


if __name__ == '__main__':
    main()
