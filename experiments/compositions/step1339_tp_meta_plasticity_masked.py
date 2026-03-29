"""
Step 1339 — Meta-plasticity: substrate discovers its own update rule using TP credit signal.
Leo mail 3802, 2026-03-29.

Every prior experiment uses a designer-chosen update rule (local Adam per layer) frozen
into the substrate. Meta-plasticity lets the substrate discover which update rule works
best using its own prediction improvement as credit.

TP at 1337: 92% compression. Credit signal (pred_loss delta) now exists. This step
combines meta-plasticity with TP credit: the substrate uses prediction improvement to
evolve which update rule it applies.

Constitutional audit:
  R0: Deterministic init, seed-free ✓
  R1: Self-supervised. Credit = prediction improvement. No external signal ✓
  R2: Theta updates from prediction loss delta — computation itself generates learning ✓
  R3: Theta changes → update rule changes → weight dynamics → behavior ✓
  R4: Implicit in RHAE(try2) ✓
  R5: Game is ground truth ✓
  R6: Deletion test: remove theta = 1337 MLP-TP baseline ✓

Conditions:
  meta_mlp_tp: MLP + TP + learnable theta per layer (discovers update rule)
  mlp_tp:      MLP + TP, fixed update rule (1337 control baseline)

Theta per layer: [alpha_hebb, alpha_anti, alpha_decay, lr_scale]
  ΔW_i = lr_i * lr_scale * (TP + alpha_hebb*(h⊗x) - alpha_anti*(h⊗x) + alpha_decay*(-W))
Credit: pred_loss_before_K - pred_loss_after_K (positive = updates helped)
Theta update: theta_l += eta * credit * grad_approx_l (gradient wrt each alpha)

Budget: 5K steps × 2 tries × 3 games × 2 conditions = 60K steps.
Leo approved 5K steps (~14 min, within 15-min threshold). Mail 3807.

Kill:  META RHAE(try2) ≤ MLP RHAE(try2). KILL.
Signal: theta converges to non-trivial values. CONTINUE.
Landmark: META RHAE(try2) > 0. SUBSTRATE DISCOVERED SOMETHING.

One draw. Level-masked (steps_to_first_progress). Seed-free substrate init.
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

GAMES, GAME_LABELS = select_games(seed=1339)

STEP        = 1339
TRY1_STEPS  = 5000
TRY2_STEPS  = 5000
TRY2_SEED   = 4
MAX_SECONDS = 360   # 6 min per game (5K steps, Leo approved ~14 min total)

CONDITIONS  = ['meta_mlp_tp', 'mlp_tp']
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1339')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [2000, 5000, 10000]

# Meta-plasticity constants
_META_K             = 100    # theta update every K training steps
_META_ETA           = 0.001  # theta learning rate
_META_LR_SCALE_MIN  = 0.01   # lr_scale floor (prevent collapse to zero)
_N_FORWARD          = 4      # input_proj, fc1, fc2, fc3

# Theta indices
_TH_HEBB     = 0
_TH_ANTI     = 1
_TH_DECAY    = 2
_TH_LR_SCALE = 3


# ---------------------------------------------------------------------------
# Obs encoding (same as 1337)
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
    """Encode any obs to 1D float32. ARC: one-hot (65536). MBPP: raw (256)."""
    arr = np.asarray(obs_arr, dtype=np.float32)
    if _is_arc_obs(arr):
        return _obs_to_one_hot(arr).astype(np.float32).flatten()  # 65536
    return arr.flatten()  # MBPP: 256


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
        return h3, h2, h1, h0  # deepest → shallowest


# ---------------------------------------------------------------------------
# Meta-plasticity substrate (META-MLP-TP)
# ---------------------------------------------------------------------------

class MetaMlpTpSubstrate:
    """MLP + TP + learnable theta. Theta discovers the update rule.

    Per forward layer i, theta_i = [alpha_hebb, alpha_anti, alpha_decay, lr_scale].
    Modified update:
        ΔW_i += lr_i * lr_scale * (
            (alpha_hebb - alpha_anti) * outer(h_mean, x_mean)  # Hebbian/anti
            + alpha_decay * (-W_i)                              # weight decay
        )
    Applied AFTER Adam's TP local update (additive meta-modification).

    Credit formula (Leo directive):
        credit = mean(pred_loss[:K//2]) - mean(pred_loss[K//2:])
        Positive = updates helped prediction. Negative = hurt.

    Gradient approximation (per layer):
        g_hebb  = +h_rms * x_rms   (positive: more Hebbian when active)
        g_anti  = -h_rms * x_rms   (negative: less anti-Hebbian when active)
        g_decay = -w_norm           (negative: less decay when learning)
        g_lr_s  = 1.0               (positive: larger updates when improving)
    Sign convention: credit>0 (learning) → hebb↑, anti↓, decay↓, lr_scale↑.
    Credit<0 (stagnation) → hebb↓, anti↑, decay↑, lr_scale↓.

    Try2 inherits WEIGHTS and THETA. Theta is the discovered rule.
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

        # Meta-plasticity state
        self._theta           = None   # (N_FORWARD, 4) — initialized with model
        self._meta_h_acc      = [None] * _N_FORWARD  # EMA of h per layer
        self._meta_x_acc      = [None] * _N_FORWARD  # EMA of x per layer
        self._meta_loss_window = deque(maxlen=_META_K)
        self._meta_train_count = 0     # training steps since last theta update
        self._theta_try1_end  = None   # theta snapshot at try1 end

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
        # Initialize theta: [alpha_hebb=0, alpha_anti=0, alpha_decay=0, lr_scale=1]
        self._theta = np.zeros((_N_FORWARD, 4), dtype=np.float64)
        self._theta[:, _TH_LR_SCALE] = 1.0  # lr_scale starts at 1 (identity)

    def reset_for_try2(self):
        """Keep weights AND theta. Reset step/loss tracking. Theta persists."""
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}
        # Save theta snapshot at try1 end
        if self._theta is not None:
            self._theta_try1_end = self._theta.copy()
        # Reset meta accumulation window (new episode)
        self._meta_train_count = 0
        self._meta_loss_window.clear()
        self._meta_h_acc = [None] * _N_FORWARD
        self._meta_x_acc = [None] * _N_FORWARD

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
        """TP training step + theta-modulated direct weight update."""
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

        # TP forward pass + targets (same as 1337, no_grad context)
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

        # Update pred_head
        with torch.enable_grad():
            pred_in2 = torch.cat([h3.detach(), action_vecs.detach()], dim=1)
            pred_n2  = self._model.pred_head(pred_in2)
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()

        # Update g functions (unchanged)
        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                h_recon = self._g[gk](h_in.detach())
                loss_g  = F.mse_loss(h_recon, h_out.detach())
                self._opt_g[gk].zero_grad(); loss_g.backward(); self._opt_g[gk].step()

        # Forward layer TP updates (Adam) + theta-modulated direct update
        forward_layers = [
            ('f3',     self._model.fc3,        h2,     target_h3, h3, 3),
            ('f2',     self._model.fc2,        h1,     target_h2, h2, 2),
            ('f1',     self._model.fc1,        h0,     target_h1, h1, 1),
            ('f_proj', self._model.input_proj, states, target_h0, h0, 0),
        ]
        for fk, layer, x_in, target, h_out, layer_idx in forward_layers:
            with torch.enable_grad():
                h_fresh = F.relu(layer(x_in.detach()))
                loss_local = F.mse_loss(h_fresh, target.detach())
                self._opt_f[fk].zero_grad(); loss_local.backward(); self._opt_f[fk].step()

            # Theta-modulated direct weight update (additive to Adam)
            with torch.no_grad():
                theta = self._theta[layer_idx]
                alpha_hebb  = float(theta[_TH_HEBB])
                alpha_anti  = float(theta[_TH_ANTI])
                alpha_decay = float(theta[_TH_DECAY])
                lr_scale    = float(theta[_TH_LR_SCALE])

                # Rank-1 Hebbian: outer(h_mean, x_mean)
                h_mean = h_fresh.detach().mean(0)  # (d_out,)
                x_mean = x_in.detach().mean(0)     # (d_in,)

                net_hebb = alpha_hebb - alpha_anti
                H = torch.outer(h_mean, x_mean)  # (d_out, d_in)
                W = layer.weight.data

                # Apply meta-modulated update
                delta_W = _LR * lr_scale * (
                    net_hebb * H
                    + alpha_decay * (-W)
                )
                W += delta_W

                # Accumulate statistics for theta update
                h_np = h_mean.cpu().numpy()
                x_np = x_mean.cpu().numpy()
                if self._meta_h_acc[layer_idx] is None:
                    self._meta_h_acc[layer_idx] = h_np.copy()
                    self._meta_x_acc[layer_idx] = x_np.copy()
                else:
                    self._meta_h_acc[layer_idx] = (
                        0.9 * self._meta_h_acc[layer_idx] + 0.1 * h_np)
                    self._meta_x_acc[layer_idx] = (
                        0.9 * self._meta_x_acc[layer_idx] + 0.1 * x_np)

        # Track pred_loss for theta credit
        self._meta_loss_window.append(pred_loss)
        self._meta_train_count += 1
        if self._meta_train_count >= _META_K:
            self._update_theta()

        return pred_loss

    def _update_theta(self):
        """Update theta based on prediction loss credit over K-step window."""
        loss_list = list(self._meta_loss_window)
        if len(loss_list) < _META_K // 2:
            self._meta_train_count = 0
            return

        half = len(loss_list) // 2
        credit = float(np.mean(loss_list[:half]) - np.mean(loss_list[half:]))

        layers = [
            self._model.input_proj,
            self._model.fc1,
            self._model.fc2,
            self._model.fc3,
        ]

        for l, layer in enumerate(layers):
            if self._meta_h_acc[l] is None:
                continue

            h = self._meta_h_acc[l]
            x = self._meta_x_acc[l]
            W_np = layer.weight.data.cpu().numpy()

            h_rms = float(np.sqrt(np.mean(h ** 2) + 1e-12))
            x_rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
            w_norm = float(np.linalg.norm(W_np, 'fro')) / (W_np.size + 1)

            # Signed gradient approximations:
            # credit > 0 (learning) → hebb↑, anti↓ (less anti-Hebbian), decay↓, lr_scale↑
            # credit < 0 (stagnation) → hebb↓, anti↑, decay↑, lr_scale↓
            g = np.array([
                h_rms * x_rms,    # g_alpha_hebb: positive
                -(h_rms * x_rms), # g_alpha_anti: negative
                -w_norm,          # g_alpha_decay: negative
                1.0,              # g_lr_scale: positive
            ])

            self._theta[l] += _META_ETA * credit * g
            # lr_scale floor: prevent collapse to zero
            self._theta[l, _TH_LR_SCALE] = max(_META_LR_SCALE_MIN,
                                                 self._theta[l, _TH_LR_SCALE])

        self._meta_train_count = 0

    def compute_weight_drift(self):
        if self._model is None or self._ep_start_sd is None:
            return 0.0
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._ep_start_sd:
                drift += (param.data.cpu() - self._ep_start_sd[name]).norm().item()
        return float(drift)

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(2000)
        l_late  = self._pred_loss_at.get(10000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_theta_summary(self):
        if self._theta is None:
            return None
        return {
            f'L{l}': {
                'alpha_hebb': round(float(self._theta[l, _TH_HEBB]),     6),
                'alpha_anti': round(float(self._theta[l, _TH_ANTI]),     6),
                'alpha_decay': round(float(self._theta[l, _TH_DECAY]),   6),
                'lr_scale':   round(float(self._theta[l, _TH_LR_SCALE]), 6),
                'net_hebb':   round(float(self._theta[l, _TH_HEBB] -
                                          self._theta[l, _TH_ANTI]),     6),
            }
            for l in range(_N_FORWARD)
        }

    def get_theta_try1_end(self):
        if self._theta_try1_end is None:
            return None
        return {
            f'L{l}': {
                'alpha_hebb': round(float(self._theta_try1_end[l, _TH_HEBB]),     6),
                'alpha_anti': round(float(self._theta_try1_end[l, _TH_ANTI]),     6),
                'alpha_decay': round(float(self._theta_try1_end[l, _TH_DECAY]),   6),
                'lr_scale':   round(float(self._theta_try1_end[l, _TH_LR_SCALE]), 6),
            }
            for l in range(_N_FORWARD)
        }


# ---------------------------------------------------------------------------
# MLP-TP substrate (control, same as step1337)
# ---------------------------------------------------------------------------

class MlpTpSubstrate:
    """MLP + TP control. Fixed update rule. No theta. Identical to step 1337."""

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
        l_early = self._pred_loss_at.get(2000)
        l_late  = self._pred_loss_at.get(10000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_theta_summary(self):
        return None  # no theta in control

    def get_theta_try1_end(self):
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
    """Return optimal_steps for RHAE. MBPP: exact. ARC: proxy=10."""
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        problem_idx = int(seed) % mbpp_game.N_EVAL_PROBLEMS
        solver = mbpp_game.compute_solver_steps(problem_idx)
        return solver.get(1)
    return ARC_OPTIMAL_STEPS_PROXY  # fixed proxy for ARC


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
        'wdrift':                  round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj':          substrate.get_pred_loss_trajectory(),
        'compression_ratio':       substrate.get_compression_ratio(),
        'theta_end':               substrate.get_theta_summary(),
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

    if condition == 'meta_mlp_tp':
        substrate = MetaMlpTpSubstrate(n_actions=n_actions)
    else:
        substrate = MlpTpSubstrate(n_actions=n_actions)

    p1, _, result_try1 = run_episode(env, substrate, n_actions,
                                     seed=0, max_steps=TRY1_STEPS)
    theta_try1_end = substrate.get_theta_try1_end()
    substrate.reset_for_try2()

    p2, _, result_try2 = run_episode(env, substrate, n_actions,
                                     seed=TRY2_SEED, max_steps=TRY2_STEPS)
    theta_try2_end = substrate.get_theta_summary()

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
        'theta_try1_end':          theta_try1_end,
        'theta_try2_end':          theta_try2_end,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _theta_to_str(theta_summary):
    """Format theta summary for display."""
    if theta_summary is None:
        return '  (no theta — control)'
    lines = []
    for layer_key, vals in theta_summary.items():
        lines.append(
            f"  {layer_key}: hebb={vals['alpha_hebb']:+.5f}  "
            f"anti={vals['alpha_anti']:+.5f}  "
            f"decay={vals['alpha_decay']:+.5f}  "
            f"lr_scale={vals['lr_scale']:.5f}  "
            f"net={vals['net_hebb']:+.5f}"
        )
    return '\n'.join(lines)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Meta-plasticity + MLP-TP: substrate discovers update rule")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Try1: {TRY1_STEPS} steps (seed 0), Try2: {TRY2_STEPS} steps (seed {TRY2_SEED})")
    print(f"Meta: K={_META_K} steps, eta={_META_ETA}, theta=[alpha_hebb, alpha_anti, alpha_decay, lr_scale]")
    print(f"Runtime: ~14 min estimated (Leo approved 5K steps, 15-min threshold)")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    # Tier 1: timing check on first ARC game
    first_arc = next((g for g in GAMES if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = GAME_LABELS[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, META-MLP-TP)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = MetaMlpTpSubstrate(n_actions=na_t)
        t0 = time.time()
        obs_t = env_t.reset(seed=0)
        for _ in range(100):
            a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
            obs_t, _, done_t, _ = env_t.step(a)
            if done_t or obs_t is None:
                obs_t = env_t.reset(seed=0)
        elapsed_100 = time.time() - t0
        est_5k = elapsed_100 / 100 * 5000
        est_total = est_5k * len(GAMES) * len(CONDITIONS) * 2
        print(f"  100 steps: {elapsed_100:.1f}s → est 5K: {est_5k:.0f}s → est total: {est_total:.0f}s")
        print()

    all_results = []
    rhae_by_condition    = {c: None for c in CONDITIONS}
    speedup_by_condition = {c: [] for c in CONDITIONS}
    eff_sq_by_game_cond  = {c: {} for c in CONDITIONS}
    theta_by_game_cond   = {c: {} for c in CONDITIONS}

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

            try2_progress[label] = p2
            optimal_steps[label] = result['optimal_steps']
            eff_sq_by_game_cond[condition][label] = eff

            # Record theta trajectory for meta condition
            if result['theta_try2_end'] is not None:
                theta_by_game_cond[condition][label] = {
                    'try1_end': result['theta_try1_end'],
                    'try2_end': result['theta_try2_end'],
                }

            print(f"  {label}: speedup={format_speedup(speedup)}  cr={cr}  eff²={eff}  ({elapsed:.1f}s)")
            speedup_by_condition[condition].append(speedup_for_chain(speedup))

            # Save per-game JSONL
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
    print(f"STEP {STEP} — RESULT (Meta-plasticity vs MLP-TP: discovered update rule)")
    print()
    print(f"  META-MLP-TP  RHAE(try2) = {rhae_by_condition['meta_mlp_tp']:.4f}")
    print(f"  MLP-TP       RHAE(try2) = {rhae_by_condition['mlp_tp']:.4f}")
    print()
    print("Per-game efficiency²:")
    for label in GAME_LABELS.values():
        meta_eff = eff_sq_by_game_cond['meta_mlp_tp'].get(label, 0.0)
        mlp_eff  = eff_sq_by_game_cond['mlp_tp'].get(label, 0.0)
        print(f"  {label}: META={meta_eff:.6f}  MLP={mlp_eff:.6f}")
    print()

    # Theta trajectory report
    print("Theta trajectory (META condition only):")
    for label, theta_data in theta_by_game_cond.get('meta_mlp_tp', {}).items():
        print(f"  {label} — Try1 end:")
        print(_theta_to_str(theta_data.get('try1_end')))
        print(f"  {label} — Try2 end:")
        print(_theta_to_str(theta_data.get('try2_end')))
        print()

    # Kill/Signal assessment
    meta_rhae = rhae_by_condition['meta_mlp_tp']
    mlp_rhae  = rhae_by_condition['mlp_tp']
    print("KILL/SIGNAL ASSESSMENT:")
    if meta_rhae > 0:
        print(f"  >>> LANDMARK: META RHAE(try2) = {meta_rhae:.4f} > 0. Substrate discovered helpful update rule.")
    elif meta_rhae > mlp_rhae:
        print(f"  >>> SIGNAL: META={meta_rhae:.4f} > MLP={mlp_rhae:.4f}. Theta helps even without progress.")
    else:
        # Check theta non-triviality
        has_nontrivial_theta = any(
            abs(v) > 0.01
            for game_data in theta_by_game_cond.get('meta_mlp_tp', {}).values()
            for snap in [game_data.get('try2_end')]
            if snap
            for layer_data in snap.values()
            for k, v in layer_data.items()
            if k != 'lr_scale'
        )
        if has_nontrivial_theta:
            print(f"  >>> CONTINUE: META RHAE={meta_rhae:.4f} ≤ MLP={mlp_rhae:.4f} "
                  f"but theta non-trivial (|alpha| > 0.01). Substrate discovered structure.")
        else:
            print(f"  >>> KILL: META RHAE={meta_rhae:.4f} ≤ MLP={mlp_rhae:.4f}. Theta trivial (~0). "
                  f"Credit signal insufficient to drive meta-plasticity.")
    print("=" * 80)

    # Write results files
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
