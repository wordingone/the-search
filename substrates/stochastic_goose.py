"""StochasticGoose substrate — exact port of DriesSmit/ARC3-solution.

Codebase: https://github.com/DriesSmit/ARC3-solution
Original author: Dries Smit

Porting notes (Jun directive, Leo mail 3662/3663, 2026-03-28):
  - CNN architecture exact: 4 conv layers (32→64→128→256) + action head + coord head
  - Experience buffer exact: 200K deque, MD5 hash dedup, binary frame-change reward
  - Training loop exact: Adam lr=0.0001, batch=64, frequency=5, entropy regularization
  - Level transition: reset model AND buffer (exact per original)
  - Observation: (1, 64, 64) int8 color indices 0-15 → one-hot (16, 64, 64)
  - Action mapping: SG indices 0-4 → PRISM keyboard 0-4; SG 5+ → PRISM click (7 + click_idx)
    - SG can't access PRISM keyboard actions 5-6 (ACTION6/7) — inherent limit of their arch
  - MBPP (n_actions=128): not supported — falls back to random

PRISM substrate interface:
  process(obs_arr)                    → action int
  update_after_step(obs_next, action, reward)  → None (frame-change reward computed here)
  on_level_transition()               → None (reset model + buffer)
  get_state()                         → dict
  get_internal_repr_readonly(obs_raw, *args)   → np.zeros(1)  (not used for SG)
"""
import numpy as np
import hashlib
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use CUDA if available — architecture/hyperparams unchanged, just device
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# CNN — exact replica of ActionModel from custom_agents/action.py
# ---------------------------------------------------------------------------

class ActionModel(nn.Module):
    """Exact ActionModel from DriesSmit/ARC3-solution custom_agents/action.py."""

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Action head — 5 discrete actions
        self.action_pool = nn.MaxPool2d(4, 4)          # 64×64 → 16×16
        action_flattened_size = 256 * 16 * 16          # 65,536
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        # Coordinate head — 64×64 click grid (fully spatial, no FC)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)  # → (batch, 1, 64, 64)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, 16, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))           # (batch, 256, 64, 64)

        # Action head
        af = self.action_pool(conv_features)            # (batch, 256, 16, 16)
        af = af.view(af.size(0), -1)                    # (batch, 65536)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)            # (batch, 5)

        # Coordinate head
        cf = F.relu(self.coord_conv1(conv_features))    # (batch, 128, 64, 64)
        cf = F.relu(self.coord_conv2(cf))               # (batch, 64, 64, 64)
        cf = F.relu(self.coord_conv3(cf))               # (batch, 32, 64, 64)
        cf = self.coord_conv4(cf)                       # (batch, 1, 64, 64)
        coord_logits = cf.view(cf.size(0), -1)          # (batch, 4096)

        return torch.cat([action_logits, coord_logits], dim=1)  # (batch, 4101)


# ---------------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------------

_BUFFER_MAXLEN = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
_LR = 0.0001
_ACTION_ENTROPY_COEF = 0.0001
_COORD_ENTROPY_COEF = 0.00001
_SG_OUTPUT_DIM = 4101  # 5 discrete + 4096 click

N_KEYBOARD = 7   # PRISM keyboard offset for click actions


def _obs_to_one_hot(obs_arr):
    """Convert (1, 64, 64) float32 (values 0.0-15.0) → (16, 64, 64) bool array."""
    frame = np.round(obs_arr).astype(np.int32).squeeze(0)  # (64, 64)
    frame = np.clip(frame, 0, 15)
    one_hot = np.zeros((16, 64, 64), dtype=np.bool_)
    for c in range(16):
        one_hot[c] = (frame == c)
    return one_hot


def _is_arc_obs(obs_arr):
    """True if observation looks like ARC (1, 64, 64) color-index frame."""
    arr = np.asarray(obs_arr, dtype=np.float32)
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


class StochasticGooseSubstrate:
    """Exact port of StochasticGoose. PRISM substrate interface.

    For MBPP (n_actions=128) or non-ARC observations: falls back to random selection.
    """

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._is_arc = (n_actions > 5)  # heuristic: MBPP has 128 but no grid
        # We'll confirm on first obs in process()

        self._model = ActionModel(input_channels=16, grid_size=64).to(_DEVICE)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=_LR)
        self._buffer = deque(maxlen=_BUFFER_MAXLEN)
        self._buffer_hashes = set()

        self._prev_one_hot = None   # stored in process(), used in update_after_step()
        self._train_counter = 0
        self.step = 0
        self._use_random = (n_actions == 128)  # MBPP

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, obs_arr):
        """Select action. Stores obs for update_after_step."""
        self.step += 1
        obs_arr = np.asarray(obs_arr, dtype=np.float32)

        if self._use_random or not _is_arc_obs(obs_arr):
            self._prev_one_hot = None
            return int(self._rng.randint(self.n_actions))

        one_hot = _obs_to_one_hot(obs_arr)
        self._prev_one_hot = one_hot

        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)  # (1,16,64,64)
        with torch.no_grad():
            logits = self._model(tensor).squeeze(0).cpu()  # (4101,)

        sg_idx = self._sample_sg_action(logits)
        return self._sg_to_prism(sg_idx)

    def update_after_step(self, obs_next, action, reward_env):
        """Compute frame-change reward, add to buffer, train."""
        if self._use_random or self._prev_one_hot is None:
            return

        obs_next_arr = np.asarray(obs_next, dtype=np.float32)
        if not _is_arc_obs(obs_next_arr):
            return

        one_hot_next = _obs_to_one_hot(obs_next_arr)
        # Frame-change reward: 1.0 if any pixel changed
        frame_changed = not np.array_equal(self._prev_one_hot, one_hot_next)
        reward = 1.0 if frame_changed else 0.0

        # Map PRISM action back to SG index for buffer storage
        sg_idx = self._prism_to_sg(action)
        self._add_to_buffer(self._prev_one_hot, sg_idx, reward)

        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            self._train_step()

    def on_level_transition(self):
        """Reset model and buffer on level advance — exact per original."""
        self._model = ActionModel(input_channels=16, grid_size=64).to(_DEVICE)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=_LR)
        self._buffer.clear()
        self._buffer_hashes.clear()
        self._prev_one_hot = None

    def get_state(self):
        return {'buffer_size': len(self._buffer), 'step': self.step}

    def get_internal_repr_readonly(self, obs_raw, *args):
        return np.zeros(1, np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_sg_action(self, logits):
        """Sample SG action index from combined logits (exact per original)."""
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

        return int(torch.multinomial(combined, 1).item())

    def _sg_to_prism(self, sg_idx):
        """Map SG index → PRISM action int."""
        if sg_idx < 5:
            # Keyboard ACTION1-5 → our actions 0-4
            return sg_idx
        else:
            # Click: SG click_idx = sg_idx - 5 → PRISM click = N_KEYBOARD + click_idx
            click_idx = sg_idx - 5  # 0..4095
            prism_action = N_KEYBOARD + click_idx  # 7..4102
            if prism_action < self.n_actions:
                return prism_action
            # Fallback: keyboard
            return sg_idx % 5

    def _prism_to_sg(self, prism_action):
        """Map PRISM action → SG index (for buffer storage)."""
        if prism_action < 5:
            return prism_action
        elif prism_action >= N_KEYBOARD:
            click_idx = prism_action - N_KEYBOARD
            return 5 + click_idx  # SG click idx
        else:
            # Actions 5-6: unmapped in SG — store as nearest keyboard
            return prism_action % 5

    def _add_to_buffer(self, one_hot, sg_idx, reward):
        """Hash-deduplicated add."""
        state_bytes = one_hot.tobytes()
        action_bytes = np.array([sg_idx], dtype=np.int32).tobytes()
        h = hashlib.md5(state_bytes + action_bytes).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state': one_hot.copy(),
                'action_idx': sg_idx,
                'reward': reward,
            })

    def _train_step(self):
        """Train step — exact per original (deque→list for O(1) random access)."""
        n = len(self._buffer)
        buf_list = list(self._buffer)  # deque[i] is O(n); list[i] is O(1)
        indices = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf_list[i] for i in indices]

        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])
        ).to(_DEVICE)  # (B, 16, 64, 64)
        actions = [b['action_idx'] for b in batch]
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32).to(_DEVICE)

        logits = self._model(states)  # (B, 4101)

        selected_logits = torch.stack([logits[i, a] for i, a in enumerate(actions)])
        main_loss = F.binary_cross_entropy_with_logits(selected_logits, rewards)

        # Entropy regularization (exact per original)
        al = logits[:, :5]
        cl = logits[:, 5:]
        ap = torch.sigmoid(al)
        cp = torch.sigmoid(cl)
        action_entropy = -(ap * (ap + 1e-8).log()).sum(dim=1).mean()
        coord_entropy = -(cp * (cp + 1e-8).log()).sum(dim=1).mean()

        total_loss = main_loss - _ACTION_ENTROPY_COEF * action_entropy - _COORD_ENTROPY_COEF * coord_entropy

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
