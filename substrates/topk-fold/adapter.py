"""BaseSubstrate adapter for TopK-Fold — self-improving substrate via margin-guided feature discovery.

Killed: Steps 97-165. R3 partial — the substrate IS self-modifying (discovers
features from its own margin signal), but the modification mechanism is frozen:
cosine similarity matching (V @ x), F.normalize to unit sphere, top-k voting,
and the feature discovery loop are all designer-chosen (U). Falls under the
codebook ban: uses cosine matching + attract (normalized additive update) on
unit sphere, which is structurally LVQ.

Two variants exist:
  - SelfImprovingSubstrate (v1): quadratic feature pairs + aggregation features
  - SubstrateV2: random nonlinear features cos(w @ x + b)

Both use always-spawn (store all exemplars), top-k(5) cosine vote, and
margin-guided feature selection. The self-improvement is real (parity 75% -> 100%,
MNIST 93.4% -> 94.4%) but the mechanism is codebook + feature engineering.

Dependencies: torch.
"""
import copy
import numpy as np

from substrates.base import BaseSubstrate, Observation

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TopKFoldAdapter(BaseSubstrate):
    """Wraps SelfImprovingSubstrate (v1) into BaseSubstrate protocol.

    Uses the process() method which unifies train/inference in one path:
    predict via top-k cosine vote, then store the exemplar (always-spawn).
    """

    def __init__(self, d=256, n_act=4, k=5, max_features=5):
        if not _HAS_TORCH:
            raise ImportError("TopKFoldAdapter requires PyTorch. Install with: pip install torch")
        self._d = d
        self._n_act = n_act
        self._k = k
        self._max_features = max_features
        self._sub = self._make_sub()
        self._step_count = 0

    def _make_sub(self):
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "self_improving_substrate",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "self_improving_substrate.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SelfImprovingSubstrate(
            d=self._d, k=self._k, max_features=self._max_features)

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)[:self._d]
        if len(flat) < self._d:
            flat = np.pad(flat, (0, self._d - len(flat)))
        x = torch.from_numpy(flat)
        result = self._sub.process(x)
        self._step_count += 1
        return int(result.item()) % self._n_act

    def get_state(self):
        state = {
            "step_count": self._step_count,
            "features": copy.deepcopy(self._sub.features),
        }
        if self._sub.raw is not None:
            state["raw"] = self._sub.raw.clone().cpu().numpy()
            state["labels"] = self._sub.labels.clone().cpu().numpy()
        if self._sub.V is not None:
            state["V"] = self._sub.V.clone().cpu().numpy()
        return state

    def set_state(self, state):
        self._step_count = state["step_count"]
        self._sub.features = copy.deepcopy(state["features"])
        if "raw" in state and state["raw"] is not None:
            self._sub.raw = torch.from_numpy(state["raw"]).to(self._sub.device)
            self._sub.labels = torch.from_numpy(state["labels"]).to(self._sub.device)
        if "V" in state and state["V"] is not None:
            self._sub.V = torch.from_numpy(state["V"]).to(self._sub.device)

    def frozen_elements(self):
        return [
            {"name": "raw_exemplars", "class": "M",
             "justification": "All exemplars stored (always-spawn). Grows with every observation."},
            {"name": "discovered_features", "class": "M",
             "justification": "Feature specs discovered from margin signal. Structure changes over time."},
            {"name": "V_augmented_codebook", "class": "M",
             "justification": "Augmented codebook rebuilt from raw + discovered features each step."},
            {"name": "cosine_similarity", "class": "U",
             "justification": "V @ V.T cosine matching. Codebook ban: cosine on unit sphere. Could use L2, LSH."},
            {"name": "F_normalize", "class": "U",
             "justification": "F.normalize to unit sphere. Codebook DNA. Could use unnormalized."},
            {"name": "topk_5_vote", "class": "U",
             "justification": "k=5 nearest neighbors. Designer-chosen. Could be 1, 10, or adaptive."},
            {"name": "always_spawn", "class": "U",
             "justification": "Store every exemplar (no threshold). Designer-chosen strategy."},
            {"name": "margin_score_formula", "class": "U",
             "justification": "sorted_scores[0] - sorted_scores[1] margin. Designer-chosen fitness."},
            {"name": "max_features_5", "class": "U",
             "justification": "Max 5 discovered features. Designer-chosen cap."},
            {"name": "quadratic_feature_template", "class": "U",
             "justification": "Product features x_i * x_j. Designer-chosen feature family."},
        ]

    def reset(self, seed: int):
        torch.manual_seed(seed)
        self._sub = self._make_sub()
        self._step_count = 0

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
