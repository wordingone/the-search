"""
TEMPLATE — ALL new experiments MUST use this template.
Jun directive 2026-03-24: every experiment runs through PRISM ChainRunner.
No standalone single-game scripts. Game order randomized. Game names hidden.

Copy this file, rename to step{NNNN}_{name}.py, fill in the substrate class.

FAMILY: [name]
R3 HYPOTHESIS: [what R3 prediction does this test?]
KILL: chain_score < [threshold] OR games_with_signal < [N]
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
import torch
from substrates.step0674 import _enc_frame
from substrates.chain import ChainRunner, ArcGameWrapper

# ─── Your substrate class here ───
class MySubstrate:
    """
    Replace with your substrate. Must implement:
      - process(obs) -> action_index
      - set_game(game_name)  [called on game switch]
    """
    def __init__(self):
        pass

    def process(self, obs):
        # Return action index
        return np.random.randint(4)

    def set_game(self, game_name):
        pass


# ─── PRISM chain setup (DO NOT MODIFY) ───
N_STEPS = 10_000  # per game
N_SEEDS = 10

chain = [
    ("CIFAR-1", ...),  # TODO: add CIFAR wrapper when available
    ("LS20", ArcGameWrapper("LS20", n_steps=N_STEPS)),
    ("FT09", ArcGameWrapper("FT09", n_steps=N_STEPS)),
    ("VC33", ArcGameWrapper("VC33", n_steps=N_STEPS)),
    # ("CIFAR-2", ...),  # TODO: add CIFAR wrapper
]

# Remove placeholders (CIFAR not yet wrapped)
chain = [(n, w) for n, w in chain if w is not ...]

runner = ChainRunner(
    chain=chain,
    n_seeds=N_SEEDS,
    randomize_order=True,  # ENFORCED — cannot be False
)

results = runner.run(MySubstrate)

# Chain score is the PRIMARY metric
print(f"\nChain score: {results['_chain_score']:.3f}")
print(f"Games with signal: {results['_games_with_signal']}/{results['_total_games']}")
