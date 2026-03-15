"""
Anima Environment Module
========================

Structured environments for testing Anima entities.
"""

import numpy as np
from typing import Optional


class AnimaEnvironment:
    """
    Structured environment for Anima.

    Provides |E| = Ω distinct environmental states via:
    - Multi-frequency sinusoidal base patterns (predictable structure)
    - Gaussian noise overlay (stochastic variation)
    - Perturbation mechanism (phase shifts, noise increase)

    This tests Anima's capacity to develop adaptive responses
    across the environmental variety space.
    """

    def __init__(self, sensory_dim: int = 8):
        self.dim = sensory_dim
        self.t = 0
        self.frequencies = np.random.uniform(0.01, 0.1, sensory_dim)
        self.phases = np.random.uniform(0, 2 * np.pi, sensory_dim)
        self.amplitudes = np.random.uniform(0.3, 1.0, sensory_dim)
        self.noise_level = 0.1

    def step(self) -> np.ndarray:
        """Generate next observation."""
        self.t += 1

        # Base pattern (predictable)
        pattern = self.amplitudes * np.sin(self.frequencies * self.t + self.phases)

        # Add noise
        noise = np.random.randn(self.dim) * self.noise_level

        return pattern + noise

    def perturb(self, magnitude: float):
        """Perturb the environment (test adaptation)."""
        self.phases = np.random.uniform(0, 2 * np.pi, self.dim)
        self.noise_level = min(0.5, self.noise_level + magnitude * 0.2)

    def reset(self):
        """Reset environment to initial state."""
        self.t = 0
        self.frequencies = np.random.uniform(0.01, 0.1, self.dim)
        self.phases = np.random.uniform(0, 2 * np.pi, self.dim)
        self.amplitudes = np.random.uniform(0.3, 1.0, self.dim)
        self.noise_level = 0.1
