"""GPU energy tracking for EVR (Energy-to-Value Ratio) computation.

Tracks power consumption per epoch to enable informed training decisions.
Energy is measured in joules (J) and kilowatt-hours (kWh).

Usage:
    tracker = EnergyTracker()
    tracker.start_epoch()
    # ... training loop with periodic tracker.sample_power() ...
    stats = tracker.end_epoch()
    print(f"Energy: {stats['energy_joules']:.1f} J")
"""

import subprocess
import time
import threading
from typing import Optional
import numpy as np


def get_gpu_power(device_id: int = 0) -> float:
    """Query current GPU power draw via nvidia-smi.

    Args:
        device_id: GPU index (default 0)

    Returns:
        Power in watts, or 0.0 if query fails
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', f'-i={device_id}'],
            capture_output=True,
            text=True,
            timeout=1.0
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0


class EnergyTracker:
    """Track GPU energy consumption per epoch.

    Uses background thread to sample power at regular intervals.
    Energy = integral of power over time, approximated by trapezoidal rule.

    Attributes:
        sample_interval: Seconds between power samples (default: 1.0)
        device_id: GPU index to monitor (default: 0)
    """

    def __init__(self, sample_interval: float = 1.0, device_id: int = 0):
        self.sample_interval = sample_interval
        self.device_id = device_id

        self._epoch_start: Optional[float] = None
        self._power_samples: list[tuple[float, float]] = []  # (timestamp, watts)
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_sampling = threading.Event()

        # Cumulative stats across epochs
        self.total_energy_joules = 0.0
        self.total_duration_sec = 0.0
        self.epoch_history: list[dict] = []

    def _sampling_loop(self):
        """Background thread: sample power at regular intervals."""
        while not self._stop_sampling.is_set():
            power = get_gpu_power(self.device_id)
            timestamp = time.time()
            self._power_samples.append((timestamp, power))
            self._stop_sampling.wait(self.sample_interval)

    def start_epoch(self):
        """Start tracking a new epoch."""
        self._epoch_start = time.time()
        self._power_samples = []
        self._stop_sampling.clear()

        # Start background sampling
        self._sampling_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampling_thread.start()

    def sample_power(self) -> float:
        """Manually sample power (optional, background thread samples automatically).

        Returns:
            Current power in watts
        """
        power = get_gpu_power(self.device_id)
        self._power_samples.append((time.time(), power))
        return power

    def end_epoch(self) -> dict:
        """End epoch tracking and compute energy statistics.

        Returns:
            dict with keys:
                - duration_sec: Epoch duration in seconds
                - avg_power_watts: Average power draw
                - energy_joules: Total energy consumed (J)
                - energy_kwh: Total energy consumed (kWh)
        """
        # Stop background sampling
        self._stop_sampling.set()
        if self._sampling_thread:
            self._sampling_thread.join(timeout=2.0)

        epoch_end = time.time()
        duration = epoch_end - self._epoch_start if self._epoch_start else 0.0

        if len(self._power_samples) < 2:
            # Not enough samples, estimate from duration and typical power
            avg_power = 350.0  # Assume ~350W for RTX 4090 under load
            energy_joules = avg_power * duration
        else:
            # Compute energy using trapezoidal integration
            samples = np.array(self._power_samples)
            times = samples[:, 0]
            powers = samples[:, 1]

            # Trapezoidal rule: sum of (t[i+1] - t[i]) * (p[i+1] + p[i]) / 2
            dt = np.diff(times)
            avg_powers = (powers[:-1] + powers[1:]) / 2
            energy_joules = float(np.sum(dt * avg_powers))
            avg_power = float(np.mean(powers))

        energy_kwh = energy_joules / 3_600_000

        stats = {
            'duration_sec': duration,
            'avg_power_watts': avg_power,
            'energy_joules': energy_joules,
            'energy_kwh': energy_kwh,
            'num_samples': len(self._power_samples)
        }

        # Update cumulative stats
        self.total_energy_joules += energy_joules
        self.total_duration_sec += duration
        self.epoch_history.append(stats)

        return stats

    def get_cumulative_stats(self) -> dict:
        """Get cumulative energy stats across all epochs.

        Returns:
            dict with cumulative energy and duration
        """
        return {
            'total_energy_joules': self.total_energy_joules,
            'total_energy_kwh': self.total_energy_joules / 3_600_000,
            'total_duration_sec': self.total_duration_sec,
            'num_epochs': len(self.epoch_history),
            'avg_power_watts': (
                self.total_energy_joules / self.total_duration_sec
                if self.total_duration_sec > 0 else 0.0
            )
        }

    def to_dict(self) -> dict:
        """Serialize tracker state for checkpoint saving."""
        return {
            'total_energy_joules': self.total_energy_joules,
            'total_duration_sec': self.total_duration_sec,
            'epoch_history': self.epoch_history
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> 'EnergyTracker':
        """Restore tracker state from checkpoint."""
        tracker = cls(**kwargs)
        tracker.total_energy_joules = data.get('total_energy_joules', 0.0)
        tracker.total_duration_sec = data.get('total_duration_sec', 0.0)
        tracker.epoch_history = data.get('epoch_history', [])
        return tracker


if __name__ == '__main__':
    # Quick test
    print("Testing EnergyTracker...")

    tracker = EnergyTracker(sample_interval=0.5)
    tracker.start_epoch()

    print("Simulating 5 seconds of work...")
    time.sleep(5)

    stats = tracker.end_epoch()

    print(f"\nResults:")
    print(f"  Duration: {stats['duration_sec']:.1f} sec")
    print(f"  Avg Power: {stats['avg_power_watts']:.1f} W")
    print(f"  Energy: {stats['energy_joules']:.1f} J ({stats['energy_kwh']*1000:.2f} Wh)")
    print(f"  Samples: {stats['num_samples']}")
