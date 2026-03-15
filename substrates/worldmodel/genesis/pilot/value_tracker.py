"""Performance value tracking for EVR (Energy-to-Value Ratio) computation.

Tracks improvement per epoch and computes EVR to enable informed training decisions.
EVR = improvement / energy_consumed (higher is better).

Usage:
    tracker = ValueTracker(baseline_score=0.0183)
    stats = tracker.record_epoch(epoch=1, score=0.0079, energy={'energy_joules': 262800})
    print(f"EVR: {stats['evr']:.2e}")
    if tracker.should_stop(threshold=5e-6):
        print("Diminishing returns - consider stopping")
"""

from typing import Optional


class ValueTracker:
    """Track performance value created per epoch.

    Computes EVR (Energy-to-Value Ratio) = improvement / energy.
    Higher EVR means more efficient learning.

    Attributes:
        baseline_score: Initial score before training (optional)
        history: List of (epoch, score, improvement, energy_joules, evr) tuples
    """

    def __init__(self, baseline_score: Optional[float] = None):
        """Initialize tracker.

        Args:
            baseline_score: Score before training starts. If None, first epoch
                           establishes the baseline.
        """
        self.baseline_score = baseline_score
        self.history: list[tuple[int, float, float, float, float]] = []

    def record_epoch(self, epoch: int, score: float, energy: dict) -> dict:
        """Record epoch results and compute EVR.

        Args:
            epoch: Epoch number (1-indexed)
            score: Current performance score (lower is better for occlusion recovery)
            energy: Energy dict from EnergyTracker.end_epoch()

        Returns:
            dict with:
                - epoch: Epoch number
                - score: Current score
                - improvement: Score decrease from previous (positive = better)
                - energy_joules: Energy consumed this epoch
                - evr: Energy-to-Value Ratio (improvement per joule)
                - cumulative_energy: Total energy so far
                - cumulative_improvement: Total improvement from baseline
                - evr_trend: 'improving', 'stable', or 'declining'
        """
        energy_joules = energy.get('energy_joules', 0.0)

        # Compute improvement (positive = score decreased = better)
        if self.history:
            prev_score = self.history[-1][1]
            improvement = prev_score - score
        elif self.baseline_score is not None:
            improvement = self.baseline_score - score
        else:
            # First epoch with no baseline - this becomes the baseline
            self.baseline_score = score
            improvement = 0.0

        # EVR: improvement per joule (avoid division by zero)
        evr = improvement / energy_joules if energy_joules > 0 else 0.0

        # Store in history
        self.history.append((epoch, score, improvement, energy_joules, evr))

        # Compute cumulative stats
        cumulative_energy = sum(h[3] for h in self.history)
        cumulative_improvement = (
            self.baseline_score - score if self.baseline_score else 0.0
        )

        # Compute EVR trend
        evr_trend = self._compute_trend()

        # Efficiency vs epoch 1
        if len(self.history) > 1 and self.history[0][4] > 0:
            efficiency_ratio = evr / self.history[0][4]
        else:
            efficiency_ratio = 1.0

        return {
            'epoch': epoch,
            'score': score,
            'improvement': improvement,
            'energy_joules': energy_joules,
            'evr': evr,
            'cumulative_energy': cumulative_energy,
            'cumulative_improvement': cumulative_improvement,
            'evr_trend': evr_trend,
            'efficiency_ratio': efficiency_ratio
        }

    def _compute_trend(self) -> str:
        """Compute EVR trend from recent history."""
        if len(self.history) < 3:
            return 'stable'

        # Look at last 3 EVR values
        recent_evr = [h[4] for h in self.history[-3:]]

        # Check if consistently declining
        if recent_evr[0] > recent_evr[1] > recent_evr[2]:
            return 'declining'
        elif recent_evr[0] < recent_evr[1] < recent_evr[2]:
            return 'improving'
        else:
            return 'stable'

    def should_stop(self, evr_threshold: float = 5e-6, consecutive: int = 2) -> bool:
        """Check if training should stop due to diminishing returns.

        Args:
            evr_threshold: Stop if EVR drops below this value
            consecutive: Number of consecutive epochs below threshold

        Returns:
            True if last `consecutive` epochs have EVR below threshold
        """
        if len(self.history) < consecutive:
            return False

        recent_evr = [h[4] for h in self.history[-consecutive:]]
        return all(evr < evr_threshold for evr in recent_evr)

    def get_recommendation(self, evr_threshold: float = 5e-6) -> str:
        """Get human-readable recommendation.

        Args:
            evr_threshold: Threshold for efficiency decision

        Returns:
            Recommendation string: CONTINUE, CONSIDER_STOPPING, or STOP
        """
        if not self.history:
            return 'CONTINUE'

        current_evr = self.history[-1][4]

        if self.should_stop(evr_threshold):
            return 'STOP'
        elif current_evr < evr_threshold * 2:
            return 'CONSIDER_STOPPING'
        else:
            return 'CONTINUE'

    def get_summary(self) -> dict:
        """Get summary statistics.

        Returns:
            dict with summary of training efficiency
        """
        if not self.history:
            return {}

        total_energy = sum(h[3] for h in self.history)
        total_improvement = self.baseline_score - self.history[-1][1] if self.baseline_score else 0.0
        avg_evr = total_improvement / total_energy if total_energy > 0 else 0.0

        # Find peak EVR epoch
        peak_evr_idx = max(range(len(self.history)), key=lambda i: self.history[i][4])
        peak_evr = self.history[peak_evr_idx][4]
        peak_epoch = self.history[peak_evr_idx][0]

        return {
            'num_epochs': len(self.history),
            'final_score': self.history[-1][1],
            'baseline_score': self.baseline_score,
            'total_improvement': total_improvement,
            'total_energy_joules': total_energy,
            'total_energy_kwh': total_energy / 3_600_000,
            'average_evr': avg_evr,
            'peak_evr': peak_evr,
            'peak_evr_epoch': peak_epoch,
            'current_evr': self.history[-1][4]
        }

    def format_dashboard(self, epoch_stats: dict, energy_stats: dict) -> str:
        """Format EVR dashboard for display.

        Args:
            epoch_stats: Output from record_epoch()
            energy_stats: Output from EnergyTracker.end_epoch()

        Returns:
            Formatted string for terminal display
        """
        rec = self.get_recommendation()
        trend_bar = self._trend_bar()

        lines = [
            '',
            '+' + '=' * 62 + '+',
            '|' + f' EPOCH {epoch_stats["epoch"]} COMPLETE'.center(62) + '|',
            '+' + '=' * 62 + '+',
            '| Performance' + ' ' * 50 + '|',
            f'|   Occlusion Score: {epoch_stats["score"]:.6f}',
            f'|   Improvement: {epoch_stats["improvement"]:+.6f} from prev',
            f'|   Total Improvement: {epoch_stats["cumulative_improvement"]:.6f} from baseline',
            '+' + '-' * 62 + '+',
            '| Energy' + ' ' * 55 + '|',
            f'|   This Epoch: {energy_stats["avg_power_watts"]:.0f}W x {energy_stats["duration_sec"]:.0f}s = {energy_stats["energy_joules"]/1000:.1f} kJ',
            f'|   Cumulative: {epoch_stats["cumulative_energy"]/1000:.1f} kJ ({epoch_stats["cumulative_energy"]/3_600_000:.3f} kWh)',
            '+' + '-' * 62 + '+',
            '| Efficiency' + ' ' * 51 + '|',
            f'|   EVR This Epoch: {epoch_stats["evr"]:.2e} improvement/joule',
            f'|   EVR Trend: {trend_bar} ({epoch_stats["evr_trend"]})',
            f'|   Efficiency vs Epoch 1: {epoch_stats["efficiency_ratio"]:.2f}x',
            '+' + '-' * 62 + '+',
            f'| Decision: {rec}' + ' ' * (52 - len(rec)) + '|',
            '+' + '=' * 62 + '+',
            ''
        ]

        # Pad lines to consistent width
        formatted = []
        for line in lines:
            if line.startswith('|') and not line.endswith('|'):
                line = line + ' ' * (63 - len(line)) + '|'
            formatted.append(line)

        return '\n'.join(formatted)

    def _trend_bar(self) -> str:
        """Generate visual trend bar."""
        if len(self.history) < 2:
            return '[----]'

        # Normalize EVRs to 0-1 range for display
        evrs = [h[4] for h in self.history[-5:]]  # Last 5 epochs
        if max(evrs) == 0:
            return '[----]'

        max_evr = max(evrs)
        normalized = [e / max_evr for e in evrs]

        # Map to visual bar
        chars = []
        for n in normalized:
            if n > 0.75:
                chars.append('#')
            elif n > 0.5:
                chars.append('=')
            elif n > 0.25:
                chars.append('-')
            else:
                chars.append('.')

        return '[' + ''.join(chars).ljust(5, ' ')[:5] + ']'

    def to_dict(self) -> dict:
        """Serialize tracker state for checkpoint saving."""
        return {
            'baseline_score': self.baseline_score,
            'history': self.history
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ValueTracker':
        """Restore tracker state from checkpoint."""
        tracker = cls(baseline_score=data.get('baseline_score'))
        tracker.history = [tuple(h) for h in data.get('history', [])]
        return tracker


if __name__ == '__main__':
    # Test with simulated data matching the analysis
    print("Testing ValueTracker with smoke test data...")

    tracker = ValueTracker()

    # Simulated epoch data from the smoke test
    epochs_data = [
        (1, 0.018326, 262800),
        (2, 0.007908, 262800),
        (3, 0.005294, 262800),
        (4, 0.004437, 262800),
        (5, 0.003740, 262800),
    ]

    for epoch, score, energy_j in epochs_data:
        energy = {'energy_joules': energy_j, 'avg_power_watts': 438, 'duration_sec': 600}
        stats = tracker.record_epoch(epoch, score, energy)
        print(tracker.format_dashboard(stats, energy))

        rec = tracker.get_recommendation(evr_threshold=5e-6)
        print(f"Recommendation: {rec}\n")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    summary = tracker.get_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
