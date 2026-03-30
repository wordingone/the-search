# Kill: Oscillatory Substrate (Stuart-Landau)
Steps: 1001-1003 | Trigger: 3 consecutive kills, LS20 9.5 vs 72.7 threshold

## What worked
- Phase diversity from input: 1.5-1.8 (oscillators respond to observations)
- 800b action mechanism functions with oscillatory features (3/10 navigate in 1003)
- Stuart-Landau dynamics stable with magnitude clipping

## What failed
- **1001:** Compression-progress modulator (M = prediction improvement) uncorrelated with game events. M stagnates when W_pred converges (8/10 seeds). Even when M fires (2/10), no navigation.
- **1002:** Observation-change modulator (M = above-average delta_obs) fires correctly but W_out learning accumulates uncorrelated signal. No positive feedback loop — all actions produce similar average observation change in random exploration.
- **1003:** Hybrid (oscillatory encoding + 800b action): 9.5/seed, 3/10 nonzero vs 916 baseline 72.7/seed. 87% gap. Random W_in produces insufficient encoding quality. 916's learned h (via W_h/W_x trained by prediction error) vastly outperforms random oscillatory features.

## What next family needs
- Action selection must be 800b (argmax amplification is the only working mechanism)
- Encoding must be LEARNED, not random — oscillatory features from random W_in ≈ noise
- Temporal credit for sequential actions cannot come from phase-coherence gating alone
- Any credit mechanism needs a CONTRAST signal (differential between productive and unproductive actions), not just an activation signal (fire on change)

## Return condition
If a mechanism is found that trains W_in from interaction (making oscillatory features game-relevant), the oscillatory encoding could be revisited. But 916's simpler recurrence already achieves this.
