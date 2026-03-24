# Kill: Hebbian W_a Action Learning
Steps: 948-960 (13 experiments) | Trigger: 1/10 structural rate unbreakable after 13 variants

## What worked
- Continuous recurrent state h encodes trajectory (seed 8 navigates, 96-150 L1)
- Hebbian W_a learning CAN drive action selection in principle
- Concurrent learning from step 1 is load-bearing (Step 950)
- Positive accumulation is load-bearing (Step 951, cross-family with ESN Step 954)
- Architecture is irrelevant — Proposition 29 (Step 955: ESN = 948 numerically)

## What failed
- 1/10 structural bootstrap rate, architecture-independent and RNG-independent
- Root cause: Proposition 30 (positive lock) refined to h CORRELATION
- Sigmoid h produces correlated representations across states
- Any linear scorer (W_a @ h) gives consistent action ranking → winner-take-all
- 13 variants tested and killed:
  - Architecture: h_dim (949), ESN/sigmoid (955), ESN/tanh (954) — irrelevant
  - Learning: warm-start (950), mean-subtraction (951), epsilon-decay (952) — kills signal
  - Selection: softmax (953), RNG-free hash (956), UCB (957) — same or worse rate
  - Population: ensemble K=5 (958) — kills lucky seeds (0/10)
  - Representation: sparse ReLU (959), signed scoring (960) — lock persists

## What next family needs
- Either: accept 800b as fixed action mechanism, improve encoding/prediction
- Or: non-linear action scoring (distance-based, attention-style) that breaks correlation lock
- Or: decorrelated representations (competitive inhibition, anti-Hebbian normalization)
- NOT: any linear scorer on sigmoid/positive representations

## Return condition
If a mechanism is found that produces state-specific (uncorrelated) representations from recurrent dynamics, Hebbian W_a could be revisited. But 13 experiments prove the lock is in the h correlation structure, not in any individual component.
