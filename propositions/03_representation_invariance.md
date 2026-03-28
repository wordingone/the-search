# Proposition 3: Representation Invariance
Status: CONFIRMED (NARROW) — does NOT close action selection
Steps: 521, 524, 525

## Statement
Let R: H -> R^|A| be a representation mapping transition history to a per-action summary vector, and let g(s) = argmin_a R(H)_a. If R is *count-monotone* (N(s,a) > N(s,a') => R(H)_a > R(H)_a'), then g produces identical action sequences regardless of the specific form of R.

## Evidence
Four architecturally distinct representations tested on LS20: LSH graph (6/10, Step 459), Hebbian weights (5/5, Step 524), Markov tensor (8/10, Step 525), N-gram history (4/5, Step 521). All converge to argmin over visit frequency. Mathematically trivial (argmin over orderings is order-invariant).

## Scope Limitation (revised 2026-03-28)

This proposition proves that WITHIN count-monotone approaches, all representations converge to argmin. It does NOT prove:
- That argmin is optimal for action selection
- That action selection is "solved"
- That non-count-based selectors are inferior

The original implication ("action selection for navigation is solved") was over-broad. It closed the entire action selection search space by defining the answer space to contain only count-based approaches. This contributed to 30+ experiments iterating on argmin variants instead of exploring reflexive-map-driven selection (Steps 1251-1290).

All count-based selectors (including argmin) are separate evaluators that violate R2. Proposition 3 characterizes a family of R2-violating approaches, not the solution space.

## Supersedes / Superseded by
Superseded in scope by: Steps 1289-1291 finding (three selectors, same W_action, same failure → training signal is the bottleneck, not the selector).
