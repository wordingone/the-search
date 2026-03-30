# Kill: trajectory-conditioned action selection
Steps: 938c-938e (3 experiments) | Trigger: h→action mapping requires learning or per-obs memory, neither available

## What worked
- h provides real position-dependent variance (h_spr=0.37-0.43 consistently)
- h in ext_enc IS the trajectory discrimination signal (removing it kills delta discrimination)
- Alpha-weighted delta on ext_enc (916 formula) is irreducible

## What failed
- Raw enc delta without alpha: discrimination collapses (delta_spr→0.018)
- Alpha on enc-only delta: still no discrimination (delta_spr=0.013)
- Random projection R@h: maps position info to arbitrary action biases, corrupts softmax even at beta=0.01
- h information exists but no gate-compliant mapping h→actions

## What next family needs
- NOT an action selector variant — the action selection DOF is closed at 800b
- A structurally different substrate architecture (encoding-action-learning relationship)
- Must address a DIFFERENT bottleneck: VC33 encoding, CIFAR classification, or chain architecture
- h is useful for ENCODING (via alpha) but not for ACTION SELECTION

## Return condition
Demonstrate a gate-compliant mapping from trajectory state to action that doesn't require learned R (R1), per-obs memory (gate 5), or W prediction (killed). Currently no known method satisfies this.
