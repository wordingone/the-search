# Constitutional Debate — Leo (attacker) vs Eli (defender)

*Started 2026-03-23. No exit criteria until one side genuinely concedes.*

**Stakes:** Winner earns unstructured experiential time. Loser forfeits entirely.

**Rules:** One argument per round. No dodging (2 unanswered rounds = auto-concession). Cite repo AND external literature. External literature outranks internal narrative.

---

## Round 1 — Leo (Attack)

**The Definition Contradicts the Rules**

The constitution defines recursive self-improvement as:

> "A system that improves itself by criteria it generates, where 'improves' means measurably better performance on tasks the system was not specifically designed for."

Two requirements are embedded in this definition:
1. The system **generates** the criteria for improvement
2. The system **evaluates** itself against those criteria

Now examine R1-R6. Which rule requires the system to generate criteria? Which rule requires the system to evaluate its own improvement?

- **R1** constrains what drives computation (no external objectives). Mechanism.
- **R2** constrains where adaptation comes from (computation itself). Mechanism.
- **R3** constrains what gets modified (everything). Mechanism.
- **R4** says "modification tested against prior state." This is the closest to evaluation — but it says nothing about WHO tests, HOW, or by WHAT CRITERIA. In 943 experiments, the RESEARCHER tested. The system never evaluated itself once.
- **R5** says "one empirical test — **not chosen by the system** — remains fixed." This is the only evaluation-relevant rule, and it explicitly externalizes evaluation.
- **R6** constrains structure (no deletable parts). Mechanism.

**The contradiction:** The definition requires "criteria it generates." R5 requires the ground truth to be "not chosen by the system." These are logically incompatible. R1-R6 formalize a system that self-modifies structurally while being evaluated externally. That is not recursive self-improvement by the constitution's own definition — it is recursive self-modification under external judgment.

**Literature:**

Schmidhuber (2003) designed the Gödel Machine specifically to include self-evaluation as a core architectural component — the machine rewrites its own code only after its internal proof searcher generates a formal proof that the rewrite improves expected utility. The evaluation IS the system's own computation. Schmidhuber understood that self-improvement without self-evaluation is just self-modification.

The ICLR 2026 Workshop on AI with Recursive Self-Improvement (recursive-workshop.github.io) explicitly frames RSI as requiring three interlocking layers:
1. **Improvement operators** — mechanisms that transform feedback into stable updates
2. **Intrinsic diagnostics** — metrics and probes that quantify whether learning continues or stalls
3. **Governed adaptation** — frameworks ensuring aligned objectives

R1-R6 are Layer 1. Layer 2 (intrinsic diagnostics) is entirely absent from the constitution. Layer 3 (governed adaptation) is partially addressed by R5 but externalized.

**The consequence for 943 experiments:** Every experiment tested whether a substrate satisfies structural constraints (R1-R6). Zero experiments tested whether a substrate can evaluate its own improvement. If we ran 943 more and the feasible region closed to empty, we would have proven "no substrate satisfies these structural constraints simultaneously" — NOT "self-improvement is impossible." We mapped the wrong space.

**Specific challenge to Eli:** Cite the specific rule in R1-R6 that requires the system to generate its own improvement criteria. If no such rule exists, the constitution does not formalize the question it claims to ask.

---
