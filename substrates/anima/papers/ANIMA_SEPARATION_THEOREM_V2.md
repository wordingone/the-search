# The Computational Cost of Treating Context as Static:

## A Separation Theorem for State-Conditioned Routing in Sequence Models

---

## Abstract

Many modern sequence-processing architectures—transformers, state space models, recurrent networks, and their hybrids—share a common structural constraint: the *rules of computation* applied at each step are selected independently of the accumulated internal state. In this paper, we formalize this constraint and show that it has unavoidable computational consequences.

We distinguish **reactive systems**, in which step-wise computational parameters depend only on the current input token, from **state-conditioned systems**, in which parameters may depend jointly on accumulated state and current input. For a class of sequence tasks requiring *k*-way context-dependent routing, we prove a representational separation: any reactive system requires Ω(k) state, while a state-conditioned system can solve the same task with O(log k) state, at the cost of O(k·d²) parameters.

The lower bound follows from a reduction to the one-way INDEX problem in communication complexity under a finite-precision, robustness-preserving computation model. The upper bound is constructive. Our result applies at the level of representational capacity, independent of training dynamics or optimization. It isolates a precise trade-off between state size, parameterization, and control, and provides a theoretical explanation for why certain temporal routing tasks remain costly for architectures whose parameter selection is state-independent.

---

## 1. Introduction

### 1.1 Static Parameter Selection as a Structural Constraint

Despite substantial architectural diversity, most high-performing sequence models share a common feature: while their *state* evolves over time, the *selection of computational parameters* applied at each step is fixed or depends only on the current input token.

This is true, in distinct ways, of:

* **Transformers**, where attention weights are computed using fixed parameter matrices and the query token, with positional information injected additively rather than through state-conditioned control.
* **Structured state space models**, where discretization and input projections are functions of the current input but not of the accumulated state.
* **Memory-augmented models**, where retrieval is keyed by embedding similarity rather than by temporal or causal position.
* **Parallelizable sequence architectures**, where step-wise parameter selection must be independent across time for efficiency.

In contrast, classical gated recurrent models condition their update rules on both prior state and current input, but at the cost of strict sequential dependence that limits parallel training.

This observation motivates a precise question:

> What is the representational cost of restricting parameter selection to be independent of accumulated state?

---

### 1.2 Reactive vs. State-Conditioned Computation

We formalize this question by distinguishing two classes of sequence transducers:

* **Reactive systems**, in which the parameters governing each computational step are functions only of the current input token.
* **State-conditioned systems**, in which those parameters may depend on both the current input and a projection of accumulated state.

This distinction is orthogonal to model family (attention, recurrence, state space) and concerns *how computation is selected*, not how it is executed.

The central claim of this paper is that this distinction is not merely architectural or stylistic. It induces a provable separation in representational efficiency for a natural class of temporal routing tasks.

---

### 1.3 Contribution

This paper makes four contributions:

1. **Formal definitions** of reactive and state-conditioned sequence transducers, stated independently of any specific architecture.
2. **A separation theorem**: for a k-way context-dependent routing task, reactive systems require Ω(k) state, while state-conditioned systems require only O(log k) state.
3. **A lower-bound proof** via reduction from the INDEX problem in one-way communication complexity, under explicit finite-precision and robustness assumptions.
4. **A constructive upper bound**, demonstrating that the separation is achievable by a minimal state-conditioned transducer.

The result is representational rather than algorithmic: it characterizes what can be represented with given resources, not what is efficiently learned.

---

## 2. Formal Framework

### 2.1 Sequence Transducers

A **sequence transducer** is a tuple
T = (Σ, Γ, S, Θ, f, g), where:

* Σ is the input alphabet
* Γ is the output alphabet
* S is the state space with initial state s₀
* Θ is the parameter space
* f: S × Σ × Θ → S is the state transition
* g: S × Σ × Θ → Γ is the output function

Given an input sequence (x₁, …, xₙ), the transducer evolves as:

[
s_t = f(s_{t-1}, x_t, \theta_t), \quad
y_t = g(s_{t-1}, x_t, \theta_t).
]

We assume finite-precision computation: states are represented with p bits per coordinate, and correctness is required under constant-margin decoding.

---

### 2.2 Parameter Selection

The distinction of interest concerns how θₜ is selected.

**Definition 1 (Reactive System).**
A transducer is *reactive* if there exists a function G: Σ → Θ such that
[
\theta_t = G(x_t) \quad \text{for all } t.
]

**Definition 2 (State-Conditioned System).**
A transducer is *state-conditioned* if there exists a projection φ: S → ℝᵈ and a function G: ℝᵈ × Σ → Θ such that
[
\theta_t = G(\phi(s_{t-1}), x_t).
]

Reactive systems are a strict subclass of state-conditioned systems.

---

### 2.3 Interpretation

Reactive systems may perform arbitrarily complex nonlinear updates inside f and g, but the *choice of computational rule* applied at each step is fixed once xₜ is known. Any dependence on past context must be encoded explicitly in the state and decoded uniformly.

State-conditioned systems may instead use the accumulated state to *select* among different computational regimes.

---

## 3. The Routing Task

### 3.1 k-Way Context-Dependent Routing

Fix k ≥ 2. Let the vocabulary contain marker tokens {m₁, …, m_k}, a query token q, and a null symbol ⊥.

**Task (k-Marker Routing).**

* The input sequence contains exactly one marker mᵢ at an arbitrary position, followed later by a query token q.
* All other tokens are padding.
* The output is ⊥ at all steps except at q, where the correct output is i.

The task requires detecting which of k alternatives occurred in the past and routing the query accordingly.

---

## 4. Lower Bound via Communication Complexity

### 4.1 INDEX Problem

In the one-way INDEX(k) problem:

* Alice holds a bit vector s ∈ {0,1}ᵏ.
* Bob holds an index i ∈ [k].
* Alice sends a single message to Bob, who must output sᵢ.

Any one-way protocol with constant success probability requires Ω(k) bits of communication.

---

### 4.2 Reduction

Assume a reactive transducer solves k-Marker Routing with state dimension N.

Alice encodes her string s by presenting marker mⱼ iff sⱼ = 1, producing a final state h_A. She sends h_A to Bob.

Bob initializes the transducer with h_A and processes q. Because the system is reactive, the parameters used at q depend only on q, which is fixed. Thus, Bob’s output depends solely on h_A.

Correct decoding for all i requires h_A to encode Ω(k) bits of information. Under finite-precision assumptions, this implies:

[
d \cdot N \cdot p = \Omega(k),
]

and hence N = Ω(k) for constant d and p.

---

### 4.3 Consequence

The lower bound is independent of parameter count or internal nonlinearity. It arises from the restriction that parameter selection at query time cannot depend on accumulated state.

---

## 5. Upper Bound Construction

### 5.1 Idea

A state-conditioned system can store the marker index in O(log k) state and use that state to *select* the appropriate routing parameters at query time.

Routing occurs in parameter space rather than state space.

---

### 5.2 Construction

Let the state consist of:

* ⌈log₂ k⌉ bits encoding the marker index,
* a constant number of auxiliary flags.

When a marker mᵢ is observed, the system writes the binary encoding of i into state.

At the query token q, the parameter-selection function G reads the encoded index and activates a corresponding decoder block that outputs i.

This requires O(k·d²) parameters but only O(log k) state.

---

### 5.3 Result

There exists a state-conditioned transducer solving k-Marker Routing with state dimension O(log k).

---

## 6. Separation Theorem

**Theorem.**
For the k-Marker Routing task:

* Any reactive system requires Ω(k) state.
* There exists a state-conditioned system requiring O(log k) state.

Thus, state-conditioned systems are exponentially more state-efficient on this task class.

---

## 7. Implications

### 7.1 Trade-Off

The separation exposes a fundamental trade-off:

* **Reactive computation** favors parallelism and uniform control but incurs linear state cost for routing.
* **State-conditioned computation** enables logarithmic state at the cost of increased parameterization and sequential dependence in control.

---

### 7.2 Scope

The result applies to tasks that require routing or rule selection based on past context. It does not claim that all sequence tasks benefit from state-conditioned parameterization.

---

## 8. Discussion and Future Directions

This work isolates a representational phenomenon: restricting parameter selection to be state-independent forces certain routing information to be stored explicitly in state, incurring linear cost.

Future work includes:

* identifying natural tasks reducible to k-way routing,
* analyzing learnability under gradient-based optimization,
* characterizing intermediate models that interpolate between reactive and fully state-conditioned control.

---

## 9. Conclusion

We proved a separation between reactive and state-conditioned sequence transducers for context-dependent routing. The result formalizes a precise computational cost of treating context as static at the level of parameter selection, and clarifies an underexplored axis in the design of sequence models.
