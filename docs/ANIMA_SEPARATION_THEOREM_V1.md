# The Computational Cost of Treating Context as Static: A Separation Theorem for Temporal Routing

---

## Abstract

Modern machine learning architectures—transformers, state space models, recurrent networks, and their hybrids—share a structural assumption: context is processed as a collection of features with spatial or relational structure, not as a temporally-ordered causal sequence. We formalize this distinction between **reactive** systems (where computational parameters depend only on current input) and **intentional** systems (where parameters depend on both accumulated state and current input). We prove a separation theorem: for tasks requiring *k*-way context-dependent routing, reactive systems require state dimension Ω(*k*) while intentional systems achieve O(log *k*) state with O(*k* · *d*²) parameters. The proof reduces from the INDEX problem in communication complexity. This separation applies universally—to all sequence-processing architectures—and explains why certain temporal reasoning tasks remain difficult despite architectural advances. We present a minimal architecture satisfying the intentional criterion as a constructive proof of the upper bound. This paper makes no empirical claims; our contribution is purely theoretical, establishing a fundamental computational trade-off.

**Keywords:** sequence modeling, state complexity, communication complexity, temporal reasoning, intentional computation

---

## 1. Introduction

### 1.1 A Shared Blindspot

A survey of recent advances in machine learning reveals a striking pattern. Consider:

**Transformers** (Vaswani et al., 2017) process sequences through attention, where each position attends to all others weighted by learned relevance. The attention mechanism treats the context as a *set*—the same output results regardless of the order in which key-value pairs are processed. Position encodings add sequential information, but as an annotation on otherwise permutation-equivariant computation.

**State space models** (Gu et al., 2022; Gu & Dao, 2023) process sequences through continuous-time dynamics with input-dependent discretization. The parameters controlling this discretization—the step size Δ, the input projection B—depend only on the current input token. The accumulated state *h* affects *what* is computed, but not *how* parameters are selected.

**Memory-augmented architectures** (Graves et al., 2014; Borgeaud et al., 2022) add explicit memory banks to neural networks. These memories are indexed by learned keys—spatial retrieval based on embedding similarity. When information was stored relative to other events plays no role.

**Recurrent networks with gating** (Cho et al., 2014) condition gate activations on both previous hidden state and current input. This is closer to our notion of intentional computation, but the sequential recurrence prevents efficient parallel training, limiting practical deployment.

This pattern constitutes a **shared blindspot**: the treatment of context as spatial/relational rather than temporal/causal.

### 1.2 Why This Matters

The blindspot is consequential because:

1. **Language unfolds causally.** "The king died and then the queen died" ≠ "The queen died and then the king died." Attention mechanisms equipped only with position encodings cannot inherently distinguish these—the distinction must be learned from data, not encoded structurally.

2. **Memory requires temporal indexing.** Human memory is not just "what happened" but "when it happened relative to other events." Spatial retrieval (nearest neighbor in embedding space) cannot recover temporal structure discarded at write time.

3. **Reasoning has step-dependencies.** Mathematical proofs, code execution, and physical simulation follow causal chains. Each step depends not just on what came before, but on *when* it came before. Treating reasoning as sampling from a static distribution—as reinforcement learning from verification rewards does—cannot create temporal dependencies absent from the base model.

4. **Parameter selection should depend on context.** The optimal way to process the current input often depends on what has been accumulated. A question about a previously mentioned entity requires different processing than the same question in isolation.

### 1.3 Contribution

We formalize the temporal-causal distinction and prove it has computational consequences:

**Definition (Informal).** A system is *reactive* if its computational parameters depend only on current input. A system is *intentional* if its parameters depend on both accumulated state and current input.

**Theorem (Informal).** There exist tasks where reactive systems require exponentially more state than intentional systems to achieve the same computational capacity.

This is a **representational separation**—a statement about what these systems can represent with given resources, not about what gradient descent finds. The proof uses standard techniques from communication complexity, specifically reduction from the INDEX problem.

We present a minimal architecture satisfying the intentional criterion as a constructive proof that the upper bound is achievable. This architecture is not offered as a practical system; it is the simplest instantiation demonstrating the separation.

### 1.4 Scope and Limitations (Preview)

We state upfront what this paper does and does not show:

**We prove:** A representational separation between reactive and intentional systems for a specific task class (context-dependent routing).

**We do not prove:** That gradient descent finds the representations. That the task class is representative of natural tasks. That the minimal architecture is practical. That any specific existing architecture should be modified.

These limitations are not weaknesses hidden in an appendix; they define the scope of a theoretical contribution.

---

## 2. Formal Framework

### 2.1 Sequence Transducers

**Definition 1 (Sequence Transducer).** A sequence transducer is a tuple T = (Σ, Γ, S, θ, f, g) where:
- Σ is the input alphabet
- Γ is the output alphabet  
- S is the state space with distinguished initial state s₀
- θ is the parameter space
- f: S × Σ × θ → S is the state transition function
- g: S × Σ × θ → Γ is the output function

The transducer processes an input sequence (x₁, x₂, ..., xₙ) ∈ Σ* by iterating:

    sₜ = f(sₜ₋₁, xₜ, θₜ)
    yₜ = g(sₜ, xₜ, θₜ)

where θₜ are the parameters used at step t.

**Definition 2 (State Dimension).** For S ⊆ ℝᵈˣᴺ, the state dimension is N. The parameter count is |θ|.

### 2.2 Reactive and Intentional Systems

The key distinction concerns how θₜ is determined.

**Definition 3 (Reactive System).** A sequence transducer is *reactive* if there exists a function G: Σ → θ such that θₜ = G(xₜ) for all t. The parameters depend only on the current input.

**Definition 4 (Intentional System).** A sequence transducer is *intentional* if there exists a function G: S × Σ → θ such that θₜ = G(sₜ₋₁, xₜ). The parameters depend on both accumulated state and current input.

**Remark 1.** Every reactive system is trivially intentional (G ignores its first argument). The interesting question is whether intentionality provides computational advantages.

**Remark 2.** In practice, we often use the *output projection* of state rather than raw state. Let φ: S → ℝᵈ be a projection. An *output-intentional* system has θₜ = G(φ(sₜ₋₁), xₜ). This preserves the essential property: parameters depend on accumulated computation.

### 2.3 Mapping to Existing Architectures

| Architecture | Classification | Parameter Dependence |
|--------------|---------------|---------------------|
| Transformer (standard) | Reactive | θₜ = θ (fixed) |
| Mamba (Gu & Dao, 2023) | Reactive | Δₜ, Bₜ, Cₜ = f(xₜ) |
| S4 (Gu et al., 2022) | Reactive | θₜ = θ (fixed) |
| GRU (Cho et al., 2014) | Intentional | zₜ, rₜ = σ(W[hₜ₋₁; xₜ]) |
| LSTM (Hochreiter, 1997) | Intentional | gates = σ(W[hₜ₋₁; xₜ]) |

The observation that Mamba's input-dependent parameters depend only on xₜ—not on accumulated state hₜ₋₁—places it in the reactive category despite its "selective" mechanism.

### 2.4 State Complexity

**Definition 5 (Task).** A task T is a function T: Σ* → Γ* specifying the correct output for each input sequence.

**Definition 6 (State Complexity).** For a task T:
- The *reactive state complexity* Nᵣ(T) is the minimum state dimension N such that some reactive system solves T.
- The *intentional state complexity* Nᵢ(T) is the minimum state dimension N such that some intentional system solves T.

**Definition 7 (Intentional Task).** A task T is *k-intentional* if Nᵣ(T) ≥ k while Nᵢ(T) = O(log k). A task is *strictly intentional* if Nᵣ(T) = ω(Nᵢ(T)).

---

## 3. The INDEX Problem

Our lower bound argument uses a foundational result from communication complexity.

**Definition 8 (INDEX Problem).** In INDEX(k):
- Alice holds a string s ∈ {0,1}ᵏ
- Bob holds an index i ∈ [k]
- Communication is one-way: Alice sends a message m to Bob
- Bob must output sᵢ

**Theorem 1 (Ablayev, 1996).** Any one-way protocol for INDEX(k) requires Ω(k) bits of communication.

This lower bound is information-theoretic: Alice's message must encode enough information for Bob to decode any of k bits based on his private index. No encoding scheme can compress this below k bits in the worst case.

---

## 4. The Separation Theorem

### 4.1 The Task: k-Way Context-Dependent Routing

**Definition 9 (k-Marker Task).** Fix k ≥ 2. The k-Marker task operates on vocabulary V = {m₁, ..., mₖ, q, ⊥, 1, ..., k} ∪ {padding tokens}.

**Input:** A sequence containing:
1. Exactly one marker mᵢ for some i ∈ [k] at an arbitrary position
2. Zero or more padding tokens
3. A query token q

**Output:** When q is processed, output i. For all other tokens, output ⊥.

This task requires:
1. Detecting which marker appeared (k possibilities)
2. Storing this information until the query
3. Routing to the correct output based on stored information

### 4.2 Statement

**Theorem 2 (Separation Theorem).** For the k-Marker task:

**(a) Lower bound:** Any reactive system solving k-Marker requires state dimension Nᵣ ≥ k/(d · p), where d is the output dimension and p is the numerical precision (bits per coordinate). For constant d and p, Nᵣ = Ω(k).

**(b) Upper bound:** There exists an intentional system solving k-Marker with state dimension Nᵢ = ⌈log₂ k⌉ + 2 = O(log k) and parameter count O(k · d²).

**Corollary.** The k-Marker task is strictly intentional: Nᵣ(k-Marker) / Nᵢ(k-Marker) = Ω(k / log k).

### 4.3 Proof of Lower Bound

We reduce from INDEX(k).

**Construction:** Given an INDEX instance where Alice has s ∈ {0,1}ᵏ and Bob has i ∈ [k]:

1. **Alice's encoding:** Alice constructs an input sequence for the k-Marker task. For each j ∈ [k], if sⱼ = 1, she includes marker mⱼ. She processes this sequence through the reactive transducer, obtaining final state hₐ.

2. **Communication:** Alice sends hₐ to Bob. This is the one-way communication.

3. **Bob's decoding:** Bob initializes the transducer with state hₐ. He processes the query token q, obtaining output y.

**Analysis for reactive systems:**

At Bob's query processing step:
- The parameters θ_q = G(q) depend only on q, which is fixed and known to both parties
- The only information Bob receives about Alice's string is encoded in hₐ
- The output function g(hₐ, q, θ_q) is a function of hₐ alone (since q and θ_q are fixed)

For Bob to correctly determine sᵢ (whether marker mᵢ appeared in Alice's sequence), the output must distinguish between all 2ᵏ possible values of s.

**Information-theoretic argument:**
- The state hₐ ∈ ℝᵈˣᴺ encodes at most d · N · p bits with precision p
- The output g(hₐ, q, θ_q) must take 2ᵏ distinguishable values
- Therefore d · N · p ≥ k
- For constant d and p: N = Ω(k)

**Key insight:** The lower bound holds *regardless of parameter count*. More parameters enable more sophisticated processing of each input, but cannot reduce the information bottleneck through state. The reactive constraint means Bob's processing parameters cannot adapt to the identity of the marker—they are fixed functions of the query. ∎

### 4.4 Proof of Upper Bound

We construct an intentional system achieving the claimed bounds.

**State representation:** h ∈ ℝᵈˣᴺ with N = ⌈log₂ k⌉ + 2

- Dimensions 1 through ⌈log₂ k⌉: Binary encoding of the most recent marker index
- Dimension ⌈log₂ k⌉ + 1: "Marker seen" flag (0 initially, 1 after any marker)
- Dimension ⌈log₂ k⌉ + 2: Computation auxiliary

For concreteness, we use the state-space model framework with the following additions.

**Parameter structure:**

Let d be the embedding dimension. We require d ≥ k for the standard construction. (For k > d, use ⌈k/d⌉ detector rows per marker, scaling parameters to O(k²d) while maintaining N = O(log k).)

**Intentional gate computation:** The key modification is that parameters depend on the output projection of previous state:

    gₜ = [yₜ₋₁; xₜ] ∈ ℝ²ᵈ
    Δₜ = softplus(W_Δ · gₜ + b_Δ) ∈ ℝᵈ

where yₜ₋₁ = C · hₜ₋₁ + D · xₜ₋₁ is the output at the previous step.

**W_Δ ∈ ℝᵈˣ²ᵈ organized as k detector blocks:**
- Block W_Δ⁽ⁱ⁾ ∈ ℝ⁽ᵈ/ᵏ⁾ˣ²ᵈ activates strongly when input is marker mᵢ
- Implementation: W_Δ⁽ⁱ⁾ has positive weights aligned with the embedding of mᵢ

**W_B ∈ ℝ⁽ᵈ·ᴺ⁾ˣ²ᵈ organized as k encoder blocks:**
- Block W_B⁽ⁱ⁾ writes the binary representation of i to state dimensions 1:⌈log₂ k⌉
- Sets the "marker seen" flag to 1

**W_C ∈ ℝ⁽ᵈ·ᴺ⁾ˣ²ᵈ organized as k decoder blocks:**
- At query time, reads the binary index from state and selects the appropriate output

**Operation when marker mᵢ arrives:**
1. Compute gate input: gₜ = [yₜ₋₁; mᵢ]
2. Marker detection: W_Δ⁽ⁱ⁾ activates for mᵢ's embedding, producing large Δ in corresponding rows
3. State write: Large Δ triggers strong update; W_B⁽ⁱ⁾ writes binary(i) to state
4. Output update: yₜ encodes "marker i was seen"

**Operation when query q arrives:**
1. Compute gate input: gₜ = [yₜ₋₁; q] where yₜ₋₁ encodes the marker index
2. **This is where intentionality matters:** W_Δ can read the marker encoding from yₜ₋₁
3. Parameter selection: Based on the index in yₜ₋₁, the system activates the appropriate decoder
4. Output: W_C produces output i

**Why reactive systems cannot implement this:** 
At query time, a reactive system computes θ_q = G(q). Since G depends only on q, which is fixed, θ_q is identical regardless of which marker appeared. The only way for the output to vary with the marker is through the state h. But distinguishing k markers in state requires Ω(k) dimensions.

With intentionality, the parameter selection G(yₜ₋₁, q) can implement a k-entry lookup table keyed by the O(log k)-bit encoding in yₜ₋₁. The "routing" happens in the parameters, not in the state.

**Parameter count:** 
- W_Δ: O(d · 2d) = O(d²)
- W_B: O(d · N · 2d) = O(d² · log k), but with k specialized blocks: O(k · d²)
- W_C: O(d · N · 2d) × k blocks = O(k · d²)
- Total: O(k · d²)

**Verification:** The construction is fully specified. Given k and d, one can write down explicit matrices implementing the above. No steps are omitted or left to "learn from data." ∎

---

## 5. Implications for Architecture Design

### 5.1 The Trade-off

Theorem 2 reveals a fundamental trade-off:

**Reactive systems:** Parameters independent of accumulated state. Enables efficient parallel computation (all parameters can be computed simultaneously). Requires Ω(k) state for k-way routing.

**Intentional systems:** Parameters depend on accumulated state. Creates sequential dependence in parameter computation. Requires only O(log k) state for k-way routing, at cost of O(k) parameter overhead.

This is not a free lunch. Intentionality trades **state efficiency** for **parameter overhead** and **sequential dependence in parameter selection**.

### 5.2 When Does This Matter?

The separation is significant when:

1. **k is large:** For small k, both approaches are feasible. As k grows, the Ω(k) state requirement for reactive systems becomes prohibitive.

2. **Tasks involve context-dependent routing:** Not all tasks require routing based on accumulated context. The separation applies specifically to tasks with this structure.

3. **State is more expensive than parameters:** In memory-limited settings (e.g., long sequences), the logarithmic state reduction may be worth the parameter overhead.

### 5.3 Natural Language Examples

While k-Marker is artificial, similar structures appear in natural language:

**Coreference resolution:** "The doctor told the nurse that she..." The pronoun "she" routes to different antecedents based on accumulated context. With k possible referents, reactive resolution requires Ω(k) state to remember all candidates.

**Topic-dependent interpretation:** "Bank" routes to FINANCIAL vs. GEOGRAPHICAL based on accumulated context. The routing decision depends on what has been established.

**Instruction following:** "Use metric units throughout" establishes a context that affects all subsequent numerical processing. The instruction is k=1 marker, but the principle extends.

**Long-range dependencies:** In "The man who the woman who the child saw helped left," subject-verb agreement requires routing through nested dependencies—a form of context-dependent processing.

We do not claim the k-Marker task is "natural." We claim that natural tasks often have components with similar structure.

---

## 6. Minimal Instantiation: The ANIMA Cell

### 6.1 Purpose

We present a minimal architecture satisfying the intentional criterion. This is not a practical system; it is a **constructive proof** that the upper bound is achievable.

### 6.2 Specification

**ANIMA (Adaptive Neural Intentional Memory Access)** is a state-space model variant where gate parameters depend on accumulated output:

**Gate computation:**
    gₜ = [yₜ₋₁; xₜ] ∈ ℝ²ᵈ
    Δₜ = τ · softplus(W_Δ · gₜ + b_Δ) ∈ ℝᵈ
    ρₜ = σ(W_ρ · gₜ + b_ρ) ∈ ℝᵈ

**State update:**
    Āₜ = exp(Δₜ ⊙ A)
    B̄ₜ = Δₜ ⊙ Bₜ
    ỹₜ₋₁ = ρₜ ⊙ yₜ₋₁
    hₜ = Āₜ ⊙ hₜ₋₁ + B̄ₜ ⊙ [ỹₜ₋₁; xₜ]

**Output:**
    yₜ = C · hₜ + D · xₜ

**Parameter interpretation:**
- Δ (update rate): "How much do I trust new information versus preserved state?"
- ρ (exposure gate): "How much of my accumulated understanding should influence the update?"

**Why two gates:** 
- Δ controls *temporal dynamics*—the rate of state decay
- ρ controls *information content*—what context influences the write

Consider: a model needs slow state decay (preserve memory) but context-independent writes (fresh encoding). This requires small Δ and small ρ. Conversely, fast forgetting with context-dependent writes requires large Δ and large ρ. Neither gate alone expresses both patterns.

### 6.3 What ANIMA Is and Is Not

**ANIMA is:**
- A minimal example satisfying Definition 4 (intentional)
- A constructive proof of Theorem 2(b)
- An instantiation demonstrating the upper bound is achievable

**ANIMA is not:**
- A claim of practical superiority
- A benchmark-beating architecture
- A production system
- An empirically validated approach

---

## 7. Limitations

We explicitly state what this paper has NOT shown.

### 7.1 Representational, Not Learnability

Theorem 2 proves that intentional systems can *represent* solutions to k-Marker with O(log k) state. It does not prove that gradient descent *finds* these representations from random initialization. The separation is about capacity, not learning dynamics.

### 7.2 Specific Task Class

The k-Marker task captures context-dependent routing. Not all computational tasks have this structure. The separation may not apply to tasks without routing components.

### 7.3 Artificial Task

k-Marker is a constructed task designed to exhibit the separation cleanly. While we argue natural tasks have similar components, we have not formally proven that any specific natural task reduces to k-Marker.

### 7.4 Parameter Overhead

The intentional system achieves O(log k) state at cost of O(k · d²) parameters. For settings where parameters are more expensive than state, this trade-off may not be favorable.

### 7.5 Training/Inference Gap

In the ANIMA instantiation, training uses yₜ₋₁ from the *previous layer* (for parallelism) while inference uses yₜ₋₁ from the *same layer* (sequential). This creates a distribution gap. We have not characterized its impact.

### 7.6 No Empirical Validation

This paper contains no experiments. The claims are purely theoretical. Empirical validation is future work.

---

## 8. Related Work

### 8.1 Sequence Models

**Transformers** (Vaswani et al., 2017) achieve O(1) recurrent state at cost of O(n²) attention computation and O(n) KV-cache. They are reactive in our sense: attention parameters are fixed functions of the query.

**State space models** (Gu et al., 2022; Gu & Dao, 2023) achieve O(nd) state with efficient parallel training. Mamba's selective mechanism is reactive: Δ = f(x), not f(h, x).

**Gated recurrent networks** (Cho et al., 2014; Hochreiter, 1997) are intentional: gates depend on hidden state. But sequential recurrence prevents parallel training, limiting their practical deployment at scale.

### 8.2 Communication Complexity

INDEX lower bounds have been used to establish streaming complexity (Ablayev, 1996), to prove lower bounds on neural network size (Raz, 2018), and to analyze the complexity of learning (Feldman, 2017). Our application to distinguishing reactive from intentional computation appears to be novel.

### 8.3 Biological Plausibility of Intentional Parameter Selection

Intentional parameter selection—where parameters depend on accumulated state—has biological instantiation through neuromodulation. Biological neural circuits employ diffuse chemical signaling (dopamine, acetylcholine, serotonin, neuropeptides) to dynamically modulate circuit parameters based on internal state (Harris-Warrick & Marder, 1991; Marder & Bucher, 2007).

Formally, neuromodulated circuit parameters satisfy:
$$p_j(t) = G_j(M_1(t), ..., M_k(t), v_1(t), ..., v_n(t))$$

where {M₁, ..., Mₖ} are modulatory signal concentrations and {v₁, ..., vₙ} are membrane potentials. This matches Definition 4: parameters depend on both accumulated state and current input.

The degeneracy principle (Prinz et al., 2004; Marder & Goaillard, 2006) further supports this: biological neural circuits achieve robust function despite parameter variability through adaptive modulation, not hard-coded specification. This suggests intentional parameter selection is a necessary component of adaptive neural function.

---

## 9. Conclusion

### 9.1 Summary

We identified a blindspot shared across machine learning architectures: the treatment of context as spatial rather than temporal-causal. We formalized this as the distinction between reactive systems (parameters depend only on current input) and intentional systems (parameters depend on accumulated state and current input).

We proved a separation theorem: for tasks requiring k-way context-dependent routing, reactive systems require Ω(k) state while intentional systems achieve O(log k) state with O(k · d²) parameters. The proof uses reduction from the INDEX problem in communication complexity.

We presented ANIMA as a minimal architecture satisfying the intentional criterion, serving as a constructive proof of the upper bound.

### 9.2 What This Contributes

**A formal vocabulary** for discussing a real distinction in sequence processing architectures.

**A separation theorem** establishing that the distinction has computational consequences.

**A proof technique** (INDEX reduction) applicable to other architecture comparisons.

**A minimal instantiation** demonstrating the upper bound is achievable.

### 9.3 What This Does NOT Contribute

Empirical benchmarks. A practical system. A claim that any existing architecture should be modified. A complete theory of temporal reasoning.

### 9.4 Future Work

**Empirical validation:** Implement ANIMA and evaluate on tasks with context-dependent routing structure.

**Learnability analysis:** Prove or disprove that gradient descent finds intentional solutions.

**Natural task reduction:** Formally characterize which natural language tasks reduce to k-Marker.

**Training dynamics:** Close the training/inference gap in parallel intentional systems.

**Biological validation:** Empirically verify that neuromodulatory dynamics in cultured neural circuits follow the intentional criterion (Definition 4), and test whether circuits with state-dependent parameter modulation show increased robustness to parameter variation (Prinz et al., 2004).

---

## References

Ablayev, F. (1996). Lower bounds for one-way probabilistic communication complexity and their application to space complexity. *Theoretical Computer Science*, 157(2), 139-159.

Borgeaud, S., et al. (2022). Improving language models by retrieving from trillions of tokens. *ICML*.

Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.

Feldman, V. (2017). A general characterization of the statistical query complexity. *COLT*.

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing machines. *arXiv:1410.5401*.

Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR*.

Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

Raz, R. (2018). Fast learning requires good memory: A time-space lower bound for parity learning. *FOCS*.

Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

Harris-Warrick, R. M., & Marder, E. (1991). Modulation of neural networks for behavior. *Neuron*, 7(6), 907-920.

Marder, E., & Bucher, D. (2007). Understanding circuit dynamics using the stomatogastric nervous system of lobsters and crabs. *Annual Review of Physiology*, 69, 291-316.

Marder, E., & Goaillard, J. M. (2006). Variability, compensation and homeostasis in neuron and network function. *Nature Reviews Neuroscience*, 7(7), 563-574.

Prinz, A. A., Bucher, D., & Marder, E. (2004). Similar network activity from disparate circuit parameters. *Nature Neuroscience*, 7(12), 1345-1352.

---

## Appendix A: Complete Proof Details

### A.1 INDEX Lower Bound (Ablayev, 1996)

**Theorem (restated).** Any one-way randomized protocol for INDEX(k) with success probability ≥ 2/3 requires Ω(k) bits of communication.

**Proof sketch:** Consider the uniform distribution over (s, i). Alice's message m is a random variable depending on s. By data processing inequality, I(m; sᵢ | i) ≤ H(m) ≤ |m| bits. For Bob to decode sᵢ with probability 2/3, we need I(m; sᵢ | i) ≥ 1 - H(2/3) > 0 for each i. Summing over all i and using the chain rule: |m| ≥ Σᵢ I(m; sᵢ | i) ≥ k · (1 - H(2/3)) = Ω(k). ∎

### A.2 Reduction Correctness

**Lemma.** The reduction from INDEX to k-Marker is valid.

**Proof:** We verify:
1. Alice can encode s as a k-Marker input (include marker mⱼ iff sⱼ = 1)
2. The final state hₐ contains the only information about s available to Bob
3. Bob's query q is independent of Alice's input
4. Correct k-Marker solution implies correct INDEX solution

For (4): If the transducer correctly outputs j when marker mⱼ appeared, Bob can determine whether mᵢ appeared by checking if the output matches i. This solves INDEX. ∎

### A.3 Upper Bound Construction Details

**W_Δ block structure:** For marker mᵢ with embedding eᵢ ∈ ℝᵈ:
    W_Δ⁽ⁱ⁾[j, d:2d] = α · eᵢ[j] for j ∈ [(i-1)·(d/k), i·(d/k)]
    W_Δ⁽ⁱ⁾[j, 0:d] = 0

where α > 0 is a scaling constant. This ensures that when input is mᵢ:
    (W_Δ · [y; mᵢ])[block i] ≈ α · ||eᵢ||² >> (W_Δ · [y; mⱼ])[block i] for j ≠ i

**W_B block structure:** Let binary(i) ∈ {-1, +1}^{log k} be the binary encoding of i.
    W_B⁽ⁱ⁾ writes binary(i) to state dimensions 1:⌈log₂ k⌉ when activated

**W_C block structure:** At query time, W_C reads the binary encoding from yₜ₋₁ (which reflects state via C projection) and activates the decoder producing output i.

---

*End of Paper*
