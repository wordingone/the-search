# paper7 train worst

> **Original PDF Size:** 1.1M
> **Text Size:** 124K
> **Conversion Date:** 2026-01-22 07:34:35

---

                                                                      Train for the Worst, Plan for the Best:
                                                               Understanding Token Ordering in Masked Diffusions


                                                             Jaeyeon Kim * 1 Kulin Shah * 2 Vasilis Kontonis 2 Sham Kakade 1 Sitan Chen 1


                                                                   Abstract                                 1. Introduction
                                                                                                            While diffusion models (Ho et al., 2020; Song et al., 2021)
arXiv:2502.06768v3 [cs.LG] 19 Aug 2025




                                                                                                            are now the dominant approach for generative modeling in
                                               In recent years, masked diffusion models (MDMs)
                                                                                                            continuous domains like image, video, and audio, efforts to
                                               have emerged as a promising alternative approach
                                                                                                            extend this methodology to discrete domains like text and
                                               for generative modeling over discrete domains.
                                                                                                            proteins (Austin et al., 2021; Lou et al., 2024; Hoogeboom
                                               Compared to autoregressive models (ARMs),
                                                                                                            et al., 2021b) remain nascent. Among numerous proposals,
                                               MDMs trade off complexity at training time with
                                                                                                            masked diffusion models (MDMs) (Lou et al., 2024; Sahoo
                                               flexibility at inference time. At training time, they
                                                                                                            et al., 2025; Shi et al., 2024) have emerged as a leading
                                               must learn to solve an exponentially large number
                                                                                                            variant, distinguished by a simple and principled objective:
                                               of infilling problems, but at inference time, they
                                                                                                            to generate samples, learn to reverse a noise process which
                                               can decode tokens in essentially arbitrary order. In
                                                                                                            independently and randomly masks tokens.
                                               this work, we closely examine these two compet-
                                               ing effects. On the training front, we theoretically         In many applications, such as language modeling, masked
                                               and empirically demonstrate that MDMs indeed                 diffusion models (MDMs) still underperform compared to
                                               train on computationally intractable subproblems             autoregressive models (ARMs) (Nie et al., 2024; Zheng
                                               compared to their autoregressive counterparts. On            et al., 2024), which instead learn to reverse a noise process
                                               the inference front, we show that a suitable strat-          that unmasks tokens sequentially from left to right. However,
                                               egy for adaptively choosing the token decoding               recent studies suggest that MDMs may offer advantages in
                                               order significantly enhances the capabilities of             areas where ARMs fall short, including reasoning (Nie et al.,
                                               MDMs, allowing them to sidestep hard subprob-                2024; Kitouni et al., 2025), planning (Ye et al., 2024), and
                                               lems. On logic puzzles like Sudoku, we show that             infilling (Gong et al., 2024). This raises a key question:
                                               adaptive inference can boost solving accuracy in             what are the strengths and limitations of MDMs compared
                                               pretrained MDMs from < 7% to ≈ 90%, even                     to ARMs, and on what type of tasks can MDMs be scaled
                                               outperforming ARMs with 7× as many parame-                   to challenge the dominance of ARMs in discrete generative
                                               ters and that were explicitly trained via teacher            modeling?
                                               forcing to learn the right order of decoding. This
                                                                                                            To understand these questions, we turn a microscope to two
                                               shows that MDMs without knowledge of the cor-
                                                                                                            key competing factors when weighing the merits of MDMs
                                               rect token generation order during training and in-
                                                                                                            over ARMs:
                                               ference can outperform ARMs trained with knowl-
                                               edge of the correct token generation order. We               • Complexity at training time: MDMs face a more chal-
                                               also show the effectiveness of adaptive MDM in-                lenging training task by design. While ARMs predict the
                                               ference on reasoning tasks such as coding and                  next token given an unmasked prefix, MDMs predict a
                                               math on the 8B large language diffusion model                  token conditioned on a set of unmasked tokens in arbi-
                                               (LLaDa 8B).                                                    trary positions. This inherently increases their training
                                                                                                              complexity.

                                           *
                                            Equal contribution 1 Harvard University 2 University of         • Flexibility at inference time: On the other hand, the
                                         Texas Austin.    Correspondence to: Kulin Shah <kulin-               sampling paths taken by an MDM are less rigid. Unlike
                                         shah@utexas.edu>.                                                    the fixed left-to-right decoding of ARMs, MDMs decode
                                         Proceedings of the 42 nd International Conference on Machine         tokens in random order at inference. Even more is possi-
                                         Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025          ble: MDMs can be used to decode in any order (including
                                         by the author(s).                                                    left-to-right).

                                                                                                        1
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Therefore, we ask:                                                   MDMs that decide the correct token generation order during
                                                                     inference via adaptive strategies can outperform ARMs that
Are the benefits of inference flexibility for MDMs enough to         are trained to learn the right token generation order via
      outweigh the drawbacks of training complexity?                 supervised teacher forcing (Shah et al., 2024; Lehnert et al.,
                                                                     2024).
In this work, we provide dual perspectives on this question.
(1) Training for the worst. First, we provide theoretical            Organization. In Section 2, we provide preliminaries on
and empirical evidence that the overhead imposed by train-           MDMs and set notation. In Section 3, we examine MDM
ing complexity quantifiably impacts MDMs’ performance.               training and demonstrate the imbalance in computational
                                                                     intractability across subproblems. In Section 4, we consider
Theoretically, we show examples of simple data distribu-             adaptive inference in MDMs and investigate its impact on
tions with a natural left-to-right order, where ARMs can             likelihood modeling across various tasks.
provably generate samples efficiently. In contrast, there
are noise levels at which a large fraction of the correspond-
ing subproblems solved by MDMs for these distributions               2. Masked Diffusion Models (MDM)
are provably computationally intractable. Empirically, we            In this section, we explain the framework of Masked Dif-
validate this claim on real-world text data, known to have           fusion Models (Shi et al., 2024; Sahoo et al., 2025) and
left-to-right order and show that the imbalance in training          highlight its interpretation as an order-agnostic learner.
complexity across subproblems persists even in real-world            MDMs gradually add masking noise to the true discrete
text data (Fig. 2, left).                                            data and learn the marginal distribution of the induced re-
                                                                     verse process. We formally define both the forward and
(2) Planning for the best. While the above might appear              reverse processes for MDMs below.
to be bad news for MDMs, in the second part of this paper,
we answer our guiding question in the affirmative by build-          Let the distribution pdata on {1, . . . , m}L be the data dis-
ing upon the observation (Chang et al., 2022; Zheng et al.,          tribution over sequences of length L and with vocabulary
2023) that MDMs which can perfectly solve all masking                {1, . . . , m}. We use 0 to denote the “mask” token.
subproblems can be used to decode in any order.
                                                                     Forward process. For a given x0 ∼ pdata and a noise
In first part of the paper, we show that the imbalance in com-       level t ∈ [0, 1], the forward process xt ∼ qt|0 (· | x0 ) is a
plexity across subproblems during the training of MDMs               coordinate-independent masking process via qt|0 (xt |x0 ) =
results in some of the subproblems being poorly trained              QL−1         i i
and the vanilla MDM inference that unmasks tokens in ran-              i=0 qt|0 (xt |x0 ), where

dom order results in evaluating the poorly trained marginals.
                                                                             qt|0 (xit | xi0 ) = Cat αt exi0 + (1 − αt )e0 .
                                                                                                                          
Therefore, in place of vanilla MDM inference, we consider
adaptive strategies that carefully select which token to un-         Here, αt is a predefined noise schedule satisfying α0 ≈
mask next. Our key insight is that the adaptive strategies           1, α1 ≈ 0 and exi0 ∈ Rm+1 is a one-hot vector correspond-
makes it possible to sidestep the hard subproblems from
                                                                     ing to the value of token xi0 . Cat(π) denotes the categorical
training (Fig. 1). In particular, we find that even without
                                                                     distribution given by π ∈ ∆m . In other words, for each i-th
modifying how MDMs are trained, the resulting models’
                                                                     coordinate, xit is masked to the mask token 0 with probabil-
logits contain enough information to determine the right
                                                                     ity 1 − αt and remains unchanged otherwise.
order in which to unmask. We show the effectiveness of the
adaptive inference in solving logic puzzles, coding, math
                                                                     Reverse process. The reverse process of the above for-
and infilling tasks. For example, on Sudoku puzzles, a sim-
                                                                     ward process is denoted by qs|t (xs |xt , x0 ) and is given
ple adaptive strategy (Section 4.1) improves the accuracy of                                    QL−1
MDMs from < 7% to almost 90%.                                        by qs|t (xs |xt , x0 ) = i=0 qs|t (xis |xt , x0 ) for any s < t,
                                                                     where
Advantage of MDMs over ARMs. We show that the                                                (                                  i
main effectiveness of MDMs lies in tasks that do not have            qs|t (xis | xt , x0 ) =
                                                                                               Cat(e
                                                                                                   xit )                   xt ̸= 0
                                                                                                    1−αs      αs −αt
the same natural token generation order across all sequences                                   Cat 1−αt e0 + 1−αt exi0        xit = 0 .
(e.g., logic puzzles and reasoning tasks like coding and
math). By carefully designing experiments on logic puzzles,          The reverse transition probability qs|t (xis |xt , x0 ) is approx-
we show that MDMs without the knowledge of the correct               imated using gθ (xis |xt ) ≜ qs|t (xis | xt , x0 ← pθ (·|xt , t))
token generation order during training and inference                 where pθ (·|xt , t) is a denoising network trained to predict
can outperform ARMs trained with the knowledge of the                the marginal distribution on xi0 via an ELBO-based loss for
correct token generation order. In particular, we show that          all masked tokens at noise scale t (i.e., for all i such that

                                                                 2
                  Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

                                                                                   2.1. Reformulating the training and inference of MDMs
                                                                                   In this section, we first discuss training of MDMs and com-
                                                                                   pare it with “left-to-right” order training of autoregressive
                                                                                   models in Section 2.1.1. Then, we reformulate vanilla MDM
                                                                                   inference in Section 2.1.2 to set the stage for the upcoming
                                                                                   discussion.

                                                                                   2.1.1. O RDER - AGNOSTIC TRAINING OF MDM S
                                                                                   Recent works (Zheng et al., 2024; Ou et al., 2024) have
                                                                                   observed that the learning problem of MDM is equivalent
                                                                                   to a masked language model. Building upon their analysis,
                                                                                   we reformulate the loss Lθ to show that Lθ is a linear com-
                                                                                   bination of the loss for all possible infilling masks. We first
                                                                                   define x0 [M ] as a masked sequence, obtained from original
                                                                                   sequence x0 where indices in the mask set M (a subset of
                                                                                   [L] ≜ {1, 2, . . . , L}) are replaced with mask token 0.
                                                                                   Proposition 2.1. Assume α0 = 1, α1 = 0 and de-
                                                                                   noising network pθ is time-embedding free. Then Lθ ≤
                                                                                   −Ex0 ∼pdata [log pθ (x0 )] and
                                                                                                X           1     1                    i
                                                                                   Lθ = −                         L        E [log pθ (x0 |x0 [M ])],
                                                                                                          |M | |M
                                                                                                                     
                                                                                                                       x0 ∼pdata
                                                                                            M ⊆[L],i∈M              |
                                                                                                                                            (1)
                                                                                   where |M | is the size of the set M and pθ (xi | x0 [M ])
Figure 1. (Top) MDM training can be seen as learning multi-
ple masked prediction problems, where some are harder to learn,
                                                                                   indicates the conditional probability of the i-th coordinate
leading to performance imbalance (Section 3). (Bottom) During                      from pθ (xt ).
inference, adaptive MDM can avoid difficult problem instances,
                                                                                   The proof of the above proposition is given in Appendix E.
improving performance (Section 4).
                                                                                   As the MDM loss is a linear combination of the loss for
                                                                                   all possible infilling mask M , the minimizer of the loss Lθ
                                                                                   learns to solve every masking problem. In other words, the
                                                             
xit = 0). To be precise, qs|t xis | xt , x0 ← pθ (·|xt , t) indi-
cates the conditional probability where pθ (·|xt , t) is placed                    optimal predictor pθ is the posterior marginal of the i-th
in the position of exi0 within qs|t (xis | xt , x0 ). The denoising                token, conditioned on x0 [M ] for all masks M .
network is trained to minimize the following loss derived                          On the other hand, Autoregressive Models (ARMs) learn
from the score-entropy (Lou et al., 2024; Sahoo et al., 2025;                      to predict ith token xi based on all preceding tokens, from
Shi et al., 2024; Ou et al., 2024):                                                x0 to xi−1 . This is equivalent to predicting xi by masking
        Z 1                                                                        positions from i to L − 1. Therefore, the training objective
                αt′                      X
                                                                                   for ARMs can be expressed as:
 Lθ =                       E                      − log pθ (xi0 |xt , t)dt,
         0    1 − αt     x ∼p
                           0    data
                                                                                                       L−1
                       xt ∼qt|0 (·|x0 ) i:xit =0                                                       X
                                                                                      log pθ (x0 ) =         log pθ (xi0 |x0 [{i, . . . , L − 1}]).   (2)
                 dαt
where αt′   =     dt  and the summation is computed over                                               i=0

masked tokens (i.e., all i such that xit = 0). In practice,                        Typically, ARMs are trained to predict tokens sequentially
a time-embedding-free architecture for the denoising net-                          from left to right. We refer to this as left-to-right training.
work, i.e., pθ (·|xt , t) = pθ (·|xt ) is generally used as xt                     However, it’s also possible to train these models to predict
implicitly contains information about t via the number of                          tokens sequentially based on a fixed, known permutation of
masked tokens.                                                                     the sequence. We refer to this general approach as order-
                                                                                   aware training.
The reverse sampling process starts from the fully masked
sentence x1 = (0, . . . , 0). Suppose we have a partially                          To understand the comparison between the training objec-
\fully masked sequence xt at a given noise level t ∈ (0, 1].                       tive of MDMs and ARMs, we want to highlight the equiva-
Then, to obtain xs for a predetermined noise level s < t,                          lence between any-order autoregressive loss and MDM loss
we sample xis ∼ gθ (·|xt ) for all i. This process is repeated                     (Hoogeboom et al., 2021a; Ou et al., 2024). In particular,
recursively from t = 1 to t = 0.                                                   under conditions of Proposition 2.1, MDM loss is equal to

                                                                               3
                   Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

                                                                         the ones encountered during order-agnostic training (such
                       "L−1
                                    
                                                               #
                                                                        as in MDMs) are computationally intractable. In Sec-
                        X             π(i)
Lθ = −       E                log pθ x0 x0 [π{i, . . . , L − 1}] ,       tion 3.2, we empirically show that text data also exhibits
          x0 ∼pdata
         π∼Unif(SL )    i=0                                              this gap between the computational complexity of order-
                                                                         aware and order-agnostic training and therefore, MDMs
where Unif(SL ) is a uniform distribution over all the per-              train on subproblems of wide variety of complexity (de-
mutations of length L (See Appendix E.1 for the proof).                  pending on the order/masks). In Section 3.3, we empiri-
Observe that if the expectation is only with respect to the              cally show that the variety in training complexity results
identity permutation, then the loss becomes an autoregres-               in performance imbalance across subproblems: MDMs
sive loss. This shows that MDM loss solves exponentially                 trained on data from such distributions exhibits small errors
more subproblems than ARM loss. In contrast to ARM                       on easy subproblems but suffers from large errors on harder
loss, MDM does not prefer any particular (e.g., left-to-right)           ones.
order during the training; therefore, we call its training
order-agnostic training.                                                 3.1. Benign distributions with hard masking problems
                                                                         We now describe a simple model of data under which we
2.1.2. O RDER - AGNOSTIC INFERENCE OF MDM S
                                                                         explore the computational complexity of masking problems
The MDM inference can be decomposed into two steps:                      and show the contrast between masking problems encoun-
(a) randomly selecting a set of positions to unmask and (b)              tered by MDMs and ARMs.
assigning token values to each position via the denoising                Definition 3.1. A latents-and-observations (L&O) distribu-
network pθ . More precisely, we can reformulate the reverse              tion is a data distribution pdata over sequence of length L
process xs ∼ gθ (·|xt ) as follows.                                      with alphabet size m (precisely, pdata is over {0, . . . , m}L )
                                                                         is specified by a permutation π over indices {1, 2, . . . , L},
    Vanilla MDM inference
                                                                         number of latent tokens N , number of observation tokens P
                                                                         such that N + P = L, prior distribution pprior of latent vari-
    (a) Sample a set of masked tokens S ⊆ {i | xit = 0},                 ables over {1, . . . , m} and efficiently learnable observation
                      s −αt
        P(i ∈ S) = α1−α   t
                            .                                            functions O1 , . . . , OP : {1, . . . , m}N → ∆({0, . . . , m}),1

    (b) For each i ∈ S, sample xis ∼ pθ (xi |xt ).                       • (Latent tokens) For i = 1, . . . , N , sample xπ(i) indepen-
                                                                           dently from the prior distribution pprior of the latents.

Therefore, the inference in MDM is implemented by ran-                   • (Observation tokens) For j = 1, . . . , P , sample xπ(N +j)
domly selecting S and then filling each token value accord-                independently from Oj (xπ(1) , . . . , xπ(N ) ).
ing to the posterior probability pθ (xis |xt ).
                                                                         L&O distributions contain two types of tokens: (1) latent to-
On the other hand, ARMs are trained to predict tokens se-                kens and (2) observation tokens. Intuitively, latent tokens are
quentially from left to right and therefore, generate tokens             tokens in the sequence, indexed by π(1), π(2), . . . , π(N )
also in left-to-right order. In contrast, vanilla MDM infer-             that serve as “seeds” that provide randomness in the se-
ence generates the tokens in a random order.                             quence; the remaining tokens, called observation tokens
                                                                         (indexed by π(N + 1), π(N + 2), . . . , π(N + P )), are de-
3. MDMs train on hard problems                                           termined as (possibly randomized) functions of the latent
                                                                         tokens via O1 , . . . , OP . Observe that L&O distributions
In this section, we provide theoretical and empirical evi-               specified by a permutation π have a natural generation order
dence that when the data distribution has left-to-right order            by permutation π.
(or any fixed known order) then autoregressive training in
left-to-right order (or in the known order) is more tractable            Order-aware training Order-aware training, i.e. by per-
than MDMs. In particular, for such distributions with fixed              muting the sequence so that π becomes the identity permuta-
order, we show that ARMs can efficiently sample from the                 tion and then performing autoregressive training, is computa-
distributions but for MDMs, we theoretically and empiri-                 tionally tractable: predicting xπ(i) given xπ(1) , . . . , xπ(i−1)
cally demonstrate that a large portion of masking subprob-               is trivial when i ≤ N as the tokens are independent, and
lems pθ (xi0 | x0 [M ]) can be difficult to learn.                       computationally tractable when i > N because xπ(i) only
In Section 3.1, we show several examples of simple, non-                     1
                                                                               Here efficiently learnable is in the standard PAC sense: given
pathological distributions for which: (1) the masking prob-              polynomially many examples of the form (z, y) where z ∼ pN        prior
lems encountered during order-aware training (such as                    and y ∼ Oj (z), there is an efficient algorithm that can w.h.p. learn
in ARMs) are computationally tractable, yet (2) many of                  to approximate Oj in expectation over pN     prior .


                                                                     4
                   Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions




Figure 2. Left: MDMs train on hard problems (Section 3.2). x-axis and y-axis correspond to log(FLOPs) and − log pθ (x), respectively.
MDM (Blue) is worse than ARM (Orange) in likelihood modeling. Most masking problems (Other lines) that MDM is trained on are
harder than those encountered by ARM, as indicated by small log-likelihoods. Right: Task error imbalance (Section 3.3). MDM’s
performance varies across different tasks. For text data (top right), this is indicated by validation loss. For L&O-NAE-SAT (bottom right),
MDM performs well on the masking problems for observation positions (light region) but struggles with latent positions (dark region).


depends on xπ(1) , . . . , xπ(N ) and is efficiently learnable by            size k and set the total number of observation latents P
assumption. In contrast, below we will show examples                         equal to the size of this set (hence P = N !/(N − k)! =
where if one performs order-agnostic training à la MDMs,                    N (N − 1) · · · (N − k + 1)). To sample a new sequence, we
one will run into hard masking problems with high proba-                     first sample latent tokens xπ(1) , . . . , xπ(N ) from the prior
bility.                                                                      distribution pprior and an observation latent corresponding
                                                                             to a k-sized subset S is given by g({xπ(i) }i∈S ). In other
                                                                             words, each observation latent corresponds to a k-sized sub-
Order-agnostic training We first note that if the ob-
                                                                             set S of {1, 2, . . . , N } and the corresponding observation
servations (O1 , . . . , OP ) are given by a cryptographic
                                                                             function OS (xπ(1) , . . . , xπ(N ) ) is given by g({xπ(i) }i∈S ).
hash function, then the masking problem of predicting
(xπ(1) , . . . , xπ(L) ) given (xπ(N +1) , . . . , xπ(N +P ) ) is com-       Proposition 3.3. Let x be a sample from an L&O dis-
putationally intractable by design because it requires invert-               tribution pdata with sparse predicate observations as de-
ing the hash function. While this is a well-known folklore                   fined in Example 3.2, with arity k and predicate g satis-
observation regarding the role of token ordering in language                 fying Assumption B.11, and let γ be the probability that
modeling, it is not entirely satisfying because this construc-               g is satisfied by a random assignment from {1, . . . , m}k .
tion is worst-case in nature – in real-world data, one rarely                Let DKS and Dcond be some constants associated with
trains on sequences given by cryptographic hash functions.                   the predicate function g (see Definition B.12). Suppose
Furthermore, it only establishes hardness for a specific mask-               each token in x is independently masked with probability
ing pattern which need not be encountered in the course of                   α, and M is the set of indices for the masked tokens. If
running the reverse process.                                                 1 − γ −1 DKS /kN k−1 ≤ α ≤ 1 − γ −1 Dcond /kN k−1 , then
                                                                             under the 1RSB cavity prediction (see Conjecture B.13),
We provide several simple instances of L&O distributions
                                                                             with probability Ωk (1) over the randomness of the mask-
that address these issues: instead of leveraging delicate cryp-
                                                                             ing, no polynomial-time algorithm can solve the resulting
tographic constructions, they are average-case in nature and
                                                                             subproblem of predicting any of the masked tokens among
furthermore we can establish hardness for typical masking
                                                                             xπ(1) , . . . , xπ(N ) given x[M ].
problems encountered along the reverse process.
In all these examples, the hardness results we establish hold                The complete proof of the proposition is given in Ap-
even if the algorithm knows all of the parameters of pdata                   pendix B.4. We also provide a proof outline in Appendix B.3
as well as the observation functions O1 , . . . , OP . Due to                for a comprehensive understanding.
space constraints, here we focus on the following example,
deferring two others to Apps. B.1 and B.2.                                   3.2. Empirical evidence of hardness via likelihoods
Example 3.2 (Sparse predicate observations). Consider the                    In the previous section, we provided theoretical evidence
following class of L&O distributions. Given arity k ≥ 2,                     that order-aware training is tractable when data has a natural
fix a predicate function g : {1, . . . , m}k → {0, 1}. Con-                  order but the order-agnostic training is not. In this section,
sider the set of all ordered subsets of {1, 2, . . . , N } of                we provide empirical evidence to support this claim, using

                                                                         5
                   Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

natural text data. Additionally, recent studies (Nie et al.,                             an inductive bias towards left-to-right ordering, we employ
2024; Zheng et al., 2024) have shown that masked diffusion                               a learnable positional embedding layer for all experiments
models (MDMs) underperform compared to autoregressive                                    to correct this. Consequently, we also re-run the baseline
models (ARMs) on natural text data. In this section, we                                  results, where RoPE was employed. To investigate how the
provide evidence that this performance gap is primarily                                  distance between π and the identity permutation affects the
due to the order-agnostic training of MDMs. Natural text                                 scaling law, we consider two interpolating distributions over
inherently follows a left-to-right token order, and we show                              permutations between Unif(SL ) (i.e, MDM training) and
that as training deviates from this order, model performance                             the point mass at the identical permutation (i.e, ARM train-
progressively declines.                                                                  ing). We sample three permutations from the interpolating
                                                                                         distribution and Unif(SL ) and plot the scaling law for each
To understand the importance of the order during the train-
                                                                                         of the permutation. Due to space constraints, we provide
ing, we use the following setting: Given a permutation π
                                                                                         further experimental details in Appendix C.1.
of indices {0, 1, . . . , L − 1}, define a π-learner to be a
likelihood model log pθ (x0 ) given as follows:
                                                                                         Results. As shown in Fig. 2, the scaling law for a π-learner
                  L−1                                                                    with uniformly random π is worse than that of an ARM.
                                   π(i)
                  X                                                  
 log pθ (x0 ) =         log pθ x0            x0 [π{i, . . . , L − 1}] (3)                This elucidates the inherent hardness of masking problems
                  i=0                                                                    pθ (xi | x0 [M ]) beyond left-to-right prediction and also
                                                                                         explains why MDM, which is trained simultaneously on
In other words, the π-learner predicts the token at position                             all π ∈ SL , is worse than ARM in likelihood modeling.
                                π(0)        π(i−1)
π(i) given the clean tokens x0 , . . . , x0         and masked                           Additionally, as π gets closer to the identity permutation,
         π(i)        π(L−1)
tokens x0 , . . . , x0      . If π is the identity permutation,                          the scaling laws also get closer to ARM (π-learner-closer
this reduces to the standard (left-to-right) autoregressive                              and π-learner-much-closer in Fig. 2). This also supports
training. Note that the MDM loss encodes a π-learner for                                 the common belief that ARM is a good fit for text data as it
every permutation π because the MDM loss (1) is equivalent                               inherently follows a left-to-right ordering.
to the average loss of those π-learners over π sampled from
                                                                                         That said, it should also be noted that even though MDMs
Unif(SL ):
                                                                                         are trained on exponentially more masking problems than
                                                                                         ARM (Θ(L2L ) versus L), its performance is not signifi-
                       "L−1
                        X                                               
                                                                             #           cantly worse than π-learners. We attribute this to the bless-
                                            π(i)
Lθ = −       E                log pθ       x0 x0 [π{i, . . . , L − 1}]           ,       ing of task diversity; multi-task training can benefit both
          x0 ∼pdata
         π∼Unif(SL )    i=0                                                              the optimization dynamics (Kim et al., 2024) and validation
                                                                                         performance (Tripuraneni et al., 2021; Maurer et al., 2016;
where SL denotes the set of all permutations over                                        Ruder, 2017) due to positive transfers across tasks.
{0, 1, . . . , L − 1}. The proof of the above equivalence is
given in Appendix E. Therefore, by measuring the ‘hardness’                              3.3. Error is imbalanced across masking problems
of each π-learner, we can probe differences in hardness be-                              In previous sections, we have demonstrated that the hard-
tween arbitrary masking problems and left-to-right masking                               ness of different masking problems pθ (xi | x0 [M ]) can
problems.                                                                                vary significantly, potentially hindering the MDM’s learn-
                                                                                         ing. In this section, we provide empirical evidence that
Experimental setup. We use the Slimpajama dataset                                        the MDM’s final performance exhibits a similar imbalance
(Soboleva et al., 2023) to evaluate the performance of train-                            across subproblems. Details are provided in App. C.2.
ing in different orders. To train a π-learner, we employ a
transformer with causal attention and use permuted data                                  L&O-NAE-SAT. Consider an L&O distribution with π
π(x0 ) as input. By varying π while maintaining all other                                given by the identity permutation and where each observa-
training configurations (e.g., model, optimization), we can                              tion Oj is deterministically given by NAE(xi1 , xi2 , xi3 ) ≜
use the resulting likelihood (computed using Equation (3))                               1 − 1[xi1 = xi2 = xi3 ] for some randomly chosen (pre-
as a metric to capture the hardness of subproblems solved                                fixed) triples (i1 , i2 , i3 ) ∈ [N ]. For an MDM trained
by the π-learner.                                                                        on this distribution, we measure the error it achieves on
In our experiments, the sequence length L is 2048, so re-                                each task log pθ (x0 |x0 [M ]) via Ex0 log pθ (x0 |x0 [M ]) −
peating the scaling laws for each π is infeasible. Instead,                                                    2
we sample π ∼ Unif(SL ) and examine the scaling law of                                   log pdata (x0 |x0 [M ]) , where pdata (x0 |x0 [M ]) denotes
the π-learner’s likelihood. We leverage the codebase from                                the Bayes-optimal predictor. Technically, we do not have
(Nie et al., 2024), where the baseline scaling laws of MDM                               access to this, so instead we train another MDM for a much
and ARM were introduced. Moreover, given that RoPE has                                   larger number of iterations and use this as a proxy. Fig. 2

                                                                                     6
                  Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

reveals that prediction tasks for latent positions (light re-
gion) exhibit larger errors compared to those for observation
positions (dark region).

Text. Here we revisit the text experiment from
Section 3.2.      Since we do not have access to
the Bayes-optimal
           hP           predictor,
                                       we use thei metric
             L−1             π(i)
Ex0 ∼pdata   i=0 log p θ   x 0    x0 [π{i, . . . , L − 1}] .
This captures the       accumulation          of error  across
                       π(i)
subproblems pθ x0 x0 [π{i, . . . , L − 1}] ,                 since
pθ (x0 |x0 [M ]) = pdata (x0 |x0 [M ]) minimizes this metric.
Fig. 2 shows a clear gap between different subproblems.
The theoretical and empirical evidence demonstrates that                 Figure 3. Generative Perplexity. We compare the resulting gener-
MDMs perform better in estimating pθ (x0 |x0 [M ]) for some              ative perplexity (GenPPL) of adaptive vs. vanilla MDM inference.
subproblems M than for others. We therefore want to avoid                We employ a pretrained 170M MDM and LLaMA-7B (Touvron
encountering hard subproblems M at inference time. In the                et al., 2023) as inference and evaluation, respectively. Adaptive
                                                                         MDM inference (Blue) leads to a substantial reduction in genera-
next section, we show that while vanilla MDM inference can
                                                                         tive perplexity, while maintaining the entropy.
run into such subproblems, simple modifications at the infer-
ence stage can effectively circumvent these issues, resulting
in dramatic, training-free performance improvements.                     pdata (x0 ). Therefore, other choices of S, not necessarily
                                                                         following Algorithm 1, still capture the true likelihood.
4. MDMs can plan around hard problems                                    In practice, unlike this ideal case, MDM does not perform
We previously argued that due to the complex nature of                   equally well on all subproblems, as shown in Section 3.3.
masking subproblems, MDM must perform poorly on cer-                     Consequently, different sampling paths result in varying
tain ones pθ (xi |xt ). Therefore, during vanilla MDM infer-             likelihood modeling abilities. Motivated by this observation,
ence, MDM inevitably encounters such difficult subprob-                  we consider adaptive inference for MDMs:
lems at Step (b). While this might suggest that we need
to fundamentally revisit how MDMs are trained, in this                        Adaptive MDM inference
section we show that, surprisingly, simple modifications
                                                                             (a) Sample a set of masked tokens S = F (θ, xt ) ⊆
at the inference stage—without any further training—can
                                                                                 {i | xit = 0}.
sidestep these issues and lead to significant performance
improvements.                                                                (b) For each i ∈ S, sample xis ∼ pθ (xi |xt ).

MDM offers multiple sampling paths. The vanilla                          Instead of selecting S randomly, adaptive MDM inference
MDM inference (Algorithm 1) aim to align the intermediate                leverages an oracle F(θ, xt ) to select S strategically to
distributions with the forward process, as used in continu-              avoid hard masking problems. This naturally raises the
ous diffusion. However, unlike continuous diffusion, the                 question of how to design an effective oracle F.
reverse process of MDM allows multiple valid sampling
paths (different orders of unmasking the tokens) that match              In the following sections, we demonstrate that adaptive
the starting distribution of the forward process of MDM.                 MDM inference with careful choices of F enhance MDM’s
                                                                         likelihood matching ability. In other words, a pretrained
We first show that when we have an ideal MDM that per-                   MDM, even if it performs poorly on certain hard subprob-
fectly solves all masking problems, i.e., pθ (xi0 |x0 [M ]) =            lems, still contains sufficient information to avoid them
pdata (xi0 |x0 [M ]), then using any sampling path (unmask-              when paired with an effective oracle F.
ing the tokens in any order) results in the same distribu-
tion. Consider the following sampler: For every step,                    4.1. Effective design of ordering oracle
S is a set with one index selected agnostically (with-
out following any distribution). For any clean sam-                      We introduce two different oracles, Top probability and Top
ple x0 generated by this sampler,          note that pθ (x0 ) =         probability margin. Intuitively, both strategies are based on
QL−1  π(i)                                                              the idea that S should be selected based on how “certain”
   i=0 pθ x0        x0 [π{i, . . . , L − 1}] by chain rule, and
                                                                       the model is about each position. We caution that these
                 QL−1                π(i)
this is equal to i=0 pdata x0 x0 [π{i, . . . , L − 1}] =                 strategies should not be confused with notions like nucleus

                                                                     7
                   Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

sampling in ARMs (Holtzman et al., 2019); the oracles we
                                                                           Table 2. Comparison of accuracy for solving the Sudoku puzzle.
describe are for selecting the position of the next token to
decode, rather than the value, and thus are only meaningful                   Method                             # Param       Accuracy
in the context of MDMs.
                                                                              ARM (w/o ordering)                                 9.73%
                                                                                                                   42M
                                                                              ARM (with ordering)                                87.18%
Table 1. L&O-NAE-SAT. Adaptive MDM inference achieves bet-
ter likelihood matching than vanilla MDM inference. Note that                 MDM (vanilla)                                      6.88%
naive guessing leads to 75% accuracy, indicating that vanilla infer-          MDM (Top probability)                 6M           18.51%
ence performs similarly or worse than naive guessing.                         MDM (Top prob. margin)                             89.49%
    (N, P )      Vanilla inference       Adaptive inference
   (25, 275)           78.06%                   93.76%
   (30, 270)           75.70%                   93.54%                     on L&O-NAE-SAT and text data, before turning to our
   (40, 260)           74.60%                   92.21%                     primary application to logic puzzles.
   (50, 250)           67.94%                   90.01%                     L&O-NAE-SAT and text data. For the L&O-NAE-SAT
  (100, 200)           62.84%                   88.91%                     distribution defined in Section 3.3, we evaluate the effec-
                                                                           tiveness of adaptive inference by measuring the accuracy in
                                                                           predicting the observation tokens. Table 1 in the appendix
Top probability (Zheng et al., 2023). Suppose we want                      reveals a clear improvement over vanilla inference. For
to unmask K positions at time step t, i.e., select |S| = K.                the text dataset, we evaluate using the standard metric of
In the top probability, the uncertainty of a position is esti-             generative perplexity, by which likelihood is measured by
mated by the maximum probability assigned to any value                     a large language model. We also compute the entropy of
in the vocabulary. More precisely, the certainty at posi-                  the generated samples to ensure both inference strategies
                                i
                  i
                         pθ (x = j|xt ) and F(θ, xt ) =
tion i is maxj∈{0,...,m−1}                                                 exhibit similar levels of diversity. As shown in Fig. 3, we
Top K max pθ (x |xt ) .                                                    observe a substantial decrease in generative perplexity using
Top probability strategy is a good proxy for many tasks and                adaptive inference. We defer further experimental details to
works well in practice (Zheng et al., 2023; Ye et al., 2024;               Appendix D.1.
Wang et al., 2024). However, this approach can often pro-                  Logic puzzles. We consider two different types of logic
vide misleading estimates of uncertainty. Consider when an                 puzzles: Sudoku and Zebra (Einstein) puzzles. Intuitively,
MDM is confused between two token values, thus assign-                     for Sudoku, some empty (masked) cells are significantly
ing them almost equal but high probabilities. In this case,                easier to predict than others and we want to choose the cells
unmasking according to top probability may still choose to                 that are easier to predict during the inference. We evaluate
unmask this position, despite its uncertainty. To mitigate                 the effectiveness of adaptive MDM inference over vanilla
this issue, we propose the following alternative strategy.                 MDM inference in selecting such cells.2

Top probability margin. In this strategy, the uncer-                       To measure the performance of an inference method, we use
tainty of a position is instead estimated using the ab-                    the percentage of correctly solved puzzles. For both puzzles,
solute difference between the two most probable values                     we use train and test datasets from (Shah et al., 2024). For
at position i. More precisely, if j1 and j2 are the two                    the Sudoku puzzle (Table 2) we observe that adaptive MDM
most probable values in vocabulary according to pθ (xi |xt )               inference, in particular, Top probability margin strategy, ob-
in position i, the certainty in the position is given by                   tains substantially higher accuracy (89.49%) compared to
|pθ (xi = j1 |xt ) − pθ (xi = j2 |xt )| and                                vanilla MDM inference (6.88%). Additionally, Top prob-
                                               F(θ, xt ) =                ability margin obtains higher accuracy (89.49%) than Top
Top K |pθ (xi = j1 |xt ) − pθ (xi = j2 |xt )| . When multi-
ple values have similar probabilities at a position, top proba-            probability strategy (18.51%). As mentioned in Section 4.1,
bility margin strategy will provide a better estimate of the               this is because Top probability margin strategy more reliably
uncertainty of a position, and when there is a single best                 estimates uncertainty when multiple competing values are
choice of value then top probability and top probability                   close in probability at a given position, as is often the case
margin work similarly.                                                     in Sudoku. For the Zebra puzzle, as shown in Table 3, we
                                                                           observe a consistent result: Top probability (98.5%) and
4.2. Adaptive MDM inference                                                   2
                                                                                A prior work (Ye et al., 2024) reported that a 6M MDM with
                                                                           Top-K inference achieves 100% accuracy on Sudoku. Given that
In this section, we experimentally validate that adaptive                  a 6M MDM with Top-K only achieves 18.51% on our dataset
MDM inference helps MDMs avoid hard subproblems, lead-                     (Table 2), this suggests that the Sudoku dataset in (Ye et al., 2024)
ing to better likelihood matching. We first show our results               is significantly easier than ours.

                                                                       8
                  Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Top probability margin (98.3%) outperform vanilla MDM                 results are presented in Table 4.
inference (76.9%).
                                                                      We see that both adaptive MDM inference strategies, top
                                                                      probability and top probability margin, consistently out-
Table 3. Comparison of accuracy for solving the Zebra puzzle.         perform vanilla MDM inference. Notably, top probability
                                                                      margin demonstrates a clear advantage over top probabil-
   Method                          # Param      Accuracy              ity in challenging tasks like HumanEval-Multiline (infill),
                                                                      HumanEval-Split Line (infill), and Math. This is because
   ARM (w/o ordering)                            80.31 %
                                     42M                              Top probability margin provides a more reliable estimate of
   ARM (with ordering)                           91.17 %
                                                                      uncertainty when multiple tokens have similar probabilities,
   MDM (vanilla)                                  76.9 %              a frequent occurrence in these difficult tasks. These results
   MDM (Top probability)             19M          98.5 %              further underscore the potential for developing new, sophis-
   MDM (Top prob. margin)                         98.3 %              ticated adaptive inference strategies for various tasks. We
                                                                      provide experimental details in Appendix D.3.

4.3. Eliciting sequence-dependent reasoning paths using               4.5. Easy to hard generalization
     adaptive MDM inference in logic puzzles
                                                                      In the previous section we showed that when the training
In this section, we study the effectiveness of adaptive MDM           and inference sequences come from the same distribution,
inference in finding the right reasoning/generation order for         order-agnostic training of MDMs combined with adaptive
tasks where every sequence has a different “natural” order.           inference can perform very well on logic puzzles. To evalu-
To do so, we will compare the performance of adaptive                 ate if the model has learned the correct way of solving the
MDM inference to that of ARM on Sudoku and Zebra puz-                 puzzles and test the robustness of adaptive inference, we
zles. For these puzzles, the natural order of generation is           also test the MDMs on harder puzzles than the ones from
not only different from left-to-right, but it is also sequence-       training, for Sudoku.
dependent. For such tasks, prior works have shown that
                                                                      We keep the training dataset the same as proposed in Shah
ARMs struggle if the information about the order is not pro-
                                                                      et al. (2024). Shah et al. (2024) created this dataset from
vided during the training (Shah et al., 2024; Lehnert et al.,
                                                                      Radcliffe (2020) by selecting the puzzles that can be solved
2024). Therefore, to obtain a strong baseline, we not only
                                                                      using 7 fixed strategies and do not require backtracking-
consider an ARM trained without the order information but
                                                                      based search. We use the remaining puzzles in Radcliffe
also consider an ARM trained with the order information
                                                                      (2020) as our hard dataset. Hence, these puzzles all use
for each sequence in the training data. Note that the latter is
                                                                      a strategy not seen during training and/or backtracking to
a much stronger baseline than the former as one can hope
                                                                      obtain the correct solution.
to teach the model to figure out the correct order by some
form of supervised teacher forcing (as performed in Shah              We measure the accuracy of MDMs and ARMs on the hard
et al. (2024); Lehnert et al. (2024)), eliminating the issue of       test set and present the results in Table 5. We see that the
finding the right order in an unsupervised manner.                    Top probability margin-based adaptive MDM inference strat-
                                                                      egy (49.88%) again significantly outperforms ARMs trained
We compare ARMs and MDMs for Sudoku in Table 2 and
                                                                      with order information (32.57%). In particular, although the
Zebra puzzles in Table 3. We observe that for both, Top
                                                                      accuracy drops for both methods due to the more challeng-
probability margin-based adaptive MDM inference not only
                                                                      ing test set, MDMs with adaptive inference appear to be
outperforms the ARM trained without ordering information,
                                                                      more robust to this distribution shift than ARMs. We believe
but it even outperforms the ARM trained with ordering
                                                                      this is due to the fact that MDMs try to solve a significantly
information! This shows that the unsupervised way of
                                                                      higher number of infilling problems than ARMs (exp(L)
finding the correct order and solving such logic puzzles
                                                                      compared to L) and therefore are able to extract knowledge
using adaptive MDM inference outperforms the supervised
                                                                      about the problem more efficiently than ARMs.
way of finding the correct order and solving such puzzles
using an ARM, and is significantly less computationally
intensive.                                                            5. Conclusion
                                                                      In this work, we examined the impact of token generation
4.4. Adaptive MDM inference on natural language tasks                 order on training and inference in MDMs. We provided
To examine the effect of different inference strategies on text       theoretical and experimental evidence that MDMs train on
benchmarks, we adapted LLaDA, the 8B MDM model from                   hard masking problems. We also demonstrated that adaptive
(Nie et al., 2025). We compare three inference strategies:            inference strategies can be used to sidestep these hard prob-
vanilla, top probability, and top probability margin. The             lems. For logic puzzles, we find that this leads to dramatic

                                                                  9
                  Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions


                Table 4. Performance of different inference strategies for LLaDa 8B model on coding and math tasks.

 Method                HumanEval-Single         HumanEval-Multi            HumanEval-Split        Math     MMLU        ROCStories
 Vanilla                      31.8%                    16.5%                     14.2%           28.5%      33.2%        21.23%
 Top probability              32.9%                    20.8%                     18.4%           31.3%      36.5%        21.10%
 Top prob. margin             33.5%                    25.4%                     22.3%           34.3%      35.4%        21.41%


                                                                       References
Table 5. Comparison of accuracy for solving the hard Sudokus.
                                                                       Alaoui, A. E. and Gamarnik, D. Hardness of sampling
   Method                         #Param        Accuracy                 solutions from the symmetric binary perceptron. arXiv
   ARM (with ordering)               42M         32.57 %                 preprint arXiv:2407.16627, 2024.

   MDM (random)                                  3.62 %                Alekhnovich, M. More on average case vs approximation
   MDM (Top probability)             6M           9.44 %                 complexity. In 44th Annual IEEE Symposium on Foun-
   MDM (Top prob. margin)                        49.88 %                 dations of Computer Science, 2003. Proceedings., pp.
                                                                         298–307. IEEE, 2003.

                                                                       Aubin, B., Perkins, W., and Zdeborová, L. Storage capacity
                                                                         in symmetric binary perceptrons. Journal of Physics A:
improvements in performance not just over vanilla MDMs,                  Mathematical and Theoretical, 52(29):294003, 2019.
but even over ARMs trained with teacher forcing to learn the
right order of decoding. An important direction for future             Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den
work is to go beyond the relatively simple adaptive strate-              Berg, R. Structured denoising diffusion models in discrete
gies to find a better generation order like top probability and          state-spaces. NeruIPS, 2021.
top probability margin considered here.
                                                                       Bavarian, M., Jun, H., Tezak, N., Schulman, J., McLeavey,
                                                                         C., Tworek, J., and Chen, M. Efficient training of
                                                                         language models to fill in the middle, 2022. URL
                                                                         https://arxiv.org/abs/2207.14255.
Acknowledgements. JK thanks Kiwhan Song for discus-
sions about MDM training. KS and VK are supported by                   Bormashenko, O. A coupling argument for the random
the NSF AI Institute for Foundations of Machine Learning                 transposition walk. arXiv preprint arXiv: 1109.3915,
(IFML). KS and VK thank the computing support on the                     2011.
Vista GPU Cluster through the Center for Generative AI
(CGAI) and the Texas Advanced Computing Center (TACC)                  Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman, W. T.
at UT Austin. KS thanks Nishanth Dikkala for the initial                 Maskgit: Masked generative image transformer. CVPR,
discussions about the project. SK acknowledges: this work                2022.
has been made possible in part by a gift from the Chan
Zuckerberg Initiative Foundation to establish the Kempner              Chen, H. and Ying, L. Convergence analysis of discrete dif-
Institute for the Study of Natural and Artificial Intelligence           fusion model: Exact implementation through uniformiza-
and support from the Office of Naval Research under award                tion. arXiv preprint arXiv: 2402.08095, 2024.
N00014-22-1-2377. SC is supported by the Harvard Dean’s
                                                                       Chen, X., Chi, R. A., Wang, X., and Zhou, D. Premise order
Competitive Fund for Promising Scholarship and thanks
                                                                         matters in reasoning with large language models. arXiv
Brice Huang and Sidhanth Mohanty for enlightening discus-
                                                                         preprint arXiv:2402.08939, 2024.
sions about computational-statistical tradeoffs for planted
CSPs.                                                                  Decelle, A., Krzakala, F., Moore, C., and Zdeborová, L.
                                                                         Asymptotic analysis of the stochastic block model for
Impact statement                                                         modular networks and its algorithmic applications. Phys.
                                                                         Rev. E, 84:066106, Dec 2011.
This paper advances the understanding of discrete diffu-
sion models, contributing to the broader field of Machine              Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT:
Learning. There are many potential societal consequences                 Pre-training of deep bidirectional transformers for lan-
of our work, none of which we feel must be specifically                  guage understanding. In Proceedings of the 2019 Confer-
highlighted here.                                                        ence of the North American Chapter of the Association for

                                                                  10
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

  Computational Linguistics: Human Language Technolo-                  Lehnert, L., Sukhbaatar, S., Su, D., Zheng, Q., McVay, P.,
  gies, Volume 1 (Long and Short Papers), pp. 4171–4186,                 Rabbat, M., and Tian, Y. Beyond a*: Better planning with
  2019.                                                                  transformers via search dynamics bootstrapping. 2024.

Gamarnik, D. The overlap gap property: A topological bar-              Liao, Y., Jiang, X., and Liu, Q. Probabilistically masked
  rier to optimizing over random structures. Proceedings of              language model capable of autoregressive generation in
  the National Academy of Sciences, 118(41):e2108492118,                 arbitrary word order. In Proceedings of the 58th Annual
  2021.                                                                  Meeting of the Association for Computational Linguistics,
                                                                         pp. 263–274. Association for Computational Linguistics,
Golovneva, O., Allen-Zhu, Z., Weston, J., and Sukhbaatar,                2020.
  S. Reverse training to nurse the reversal curse. arXiv
  preprint arXiv:2403.13799, 2024.                                     Liu, A., Broadrick, O., Niepert, M., and Broeck, G. V. d. Dis-
                                                                         crete copula diffusion. arXiv preprint arXiv:2410.01949,
Gong, S., Agarwal, S., Zhang, Y., Ye, J., Zheng, L., Li, M.,
                                                                         2024a.
 An, C., Zhao, P., Bi, W., Han, J., et al. Scaling diffu-
  sion language models via adaptation from autoregressive              Liu, S., Mohanty, S., and Raghavendra, P. On statistical
  models. arXiv preprint arXiv:2410.17891, 2024.                         inference when fixed points of belief propagation are
                                                                         unstable . In 2021 IEEE 62nd Annual Symposium on
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion proba-
                                                                         Foundations of Computer Science (FOCS), pp. 395–405.
  bilistic models. Advances in neural information process-
                                                                         IEEE Computer Society, 2022.
  ing systems, 33:6840–6851, 2020.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,               Liu, S., Nam, J., Campbell, A., Stärk, H., Xu, Y., Jaakkola,
  Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,            T., and Gómez-Bombarelli, R. Think while you generate:
 Welbl, J., Clark, A., et al. Training compute-optimal                   Discrete diffusion with planned denoising. arXiv preprint
  large language models. arXiv preprint arXiv:2203.15556,                arXiv:2410.06264, 2024b.
  2022.
                                                                       Loshchilov, I. and Hutter, F. Decoupled weight decay regu-
Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The             larization. arXiv preprint arXiv:1711.05101, 2017.
  curious case of neural text degeneration. arXiv preprint
                                                                       Lou, A., Meng, C., and Ermon, S. Discrete diffusion model-
  arXiv:1904.09751, 2019.
                                                                         ing by estimating the ratios of the data distribution. ICML,
Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B.,                2024.
  Berg, R. v. d., and Salimans, T. Autoregressive diffusion
  models. arXiv preprint arXiv:2110.02037, 2021a.                      Maurer, A., Pontil, M., and Romera-Paredes, B. The benefit
                                                                        of multitask representation learning. JMLR, 17(81):1–32,
Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., and Welling,         2016.
  M. Argmax flows and multinomial diffusion: Learning
  categorical distributions. NeurIPS, 2021b.                           Montanari, A. Estimating random variables from random
                                                                        sparse observations. European Transactions on Telecom-
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B.,                 munications, 19(4):385–403, 2008.
  Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and
  Amodei, D. Scaling laws for neural language models.                  Nie, S., Zhu, F., Du, C., Pang, T., Liu, Q., Zeng, G., Lin, M.,
  arXiv preprint arXiv:2001.08361, 2020.                                 and Li, C. Scaling up masked diffusion models on text.
                                                                         arXiv preprint arXiv:2410.18514, 2024.
Kim, J., Kwon, S., Choi, J. Y., Park, J., Cho, J., Lee, J. D.,
  and Ryu, E. K. Task diversity shortens the icl plateau.              Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J.,
  arXiv preprint arXiv:2410.05448, 2024.                                 Lin, Y., Wen, J.-R., and Li, C. Large language diffusion
                                                                         models. arXiv preprint arXiv:2502.09992, 2025.
Kitouni, O., Nolte, N. S., Williams, A., Rabbat, M., Boucha-
  court, D., and Ibrahim, M. The factorization curse: Which            Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., Li, Z., and Li,
  tokens you predict underlie the reversal curse and more.               C. Your absorbing discrete diffusion secretly models the
  Advances in Neural Information Processing Systems, 37:                 conditional distributions of clean data. arXiv preprint
  112329–112355, 2025.                                                   arXiv:2406.03736, 2024.

Krzakala, F. and Zdeborová, L. Hiding quiet solutions in              Papadopoulos, V., Wenger, J., and Hongler, C. Arrows
  random constraint satisfaction problems. Physical review               of time for large language models. arXiv preprint
  letters, 102(23):238701, 2009.                                         arXiv:2401.17505, 2024.

                                                                  11
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Peng, F. Z., Bezemek, Z., Patel, S., Yao, S., Rector-                    M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W.,
  Brooks, J., Tong, A., and Chatterjee, P. Path planning                 Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn,
  for masked diffusion model sampling. arXiv preprint                    A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez,
  arXiv:2502.03540, 2025.                                                V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S.,
                                                                         Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y.,
Radcliffe, D. G. 3 million sudoku puzzles with rat-                      Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog,
  ings, 2020. URL https://www.kaggle.com/                                I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi,
  dsv/1495975.                                                           K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R.,
Rector-Brooks, J., Hasan, M., Peng, Z., Quinn, Z., Liu, C.,              Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X.,
  Mittal, S., Dziri, N., Bronstein, M., Bengio, Y., Chatterjee,          Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur,
  P., et al. Steering masked discrete diffusion models via               M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S.,
  discrete denoising posterior prediction. arXiv preprint                and Scialom, T. Llama 2: Open foundation and fine-tuned
  arXiv:2410.08134, 2024.                                                chat models. arXiv preprint arXiv: 2307.09288, 2023.

                                                                       Tripuraneni, N., Jin, C., and Jordan, M. I. Provable meta-
Ruder, S. An overview of multi-task learning in deep neural
                                                                         learning of linear representations. ICML, 2021.
  networks. arXiv 1706.05098, 2017.
                                                                       Varma, H., Nagaraj, D., and Shanmugam, K. Glauber gen-
Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin,
                                                                         erative model: Discrete diffusion models via binary clas-
  E., Chiu, J., Rush, A., and Kuleshov, V. Simple and
                                                                         sification. arXiv preprint arXiv: 2405.17035, 2024.
  effective masked diffusion language models. Advances
  in Neural Information Processing Systems, 37:130136–                 Wang, X., Zheng, Z., Ye, F., Xue, D., Huang, S., and Gu, Q.
  130184, 2025.                                                         Diffusion language models are versatile protein learners.
                                                                        ICML, 2024.
Schiff, Y., Sahoo, S. S., Phung, H., Wang, G., Boshar, S.,
  Dalla-torre, H., de Almeida, B. P., Rush, A., Pierrot, T.,           Xu, M., Geffner, T., Kreis, K., Nie, W., Xu, Y., Leskovec,
  and Kuleshov, V. Simple guidance mechanisms for dis-                  J., Ermon, S., and Vahdat, A. Energy-based diffusion
  crete diffusion models. arXiv preprint arXiv:2412.10193,               language models for text generation. arxiv preprint arXiv:
  2024.                                                                 2410.21357, 2024.
Shah, K., Dikkala, N., Wang, X., and Panigrahy, R.                     Ye, J., Gao, J., Gong, S., Zheng, L., Jiang, X., Li, Z., and
  Causal language modeling can elicit search and rea-                    Kong, L. Beyond autoregression: Discrete diffusion for
  soning capabilities on logic puzzles. arXiv preprint                   complex reasoning and planning. arXiv preprint arXiv:
  arXiv:2409.10502, 2024.                                                2410.14157, 2024.
Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. K.             Zhang, P., Zeng, G., Wang, T., and Lu, W. Tinyllama: An
  Simplified and generalized masked diffusion for discrete               open-source small language model. arXiv preprint arXiv:
  data. NeurIPS, 2024.                                                   2401.02385, 2024.
Shih, A., Sadigh, D., and Ermon, S. Training and infer-                Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., and
  ence on any-order autoregressive models the right way.                 Zhang, Q. Masked diffusion models are secretly time-
  NeurIPS, 2022.                                                         agnostic masked models and exploit inaccurate categori-
                                                                         cal sampling. arXiv preprint arXiv:2409.02908, 2024.
Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J. R., Hes-
  tness, J., and Dey, N. Slimpajama: A 627b token cleaned              Zheng, L., Yuan, J., Yu, L., and Kong, L. A reparameter-
  and deduplicated version of redpajama, June 2023.                      ized discrete diffusion model for text generation. arXiv
                                                                         preprint arXiv:2302.05737, 2023.
Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and
  Ganguli, S. Deep unsupervised learning using nonequi-
  librium thermodynamics. ICML, 2015.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Er-
  mon, S., and Poole, B. Score-based generative modeling
  through stochastic differential equations. ICLR, 2021.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
  A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,
  Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen,

                                                                  12
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

A. Related works
Discrete diffusion models. (Continuous) diffusion models were originally built on continuous-space Markov chains with
Gaussian transition kernels (Sohl-Dickstein et al., 2015; Ho et al., 2020). This was later extended to continuous time through
the theory of stochastic differential equations (Song et al., 2021). In a similar vein, discrete diffusion models have emerged
from discrete-space Markov chains (Hoogeboom et al., 2021b). Specifically, (Austin et al., 2021) introduced D3PM with
various types of transition matrices. Later, Lou et al. (2024) proposed SEDD, incorporating a theoretically and practically
robust score-entropy objective. Additionally, Varma et al. (2024); Liu et al. (2024b) introduced novel modeling strategies
that classify tokens in a noisy sequence as either signal (coming from clean data) or noise (arising from the forward process).
In particular, Liu et al. (2024b) uses this to give a planner that adaptively determines which tokens to denoise. While this is
similar in spirit to our general discussion about devising adaptive inference strategies, we emphasize that their approach is
specific to discrete diffusions for which the forward process scrambles the token values, rather than masking them.

Masked diffusion models. Meanwhile, the absorbing transition kernel has gained popularity as a common choice due
to its better performance than other kernels. Building on this, Sahoo et al. (2025); Shi et al. (2024) aligned its framework
with continuous diffusion, resulting in a simple and principled training recipe, referring to it as Masked Diffusion Model.
Subsequent studies have explored various aspects of MDM. Gong et al. (2024) efficiently trained MDM via adaptation from
autoregressive models, scaling MDM up to 7B parameters. Zheng et al. (2024) interpreted MDMs as order-agnostic learners
and proposed a first-hitting sampler based on this insight. Ye et al. (2024); Gong et al. (2024) demonstrated that MDM
outperforms autoregressive models in reasoning and planning tasks, emphasizing its impact on downstream applications.
Nie et al. (2024) examined the scaling laws of MDM, while Xu et al. (2024); Liu et al. (2024a) identified limitations in
capturing coordinate dependencies when the number of sampling steps is small and proposed additional modeling strategies
to address this issue. Schiff et al. (2024) studied conditional generation using MDM and Rector-Brooks et al. (2024) tackled
the challenge of controlling generated data distributions through steering methodologies. Chen & Ying (2024) provided a
theoretical analysis showing that sampling error is small given accurate score function estimation.

Any-order reasoning. Even though language tasks generally have a natural order of “left-to-right” token generation,
in many tasks like planning, reasoning, and combinatorial optimization, the natural order of token generation can be
quite different from “left-to-right”. Even though prominent autoregressive-based language models achieve impressive
performance on various tasks, many works (Golovneva et al., 2024; Chen et al., 2024; Kitouni et al., 2025) have shown that
this performance is tied to the training order of the tasks and therefore can cause brittleness from it. For example, Chen
et al. (2024) showed that simply permuting the premise order on math tasks causes a performance drop of 30%. The reason
behind such brittleness regarding the ordering is the inherent “left-to-right” nature of the autoregressive models. Several
works (Liao et al., 2020) have tried to address this issue in the autoregressive framework. In particular, (Papadopoulos et al.,
2024) highlighted the significance of left-to-right ordering in natural language by comparing its likelihood to that of the
reverse (right-to-left) ordering.
Recently, discrete diffusion models have emerged as a promising approach for discrete data apart from autoregressive
models. Additionally, the order-agnostic training of discrete diffusion models opens up the multiple sampling paths during
the inference but it also faces some challenges during the training therefore, they seem a promising approach to elicit any
order reasoning. Zheng et al. (2023) proposed different ways of implementing an adaptive inference strategy for MDM but
a concrete understanding of why such an adaptive inference strategy is needed is still lacking. In this work, we explore
various aspects of vanilla MDM training and how adaptive MDM inference can mitigate the issues raised by vanilla MDM
training and elicit any order reasoning.
We also want to mention the concurrent work by Peng et al. (2025) that proposes an alternative adaptive inference strategy
by selecting F(θ, xt ) based on the BERT model or the denoiser itself. In particular, Peng et al. (2025) uses the BERT model
or the denoiser to obtain the uncertainty of a token and then uses Top-K to decide the positions to unmask it. In contrast to
their work, we disentangle the impact of token ordering on MDM training vs. MDM inference and provide a more complete
understanding of the motivations for and benefits of adaptive inference. Additionally, our results indicate drawbacks to
using Top-K strategy as opposed to Top-K margin in deciding which tokens to unmask when there are multiple values with
high probabilities.

Beyond autoregressive models. Efforts to learn the natural language using non-autoregressive modeling began with
BERT (Devlin et al., 2019). Non-causal approaches can take advantage of the understanding the text data representation.

                                                              13
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

(Chang et al., 2022) adopted a similar approach for learning image representations. Building on these intuitions, (Shih
et al., 2022; Hoogeboom et al., 2021a) proposed any-order modeling, which allows a model to generate in any desired order.
Shih et al. (2022) made the same observation that any-order models by default have to solve exponentially more masking
problems than autoregressive models. However, whereas our work shows that learning in the face of this challenging task
diversity can benefit the model at inference time, their work sought to alleviate complexity at training time by reducing the
number of masking problems that need to be solved.

B. Technical details from Section 3
Notations. Throughout this section, we use xi to denote the i-th coordinate of the vector x and z(j) to denote the j-th
                                                                  i
example. The i-th coordinate of the vector z(j) is denoted by z(j) .

B.1. Additional example: sparse parity observations
Example B.1 (Noisy sparse parity observations). Let m = 2, k ∈ N, and N 2 log N ≪ P ≤ N 0.49k . Fix noise rate η > 0 as
well as strings z(1), . . . , z(P ) sampled independently and uniformly at random from the set of k-sparse strings in {0, 1}N .
For each j ∈ [P ], define Oj (x) to be the distribution which places mass 1 − η on 1 (resp. 2) and mass η on 2 (resp. 1) if
P i        i
  i x z(j) is odd (resp. even). Note that for k = O(1), each of these observations is efficiently learnable by brute-force.

Below we show that for a certain range of masking fractions, a constant fraction of the masking problems for the correspond-
ing L&O distributions are computationally hard under the Sparse Learning Parity with Noise assumption (Alekhnovich,
2003). Formally we have:
Proposition B.2. Let 0 < α < 1 be an arbitrary absolute constant, and let η = 1/poly(N ) be sufficiently large. Let x be a
sample from a L&O distribution pdata with noisy parity observations as defined in Example B.1. Suppose each token is
independently masked with probability α, and M is the set of indices for the masked tokens. If 1 − 1/N ≤ α ≤ 1 − 1/2N ,
then under the Sparse Learning Parity with Noise (SLPN) assumption (see Definition B.3), with constant probability over
M , no polynomial-time algorithm can solve the resulting masking problem of predicting any of the masked tokens among
xπ(1) , . . . , xπ(N ) given x[M ].

We note that it is important for us to take the observations to be sparse parities and to leverage the Sparse Learning Parity
with Noise assumption. If instead we used dense parities and invoked the standard Learning Parity with Noise (LPN)
assumption, we would still get the hardness of masking problems, but the observations themselves would be hard to learn,
assuming LPN. This result is based on the following standard hardness assumption:
Definition B.3 (Sparse Learning Parity with Noise). Given input dimension N , noise parameter 0 < η < 1/2, and sample
size P , an instance of the Sparse Learning Parity with Noise (SLPN) problem is generated as follows:

   • Nature samples a random bitstring x from {0, 1}N
   • We observe P examples of the form (x(i), y(i)) where x(i) is sampled independently and uniformly at random from
     k-sparse bitstrings in {0, 1}N , and y is given by ϵi + ⟨x(i), x⟩ (mod 2), where ϵi is 1 with probability η and 0
     otherwise.

Given the examples {(x(i), y(i))}P
                                 i=1 , the goal is to recover x.

The SLPN assumption is that for any P = N (1−ρ)k/2 for constant 0 < ρ < 1, and any sufficiently large inverse polynomial
noise rate η, no poly(N )-time algorithm can recover x with high probability.

Proof of Proposition B.2. With probability at least 1 − (1 − 1/N )N ≥ Ω(1), all of the variable tokens xπ(i) for i ≤ N are
masked. Independently, the number of unmasked tokens among the observation tokens Oj is distributed as Bin(P, 1 − α), so
                                                           2
by a Chernoff bound, with probability at least 1 − e−Ω(P/N ) = 1 − 1/poly(N ) we have that at least P/4N = Ω(N log N )
observation tokens are unmasked. The masking problem in this case amounts to an instance of SLPN with input dimension
N and sample size in [Ω(N log N ), O(N 0.49k )]. Because of the lower bound on the sample size, prediction of xM
is information-theoretically possible. Because of the upper bound on the sample size, the SLPN assumption makes it
computationally hard. As a result, estimating the posterior mean on any entry of xM given the unmasked tokens is
computationally hard as claimed.

                                                             14
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

B.2. Additional example: random slab observations
Example B.4 (Random slab observations). Let m = 2 and P = γN 2 for constant γ > 0. Fix slab width β and vectors
                                             √ N (0, I). For each j ∈ [P ], define the corresponding observation Oj (x) to be
z(1), . . . , z(P ) sampled independently from
deterministically 1 if |⟨z(j), 2x − 1⟩| ≤ β N , and deterministically 0 otherwise.

In (Alaoui & Gamarnik, 2024), it was shown that stable algorithms (Definition B.7), which encompass many powerful
methods for statistical inference like low-degree polynomial estimators, MCMC, and algorithmic stochastic localiza-
tion (Gamarnik, 2021), are  √ unable to sample from the posterior distribution over a random bitstring conditioned on it
satisfying |⟨z(j), x⟩| ≤ β N for any Θ(N ) number of constraints z(1), . . . , z(P ′ ), provided P ′ is not too large that the
support of the posterior is empty. This ensemble is the well-studied symmetric perceptron (Aubin et al., 2019). The following
is a direct reinterpretation of the result of (Alaoui & Gamarnik, 2024):
Proposition B.5. Let pdata be a L&O distribution with random slab observations as defined in Example B.4, with parameter
γ > 0 and slab width β > 0. There exists a constant cβ > 0 such that for any absolute constant 0 < c < cβ , if
1 − cβ N/2P ≤ α ≤ 1 − cN/P and γ > cβ , the following holds. Let p′data denote √     the distribution given by independently
masking every coordinate in pdata with probability α. Then any (1 − Ω̃(1/ N ))-stable algorithm, even one not based
on masked diffusion, which takes as input a sample x′ from p′data and, with probability 1 − o(1) outputs a Wasserstein-
approximate3 sample from pdata conditioned on the unmasked tokens in x′ , must run in super-polynomial time.

The upshot of this is that any stable, polynomial-time masked diffusion sampler will, with non-negligible probability,
encounter a computationally hard masking problem at some point during the reverse process.
For the proof, we first formally define the (planted) symmetric Ising perceptron model:
Definition B.6. Let α, β > 0. The planted symmetric Ising perceptron model is defined as follows:

   • Nature samples σ uniformly at random from {±1}N

      √ each j = 1, . . . , P = ⌊αN ⌋, we sample z(j) independently from N (0, IN ) conditioned on satisfying |⟨z(j), σ⟩| ≤
   • For
     β N.

The goal is to sample from the posterior on σ conditioned on these observations {z(i)}P
                                                                                      i=1 .

Next, we formalize the notion of stable algorithms.
                                                                   √
Definition B.7. Given a matrix Z ∼ N (0, 1)⊗P ×N , define Zt = tZ + 1 − t2 Z ′ for independent Z ′ ∼ N (0, 1)⊗P ×N .
A randomized algorithm A which takes as input Z ∈ RP ×N and outputs an element of {±1}N is said to be tN -stable if
limN →∞ W2 (law(A(Z)), law(A(Zt ))) = 0.

As discussed at depth in (Gamarnik, 2021), many algorithms like low-degree polynomial estimators and Langevin dynamics
are stable.
Theorem B.8 (Theorem 2.1 in (Alaoui & Gamarnik, 2024)4 ). For any constant β > 0, there exists cβ > 0 such that the
following holds for all constants 0 < α < cβ . For tN ≤ 1 − Ω(log2 (n)/n2 ), any tN -stable randomized algorithm A
which takes as input Z = (z(1), . . . , z(P )) and outputs an element of {±1}N will√fail to sample from the posterior on σ
conditioned on Z in the symmetric Ising perceptron model to Wasserstein error o( N ).

Proof of Proposition B.5. By a union bound, with probability at least 1 − (1 − α)N ≥ 1 − cβ N 2 /P ≥ 1 − cβ /γ over a
draw x′ ∼ p′data , all of the xπ(i) tokens are masked. The number of unmasked tokens in x′ among the observations Oj is
distributed as Bin(P, 1 − α). By a Chernoff bound, this is in [3cN/4, 3cβ N/4] with at least constant probability. The claim
then follows immediately from Theorem B.8 above.

B.3. Proof outline of Proposition 3.3
To understand the proof idea, we consider the case where all the latent tokens are masked and some of the observation
tokens are unmasked. In this case, the prediction task reduces to learning to recover the latent tokens that are consistent
   3
    Here the notion of approximation is o(1)-closeness in Wasserstein-2 distance.
   4
    Note that while the theorem statement in (Alaoui & Gamarnik, 2024) refers to the non-planted version of the symmetric binary
perceptron, the first step in their proof is to argue that these two models are mutually contiguous in the regime of interest.

                                                              15
                  Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions




Figure 4. Overlap achieved by belief propagation initialized at ground truth versus random for planted CSP with k = 3, m = 3, and
g = NAE, for N = 10000 and varying choices of average degree D. DKS /K can be shown analytically to be 64, consistent with the
phase transition depicted. Plot suggests Dcond /K ≈ 50. By Prop. 3.3 this implies a range of masking fractions at which Ω(1) fraction of
masking problems are computationally hard.



with the observations. Intuitively, each observation provides some constraints and the task is to recover an assignment that
satisfies the constraints. This is reminiscent of Constraint Satisfaction Problems (CSPs). Indeed, to show the hardness result,
we use the rich theory developed for planted CSPs at the intersection of statistical physics and average-case complexity.
In a planted CSP, there is an unknown randomly sampled vector y of length N and, one is given randomly chosen Boolean
constraints which y is promised to satisfy, and the goal is to recover y as best as possible (see Definition B.9). Prior works
have shown the hardness of efficiently learning to solve the planted CSP problem (Krzakala & Zdeborová, 2009; Alaoui &
Gamarnik, 2024). We show the hardness of masking problems in L&O distributions based on these results. Consider the
ground truth latent tokens as the random vector y and each observation as a constraint. In this case, the problem of learning
to recover the latent tokens from the observation tokens reduces to recovery for the planted CSP.
There are precise predictions for the values of vocabulary size m and the number of observations for which the information-
theoretically best possible overlap and the best overlap achievable by any computationally efficient algorithm are different.
We show that these predictions directly translate to predictions about when masking problems become computationally
intractable:
As a simple example, let us consider sparse predicate observations with k = 2 and g(x′ , x′′ ) = 1[x′ ̸= x′′ ]. These can
be formally related to the well-studied problem of planted m-coloring. In the planted m-coloring, a random graph of
average degree D is sampled consistent with an unknown vertex coloring and the goal is to estimate the coloring as well
as possible (Krzakala & Zdeborová, 2009), as measured by the overlap of the output of the algorithm to the ground-truth
coloring (see Definition B.9). As a corollary of our main result, we show that when all the latent tokens xπ(1) , . . . , xπ(N )
are masked and a few unmasked observation tokens provide the information of the form g(xπ(i) , xπ(j) ) = 1[xπ(i) ̸= xπ(j) ]
for i, j ≤ N , then solving the masking problem can be reduced to solving planted coloring.
For planted m-coloring, when m = 5 the thresholds in Proposition 3.3 are given by DKS /2 = 16 and Dcond /2 ≈
13.23 (Krzakala & Zdeborová, 2009) (the factor of 2 here is simply because the observations correspond to ordered subsets
of size 2). For general predicates and arities, there is an established recipe for numerically computing DKS and Dcond based
on the behavior of the belief propagation algorithm (see the discussion in Appendix B.4). As an example, in Fig. 4, we
execute this recipe for m = 3, k = 3, and g given by the Not-All-Equal predicate NAE(x′ , x′′ , x′′ ) = 1 − 1[x′ = x′′ = x′′′ ]
to obtain thresholds that can be plugged into Proposition 3.3.

Additional examples of the hardness. The above setup can also be generalized to capture Bayesian constraint satisfaction
problems (Montanari, 2008; Liu et al., 2022), one notable example of which is the stochastic block model (Decelle et al.,
2011). There are analogous predictions for the onset of hardness of inference, which can likewise be translated to hardness
of masking problems for seemingly benign L&O distributions. In Appendix B.1 and B.2, we give two more examples of
L&O distributions for which order-aware training is tractable yet order-agnostic training of the MDM is computationally

                                                                  16
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

hard.
First, we consider L&O distributions whose observations are sparse, noisy parities in the latents and deduce hardness for
order-agnostic training from the Sparse Learning Parity with Noise assumption (Alekhnovich, 2003). We then consider
L&O distributions whose observations are generalized linear models in the latents, and deduce hardness for a large class
of efficient algorithms from existing results on Lipschitz hardness (Alaoui & Gamarnik, 2024) for the symmetric binary
perceptron (Aubin et al., 2019).

B.4. Proof of Proposition 3.3: sparse predicate observations
Here we formally define the relevant notions needed to formalize our claim about hardness in Proposition 3.3.
Definition B.9 (Planted CSPs). Given arity k ∈ N, vocabulary/alphabet size m ∈ N, predicate g : {1, . . . , m}k → {0, 1},
latent dimension N , and clause density P/N , the corresponding planted constraint satisfaction problem is defined as
follows: Nature samples an unknown assignment σ uniformly at random from {1, . . . , m}N , and then for each ordered
k-tuple S of distinct elements from [N ], we observe the clause S independently with probability ϕ/N k−1 if g(σ|S ) = 1.
To measure the quality of an algorithm for recoveringP σ given the observations, define the overlap between an estimate
σ̂ and the ground truth σ by d(σ, σ̂) ≜ minπ∈SN i 1[σi = π(σ̂i )] where SN denotes the set of all permutations of
{0, 1, . . . , N − 1}. Define the average degree to be kP/N , i.e. the expected number of variables that share at least one
clause with a given variable.

We begin by defining the central algorithm driving statistical physics predictions about hardness for random constraint
satisfaction problems: belief propagation (BP).
Definition B.10 (BP update rules). Belief propagation is an algorithm that iteratively updates a set of messages
{MSi→S
     c    [t], MSS→i
                 c   [t]}, where i, S range over all pairs of variable indices i ∈ [N ] and observations S ∋ i. At time
t + 1, the messages are computed via
                                                    Y
                               MSi→Sc   [t + 1] ∝         MSTc →i [t]                                                (4)
                                                   T :i∈T ̸=S
                                                        X                          Y
                                MSS→i
                                  c   [t + 1] ∝                      g(σ ∪i c)              MSσj→S
                                                                                                j
                                                                                                   [t] ,                      (5)
                                                   σ∈{1,...,m}S\i                j:i̸=j∈S


where σ ∪i c ∈ {1, . . . , m}S assigns c to entry i and σ to the remaining entries.
A set of messages can be used to estimate the marginals of the posterior on σ conditioned on theQ         observations as follows.
The marginal on the i-th variable has probability mass function over {1, . . . , m} proportional to { T :i∈T MSTc →i }. Given
a set of marginals, a natural way to extract an estimate for σ is to round to the color in {1, . . . , m} at which the probability
mass function is largest.

Throughout we will make the following assumption that ensures that the trivial messages MSi→S    c   = 1/m and MSS→i
                                                                                                                 c   =
1/m are a fixed point, sometimes called the paramagnetic fixed point, for the iteration above:
                                P
Assumption B.11. The quantity σ∈{1,...,m}[k] \i g(σ ∪i c) is constant across all c ∈ {1, . . . , m} and i ∈ [k].
Definition B.12. Given k, m, g, the Kesten-Stigum threshold DKS is defined to be the largest average degree for which BP
is locally stable around the paramagnetic fixed point, that is, starting from a small perturbation of the paramagnetic fixed
point, it converges to the paramagnetic fixed point. More formally, DKS is the largest average degree at which the Jacobian
of the BP operator {MSi→S [t]} 7→ {MSi→S [t + 1]} has spectral radius less than 1.
The condensation threshold Dcond is defined to be the largest average degree at which the planted CSP ensemble and the
following simple null model become mutually contiguous and thus statistically indistinguishable as N → ∞. The null
model is defined as follows: there is no single unknown assignment, but instead for every ordered subset S of k variables,
Nature independently samples an unknown local assignment σS ∈ {1, . . . , m}S , and the observation is included with
probability ϕ/N k−1 if g(σS ) = 1.

For Dcond < kP/N < DKS , there exists some other fixed point of the BP operator whose marginals, once rounded to an
assignment, achieves strictly higher overlap than does BP with messages initialized randomly. The prediction is that in this
regime, no efficient algorithm can achieve optimal recovery (Krzakala & Zdeborová, 2009).

                                                                17
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

Conjecture B.13 (1RSB cavity prediction). Suppose k, m, g satisfy Assumption B.11, and let DKS and Dcond denote the
associated Kesten-Stigum and condensation thresholds for the average degree. Then for all P for which Dcond < kP/N <
DKS , the best overlap achieved by a computationally efficient algorithm for recovering σ is strictly less than the best overlap
achievable.

Proof of Proposition 3.3. At masking fraction α satisfying the bounds in the Proposition, with probability at least αN ≥
(1 − γ −1 DKS /N k−1 )N ≥ Ω(1) we have that all tokens corresponding to latents xπ(i) get masked. Independently of this, the
number of unmasked tokens among the observation tokens OS is distributed as Bin(N (N − 1) · · · (N − k + 1), 1 − α), so by
standard binomial tail bounds, with constant probability (depending on the gap between Dcond and DKS ) this lies between
γ −1 Dcond N/k and γ −1 DKS N/k. Furthermore, of these unmasked tokens in expectation γ fraction of them correspond to
observations for which the associated predicate evaluates to 1. Conditioned on the above events, the masking problem thus
reduces exactly to inference for a planted constraint satisfaction problem at average degree Dcond < D < DKS , from which
the Proposition follows.

C. Experimental details in Section 3
C.1. Experimental details in Section 3.2
π-learner configurations. We consider two distributions of π that interpolate between Unif (SL ) where SL denote the
uniform distribution over all permutations of indices {0, 1, . . . , L − 1} and the point mass at the identical distribution:
(Closer) and (Much-closer). To construct those distributions, we start from the identity permutation and perform a certain
number of random swapping operations. Since L log(L) √     number of swaps results in a distribution that is very close to
Unif (SL ) (Bormashenko, 2011), we use L/10 and L swaps to construct the (Closer) and (Much-closer) distributions,
respectively. For consistency, we repeat this sampling process three times.

Model and training configurations. As explained in Section 3.2, to evaluate the scaling law of the π-learner, we can
simply adapt the autoregressive training setup (a transformer with causal attention) by modifying the input to π(x0 ) and
using a learnable positional embedding layer instead of RoPE. We borrow the training configurations from (Nie et al., 2024),
which are also consistent with the TinyLlama (Zhang et al., 2024) configurations. In particular, we use AdamW optimizer
(Loshchilov & Hutter, 2017), setting β1 = 0.9, β2 = 0.95, and a weight decay of 0.1 and L = 2048. A cosine learning rate
schedule is applied, with a maximum learning rate of 4 × 10−4 and a minimum learning rate of 4 × 10−5 . We also note that
unless otherwise specified, we maintain the same training configuration throughout the paper.

Examining scaling laws. We conduct IsoFLOP analysis (Hoffmann et al., 2022). For a given number of FLOPs C, by
varying the number of non-embedding parameters of transformers, we set the iteration numbers so that the total number of
tokens observed by the model during training equals C/6N , following prior studies (Hoffmann et al., 2022; Kaplan et al.,
2020). We then select the smallest validation loss and set it as a data point.

C.2. Experimental details in Section 3.3
C.2.1. E XPERIMENT ON L&O-NAE-SAT DISTRIBUTION
We consider the L&O-NAE-SAT distribution with (N, P ) = (20, 280). For each example sequence from L&O-NAE-SAT,
we pad the last 212 tokens with an additional token value of 2. We employ a 19M MDM with RoPE and a maximum
sequence length of 512. Then, this MDM is trained for 2 × 103 iterations. To attain a proxy MDM for the Bayes optimal
predictor, we further train it for 5 × 104 iterations.
To measure the error across different tasks, we consider the following setup. For each ℓ ∈ [1, N − 1], we randomly mask
ℓ tokens in the latent positions and ℓ × (P/N ) tokens in the observed positions. Across all masked prediction positions,
ℓ(1 + P/N ), we measure the error for each position. For certainty, we repeat this process 1000 times. The result in Figure 2
corresponds to the case when ℓ = 11, and we observe the same tendency for other values of ℓ.

C.2.2. E XPERIMENT ON TEXT DATA
We take a 170M MDM pretrained with text data for a baseline model. To measure the performance imbalance between
likelihood modeling tasks

                                                              18
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

                                                "L−1                                              #
                                                                                             
                                                                     π(i)
                                                 X
                                   Ex0 ∼pdata          log pθ       x0 x0 [π{i, . . . , L − 1}]       .
                                                 i=0

As done in the experiments in Section 3.2, we sample πs from three different distributions: Unif(SL ), (Closer), the point
mass of identical distribution. For each case, we calculate the expectation over 1024 samples of x0 ∼ pdata .

D. Experimental details in Section 4
D.1. Experimental details in Section 4.2
D.1.1. E XPERIMENT ON L&O-NAE-SAT DISTRIBUTION
We consider five instances of L&O-NAE-SAT: (N, P ) = (25, 275), (30, 270), (40, 260), (50, 250), (100, 200). For each
distribution, we train a 19M MDM and measure the accuracy difference between vanilla inference and adaptive inference
using top probability margin.

D.1.2. E XPERIMENT ON TEXT DATA
Top probability margin sampler with temperature. To modify our inference for text data modeling, which does not
have a determined answer, we found that adding a certain level of temperature to the oracle is useful. This is because the
top probability margin or the top probability often leads to greedy sampling, which harms the diversity (entropy) of the
generated samples. Therefore, we consider a variant of the oracle as follows, incorporating a Gaussian noise term ϵ.

                              F(θ, xt ) = Top K |pθ (xi = j1 |xt ) − pθ (xi = j2 |xt )| + ϵ .
                                                                                           

Note that this approach has also been employed for unconditional sampling (Wang et al., 2024; Zheng et al., 2023).

Generative perplexity and entropy. We employ a 1.1B MDM pretrained on text data as a baseline. For each sampling
step, we unconditionally generate samples using both vanilla and adaptive inference. Next, we calculate the likelihood
                                                                                                                 P using
LLama2-7B as a baseline large language model. Moreover, we denote the entropy of a generated sample x as pi log pi ,
where pi = #{xi = i}/L.

Choice of number of tokens to unmask. We set the number of tokens to unmask K so that the number of unmasked tokens
matches that of vanilla MDM inference in expectation. For an inference transition from step t to s, vanilla MDM expects
                                       s −αt
(# mask tokens in the current xt ) × α1−α  t
                                             unmasked. Accordingly, we choose K = (# mask tokens in the current xt ) ×
 αs −αt
 1−αt   . This choice keeps the number of revealed tokens balanced throughout inference. Alternatively, one can sample K
                                                               s −αt
stochastically from Binom(# mask tokens in the current xt , α1−α   t
                                                                     ). We found that both the deterministic and stochastic
choices of K result in comparable generative perplexity.
This choice of K can be potentially helpful when the network is time-conditioned, since this keeps
(#mask tokens in the current xt ) ≈ (1 − αt ) × L where L is the max sequence length–matching the marginal that the
model saw during training.

D.2. Experimental details on Sudoku and Zebra puzzles
Dataset. For both Sudoku and Zebra puzzles, we use the dataset provided in Shah et al. (2024) to train our model. To
evaluate our model on the same difficulty tasks, we use the test dataset proposed in Shah et al. (2024). This dataset is created
by filtering the puzzles from (Radcliffe, 2020) that can be solved using a fixed list of 7 strategies. To create a hard dataset
to evaluate easy-to-hard generalization, we use the remaining puzzles from (Radcliffe, 2020) as they either require a new
strategy unseen during the training and/or require backtracking. The hard dataset contains around 1M Sudoku puzzles.

Model, training, and inference. For the training and inference, we use the codebase of (Ye et al., 2024) with keeping
most of the hyperparameters default given in the codebase. For the Sudoku dataset, we use 6M GPT-2 model, and for the
Zebra dataset, we use 19M model. We set the learning rate to 0.001 with a batch size of 128 to train the model for 300
epochs. For the inference, we use 50 reverse sampling steps using the appropriate strategy. Additionally, we add Gumbel
noise with a coefficient of 0.5 to the MDM inference oracle F.

                                                                     19
                   Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

D.3. Experimental details on LLaDA-8B
Our evaluation covers two task categories: (i) infilling(HumanEval-Infill and ROCStories) and (ii) instruction–answering
(Math). For instruction–answering tasks, we employ a semi-autoregressive sampling strategy, whereas for infilling tasks we
retain the non-autoregressive approach. For infilling tasks, the output length is predetermined—matching the size of the
masked span—whereas instruction–answering tasks require an explicit length specification. For the latter, we follow the
sampling configuration of (Nie et al., 2025).
For HumanEval-Infill, we adopt the problem set introduced by (Bavarian et al., 2022). Each instance is grouped by the span
of the masked code—the region the model must infill—into three categories: single-line, multi-line, and split. The task
difficulty rises as the length of the masked span increases.

E. Omitted proofs
Proof of Proposition 2.1. We build on Proposition 3.1 from (Zheng et al., 2024) to obtain the result of Proposition 2.1.
We first re-state the result from (Zheng et al., 2024) for the case when the denoising network pθ does not depend on the
noise-scale t explicitly. Let x(n) be a sequence with n tokens being masked from x0 , and xi (n) denotes the ith token value
of the sequence x(n). Let q̃(x(n)|x0 ) be the probability distribution corresponding to randomly and uniformly masking n
tokens of x0 .



Proposition E.1 (Proposition 3.1 of (Zheng et al., 2024)). For clean data x0 , let q̃(x(n) | x0 ) be the discrete forward
process that randomly and uniformly masks n tokens of x0 . Suppose the noise schedules αt satisfies α0 = 0 and α1 = 1.
Then, the MDM training loss (1) can be reformulated as

                                                                                                                
                                                     L
                                                                          1
                                                     X                              X
                                        Lθ = −                   E                            log pθ (xℓ0 | x(n)) .                                   (6)
                                                            x(n)∼q̃(·|x0 ) n
                                                     n=1                        ℓ:xℓ (n)=0




                                                                     x(n) ∼ q̃(· | x0 ). Since there are total L positions
To obtain an alternative formulation of (6), we expand the expectation
of x0 , we have the probability assigned for each x(n) equals 1/ Ln . Therefore, expanding the above equation with the
expectation x(n) and treating x(n) as x[M ] for some set M of size n, we obtain the result.

                                                              X          1           1
                                                Lθ = −                   L
                                                                             ·          log pθ (xℓ0 | x[M ]).
                                                                        |M |
                                                                                    |M |
                                                           M ∈[L],i∈M




E.1. Equivalence between the MDM loss and any-order autoregressive loss
In this section, we will demonstrate the equivalence for MDM loss and any-order autoregressive loss. In particular, for all
x0 , we show

                                                                                         
                         L−1
                         X                  
                                                 π(j)
                                                                                                    X           1        1
       −       E                  log pθ       x0 x0 [π{j}, . . . , π{L − 1}]            =−
                                                                                                                L
                                                                                                                              log pθ (xi0 |x0 [M ]).
                                                                                                                         |M |
                                                                                                                     
           π∼Unif(SL )
                             j=0                                                                 M ⊆[L],i∈M    |M |



We now consider {π(j), . . . , π(L − 1)} = M ⊆ [L] and π(j) = i and count the number of π ∈ SL that induces a specific
term log pθ (xi0 |x0 [M ]). To induce the term, for a given M ∈ [L] and i ∈ M , π must satisfy

                                                      π(j) = i,    {π(j), . . . , π(L − 1)} = M.

                                                                               20
                 Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions

The number of π that satisfies above is (L − |M |)! × (|M | − 1)!. Using this and the number of total permutations is L!, we
obtain the result.
                                                                                       
                                          L−1                                        
                                                        π(j)
                                           X
                                  E          log pθ x0 x0 [π{j}, . . . , π{L − 1}] 
                              π∼Unif(SL )
                                             j=0
                                                L−1
                              1       X         X             
                                                                π(j)
                                                                                              
                          =                             log pθ x0 x0 [π{j}, . . . , π{L − 1}]
                              L!
                                   π∈Unif(SL ) j=0
                            1         X
                                                    log pθ (xi0 |x0 [M ]) × (L − 1 − |M |)! × (|M | − 1)!
                                                                                                        
                          =
                            L!
                                   M ∈[L],i∈M
                                   X            1        1
                          =                  L
                                                             log pθ (xi0 |x0 [M ]).
                                                        |M |
                                                    
                              M ∈[L],i∈M    |M |




                                                                       21
