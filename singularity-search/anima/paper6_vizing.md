# paper6 vizing

> **Original PDF Size:** 1020K
> **Text Size:** 204K
> **Conversion Date:** 2026-01-22 07:34:34

---

                                                                 Vizing’s Theorem in Near-Linear Time
                                               Sepehr Assadi∗           Soheil Behnezhad†            Sayan Bhattacharya‡              Martı́n Costa§
                                                                           Shay Solomon¶                 Tianyi Zhang‖
arXiv:2410.05240v3 [cs.DS] 14 Oct 2025




                                                                                             Abstract
                                                   Vizing’s theorem states that any n-vertex m-edge graph of maximum degree ∆ can be edge
                                               colored using at most ∆ + 1 different colors [Vizing, 1964]. Vizing’s original proof is algorithmic
                                               and shows that such
                                                                 √ an edge coloring can be found in O(mn) time. This was subsequently
                                               improved to Õ(m n) time, independently by [Arjomandi, 1982] and by [Gabow et al., 1985].1
                                                   Very recently, independently and concurrently, using randomization, this runtime bound was
                                               further improved to Õ(n2 ) by [Assadi, 2024] and Õ(mn1/3 ) by [Bhattacharya, Carmon, Costa,
                                               Solomon and Zhang, 2024] (and subsequently to Õ(mn1/4 ) by [Bhattacharya, Costa, Solomon
                                               and Zhang, 2024]).
                                                   In this paper, we present a randomized algorithm that computes a (∆ + 1)-edge coloring in
                                               near-linear time—in fact, only O(m log ∆) time—with high probability, giving a near-optimal
                                               algorithm for this fundamental problem.




                                         ∗
                                           University of Waterloo, sepehr@assadi.info
                                         †
                                           Northeastern University, s.behnezhad@northeastern.edu
                                         ‡
                                           University of Warwick, s.bhattacharya@warwick.ac.uk
                                         §
                                           University of Warwick, martin.costa@warwick.ac.uk
                                         ¶
                                           Tel Aviv University, solo.shay@gmail.com
                                         ‖
                                           ETH Zürich, tianyi.zhang@inf.ethz.ch
                                         1
                                             Throughout, we use Õ(f ) := O(f polylog (n)) to suppress log-factors in the number of vertices of the graph.
Contents
1 Introduction                                                                                         1
  1.1 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     2
  1.2 Technical Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     2
2 Basic Notation                                                                                       7
3 Showcase: Our Algorithm Instantiated on Bipartite Graphs                                 7
  3.1 Our Bipartite Color Extension Algorithm in Lemma 3.2 . . . . . . . . . . . . . . . . 8
  3.2 Analysis of the Bipartite Color Extension Algorithm: Proof of Lemma 3.2 . . . . . . 9
  3.3 Extension to General Graphs: Roadmap for the Rest of the Paper . . . . . . . . . . 13
4 Preliminaries: Vizing Fans and Vizing Chains                                                        14
5 Basic Building Blocks                                                                               15
  5.1 U-Fans and U-Edges . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      15
  5.2 Data Structures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   17
  5.3 Vizing Fans in Separable Collections . . . . . . . . . . . . . . . . . . . . . . . . . . .      18
6 The Main Algorithm                                                                                  19
7 The Algorithm Construct-U-Fans: Proof of Lemma 6.2                                                  20
  7.1 The Subroutine Prune-Vizing-Fans . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      21
  7.2 The Subroutine Reduce-U-Edges . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       24
  7.3 Analysis of Construct-U-Fans: Proof of Lemma 6.2 . . . . . . . . . . . . . . . . . . .          31
8 The Algorithm Color-U-Fans: Proof of Lemma 6.3                                                      32
  8.1 The Subroutine Prime-U-Fans . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       33
  8.2 The Subroutine Activate-U-Fans . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      37
  8.3 Analysis of Color-U-Fans: Proof of Lemma 6.3 . . . . . . . . . . . . . . . . . . . . . .        38
9 Implementation and Data Structures                                                         38
  9.1 Implementing the Operations from Section 5.2 . . . . . . . . . . . . . . . . . . . . . 40
10 The Final Algorithm: Proof of Theorem 1.1                                                         40
   10.1 Fine-Grained Variants of Lemma 6.3 and Lemma 6.4 . . . . . . . . . . . . . . . . . 41
   10.2 Proof of Theorem 1.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
A Vizing’s Theorem for Multigraphs in Near-Linear Time                                                49
  A.1 Vizing Fans and Separable Collection in Multigraphs . . . . . . . . . . . . . . . . . .         49
  A.2 Proof of Theorem A.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      51
  A.3 Proof Sketch of Lemma A.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       52
B Shannon’s Theorem for Multigraphs in Near-Linear Time                                               54
  B.1 Shannon Fans . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    54
  B.2 Proof of Theorem B.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      55
  B.3 Proof of Lemma B.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      56
C Edge Coloring Algorithms for Small Values of ∆                                                 58
  C.1 Proof of Corollary C.1 for Simple Graphs . . . . . . . . . . . . . . . . . . . . . . . . 58
  C.2 Proof of Corollary C.1 for Multigraphs . . . . . . . . . . . . . . . . . . . . . . . . . . 60
D A Specialized Concentration Inequality                                                              60
1    Introduction
Given a simple undirected graph G = (V, E) on n vertices and m edges, as well as an integer
κ ∈ N+ , a κ-edge coloring χ : E → {1, 2, . . . , κ} of G assigns a color χ(e) to each edge e ∈ E so
that any two adjacent edges receive distinct colors. The minimum possible value of κ for which a
κ-edge coloring exists in G is known as the edge chromatic number of G. If G has maximum vertex
degree ∆, any proper edge coloring would require at least ∆ different colors. A classical theorem
by Vizing shows that ∆ + 1 colors are always sufficient [Viz64]. Moreover, it was proven by [Hol81]
that it is NP-complete to distinguish whether the edge chromatic number of a given graph is ∆ or
∆ + 1, and therefore ∆ + 1 is the best bound we can hope for with polynomial time algorithms.
    Vizing’s original proof easily extends to an O(mn) time algorithm, which was improved to
     √
Õ(m n) in the 1980s by [Arj82] and [GNK+ 85] independently. More recently, the algorithms
of [Arj82, GNK+ 85] were simplified in [Sin19], while shaving off extra logarithmic factors from their
                                              √
runtime complexities, achieving a clean O(m n) runtime bound. Very recently, this longstanding
     √
O(m n) time barrier was bypassed in two concurrent works [Ass25] and [BCC+ 24] which improved
the runtime bound to two incomparable bounds of Õ(n2 ) and Õ(mn1/3 ), respectively. In a follow-up
work, the Õ(mn1/3 ) runtime bound of [BCC+ 24] to was further improved to Õ(mn1/4 ) in [BCSZ25].
    In this work, we resolve the time complexity of randomized (∆ + 1)-edge coloring up to at most
a log factor by presenting a near-linear time algorithm for this problem.

Theorem 1.1. There is a randomized algorithm that, given any simple undirected graph G = (V, E)
on n vertices and m edges with maximum degree ∆, finds a (∆+1)-edge coloring of G in O(m log n)
time with high probability.

Remarks: Several remarks on our Theorem 1.1 are in order:
• Our algorithm in Theorem 1.1 does not rely on any of the recent developments in [Ass25, BCC+ 24,
  BCSZ25] and takes an entirely different path. We present an overview of our approach, as well
  as a comparison to these recent developments in Section 1.2.
• Our main contribution in this work is to improve the time-complexity of (∆ + 1)-edge coloring
  by polynomial factors all the way to near-linear. For this reason, as well as for the sake of
  transparency of our techniques, we focus primarily on presenting an O(m log3 n) time randomized
  algorithm (Theorem 6.1), which showcases our most novel ideas. Later in Section 10, we show
  that a more careful implementation of the same algorithm achieves a clean O(m log n) runtime.
• We can additionally use a result of [BD23] to further replace the log n term in Theorem 1.1
  with a log ∆ term, leading to an algorithm for (∆ + 1)-edge coloring in O(m log ∆) time with
  high probability (Corollary C.1). This matches the longstanding time bound for ∆-edge coloring
  bipartite graphs [COS01] (which, to the best of our knowledge, is also the best known runtime
  for (∆ + 1)-edge coloring bipartite graphs). This bound is also related to a recent line of work
  in [BD23, BD24, Dha24b] that focused on the ∆ = no(1) case and gave a randomized (∆ + 1)-
  coloring algorithm that runs in O(m∆4 log ∆) time with high probability [BD24].
• Vizing’s theorem generalizes for (loop-less) multigraphs, asserting that any multigraph with edge
  multiplicity at most µ can be (∆ + µ)-edge colored [Viz64, Viz65]. A related result is Shannon’s
  theorem [Sha49] that asserts that any multigraph can be ⌊3∆/2⌋ edge colored independent of
  µ; both these bounds are tight: see the so-called Shannon multigraphs [Viz65]. We show that
  our techniques extend to these theorems as well, giving O(m log ∆) time algorithms for both
  problems (see Theorems A.1 and B.1 and Corollary C.1).


                                                  1
1.1   Related Work
In addition to algorithms for Vizing’s theorem, there has also been a long line of work on fast
algorithms for edge coloring that use more than ∆ + 1 colors. It was
                                                                 √ first shown in [KS87] that an
edge coloring can be computed in Õ(m) time when we have ∆+ Õ( ∆) different colors. In addition,
there are algorithms which run in linear or near-linear time for (1 + ϵ)∆-edge coloring [DHZ19,
BCPS24c, EK24, BD24, Dha24b] when ϵ ∈ (0, 1) is a constant. Most recently, it was shown in
[Ass25] that even a (∆ + O(log n))-edge coloring can be computed in O(m log ∆) expected time.
    There are other studies on restricted graph classes. In bipartite graphs, a ∆-edge coloring
can be computed in Õ(m) time [CH82, COS01, Alo03, GKK10]. In bounded degree graphs, one
can compute a (∆ + 1)-edge coloring in Õ(m∆) time [GNK+ 85], and it was generalized recently
for bounded arboricity graphs [BCPS24b]; see also [BCPS24a, CRV24, Kow24] for further recent
results on edge coloring in bounded arboricity graphs. Subfamilies of bounded arboricity graphs,
including planar graphs, bounded tree-width graphs and bounded genus graphs, were studied in
[CY89, CN90, CK08].
    Beside the literature on classical algorithms, considerable effort has been devoted to the study
of edge coloring in various computational models in the past few years, including dynamic [BM17,
BCHN18, DHZ19, Chr23, BCPS24c, Chr24], online [CPW19, BGW21, SW21, KLS+ 22, BSVW24,
BSVW25, DGS25], distributed [PR01, EPS14, FGK17, GKMU18, GP20, BBKO22, CHL+ 20, Ber22,
Chr23, Dav23], and streaming [BDH+ 19, SB24, CMZ24, GS24] models, among others.

1.2   Technical Overview
We now present an overview of prior approaches to (∆+1)-edge coloring and describe our techniques
at a high level. For the following discussions, we will assume basic familiarity with Vizing’s proof
and its underlying algorithm; see Section 4 for more details on Vizing’s algorithm.

Prior Approaches: A generic approach for (∆ + 1)-edge coloring, dating back to Vizing’s proof
itself, is to extend a partial (∆ + 1)-edge coloring of the graph one edge at a time, possibly by
recoloring some edges, until the entire graph becomes colored. As expected, the main bottleneck in
the runtime of this approach comes from extending the coloring to the last few uncolored edges. For
instance, given a graph G = (V, E) with maximum degree ∆, we can apply Eulerian partitions to
divide G into two edge-disjoint subgraphs G = G1 ∪ G2 with maximum degrees at most ⌈∆/2⌉. We
then find (⌈∆/2⌉ + 1)-edge colorings of the subgraphs G1 and G2 recursively. Directly combining
the colorings of G1 and G2 gives a coloring of G with ∆+3 colors, so we have to uncolor a 2/(∆+3)
fraction of the edges—amounting to O(m/∆) edges—and try to extend the current partial (∆ + 1)-
edge coloring to these edges. Thus, the “only” remaining part is to figure out a way to color these
final O(m/∆) edges, and let the above approach take care of the rest.
    This task of extending the coloring to the last Θ(m/∆) edges is the common runtime bottleneck
of all previous algorithms. Vizing’s original algorithm [Viz64] gives a procedure to extend any partial
coloring to an arbitrary uncolored edge (u, v) by rotating some colors around u and flipping the
colors of an alternating path starting at u. The runtime of this procedure would be proportional to
the size of the rotation, usually called a Vizing fan, and the length of the alternating path, usually
called a Vizing chain, which are bounded by ∆ and n respectively. As such, the total runtime for
coloring the remaining O(m/∆) edges using Vizing fans and Vizing chains will be O(mn/∆) time.
   As one can see, flipping long alternating paths is the major challenge in Vizing’s approach. To
improve this part of the runtime, [GNK+ 85] designed an algorithm that groups all uncolored edges


                                                  2
into O(∆2 ) types depending on the two colors of the Vizing chain induced by this edge. Since all
the Vizing chains of the same type are vertex-disjoint, they can be flipped simultaneously and their
total length is only O(n). This means that the runtime of coloring all edges of a single type can be
bounded by O(n) as well. This leads to an O(n∆2 ) time algorithm for handling all O(∆2 ) types; a
more careful analysis can bound this even with O(m∆) time. Finally, balancing the two different
                                                                   √
bounds of O(mn/∆) and O(m∆) yields a runtime bound of O(m n) for coloring O(m/∆) edges,
                        √
which leads to an Õ(m n) time algorithm using the above framework.
                                                                                                 √
    There has been some very recent progress that broke through this classical barrier of O(m n)
time in [GNK+ 85, Arj82]. In [BCC+ 24], the authors speed up the extension of the coloring to
uncolored edges when these edges admit a small vertex cover. They then show how to precondition
the problem so that uncolored edges admit a small vertex cover, leading to a Õ(mn1/3 ) time
algorithm. In [Ass25], the author avoided the need for Eulerian partition and recursion altogether by
instead designing a new near-linear time algorithm for (∆+O(log n))-edge coloring. This algorithm
borrows insights from sublinear matching algorithms in regular bipartite graphs by [GKK10] and
is thus completely different from other edge coloring algorithms mentioned above. By using this
algorithm, finding a (∆ + 1)-edge coloring directly reduces to the color extension problem with
O((m log n)/∆) uncolored edges (by removing the colors of a Θ(log n)/∆ fraction of the edges
in the (∆ + O(log n))-edge coloring to obtain a partial (∆ + 1)-edge coloring first). Applying
Vizing’s procedure for these uncolored edges takes additional Õ(mn/∆) = Õ(n2 ) time, leading to
an Õ(n2 ) time algorithm for (∆ + 1)-edge coloring. Finally, in [BCSZ25], the authors showed that
a (∆ + 1)-coloring can be computed in Õ(mn1/4 ) time, by using the algorithm of [Ass25] for initial
coloring of √the graph and then presenting an improved color extension subroutine with a runtime
        2
of Õ(∆ + ∆n) for coloring each remaining edge; the best previous color extension time bounds
were either the trivial O(n) bound or the bound Õ(∆4 ) by [Ber22, BD24].
Our Approach: A Near-Linear Time Color Extension Algorithm
We will no longer attempt to design a faster color extension for a single edge, and instead color
them in large batches like in [GNK+ 85], which allows for a much better amortization of runtime in
coloring multiple edges. This ultimately leads to our main technical contribution: a new randomized
algorithm for solving the aforementioned color extension problem for the last O(m/∆) edges in
Õ(m) time. With this algorithm at hand, we can follow the aforementioned Eulerian partition
approach and obtain a (∆ + 1)-edge coloring algorithm whose runtime T (m) follows the recursion
T (m) ≤ 2T (m/2) + Õ(m) with high probability; this implies that T (m) = Õ(m), hence, giving
us a near-linear time randomized algorithm for (∆ + 1)-edge coloring. This way, we will not even
need to rely on the (∆ + O(log n))-edge coloring algorithm of [Ass25] to color the earlier parts of
the graph (although one can use that algorithm instead of Eulerian partition approach to the same
effect).
    We now discuss the main ideas behind our color extension algorithm. In the following, it helps
to think of the input graph as being near-regular (meaning that the degree of each vertex is Θ(∆)),
and thus the total number of edges will be m = Θ(n∆); this assumption is not needed for our
algorithm and is only made here to simplify the exposition.

Color Type Reduction: Recall that the runtime of O(n∆2 ) for the color extension algorithm
of [GNK+ 85] is due to the fact that there are generally O(∆2 ) types of alternating paths in the graph
and that the total length of the paths of each color type is bounded by O(n) edges. However, if it
so happens that the existing partial coloring only involves O(∆) color types instead, then the same
algorithm will only take O(n∆) = O(m) time (by the near-regularity assumption). The underlying
idea behind our algorithm is to modify the current partial edge coloring (without decreasing the

                                                  3
number of uncolored edges) so that the number of color types reduces from O(∆2 ) to O(∆) only.
    To explore this direction, let us assume for now the input graph is bipartite, which
greatly simplifies the structure of Vizing fans and Vizing chains, thus allowing us to convey the
key ideas more clearly (see Section 3 for a more detailed exposition); later we highlight some of the
key challenges that arise when dealing with general graphs. We shall note that it has been known
since the 80s that one can ∆-edge color bipartite graphs in Õ(m) time [CH82, COS01]. However,
the algorithms for bipartite graphs use techniques that are entirely different from Vizing fans and
Vizing chains, and which do not involve solving the color extension problem at all. In particular,
prior to this work, it was unclear whether one can efficiently solve the color extension problem in
bipartite graphs. Therefore, the assumption of a bipartite input graph does not trivialize our goal of
using Vizing fans and Vizing chains for efficiently solving the color extension problem. Additionally,
we can assume that in the color extension problem, the last O(m/∆) edges to be colored can be
partitioned into O(1) matchings (this guarantee follows immediately from the recursive framework
we outlined earlier), and that we deal with each of these matchings separately. In other words, we
can also assume that the uncolored edges are vertex-disjoint.
    Let χ be a partial (∆ + 1)-edge coloring, and for any vertex w ∈ V , let missχ (w) ⊆ [∆ + 1] be
the set of colors missing from the edges incident to w under χ. Given any uncolored edge (u, v),
the color type of this edge (u, v) would be {cu , cv } for some arbitrary choices of cu ∈ missχ (u) and
cv ∈ missχ (v) (it is possible for an edge to be able to choose more than one color type, but we fix
one arbitrary choice among them); in other words, if we flip the {cu , cv }-alternating path starting
at u, then we can assign χ(u, v) to be cv . To reduce the total number of different color types
to O(∆), we would have to make some color types much more popular : at the beginning, a type
spans an Ω(1/∆2 ) proportion of the uncolored edges but we would like to have a type spanning an
Ω(1/∆) proportion. For this purpose, we fix an arbitrary color type {α, β}, and want to modify χ
to transform the type of an arbitrary uncolored edge (u, v) from {cu , cv } to {α, β} – we call this
popularizing the edge (u, v). To do this, we can simply flip the {α, cu }-alternating path Pu starting
at u and the {β, cv }-alternating path Pv starting at v.
    There are two technical issues regarding this path-flipping approach. Firstly, the alternating
paths Pu and Pv could be very long, and require a long time for being flipped. More importantly,
flipping Pu and Pv could possibly damage other {α, β}-type (uncolored) edges that we popularized
before. More specifically, say that we have popularized a set Φ of uncolored edges. When popu-
larizing the next uncolored edge (u, v), it could be the case that the {α, cu }-alternating path Pu is
ending at a vertex u′ for some edge (u′ , v ′ ) ∈ Φ. If we flip the path Pu , then (u′ , v ′ ) would no longer
be of {α, β}-type as α would not be missing at u′ anymore. See Figure 1 for an illustration.
    Our key observation is that when |Φ| is relatively small, most choices for an alternating path Pu
cannot be ending at edges in Φ. Consider the above bad example where Pu is a {cu , α}-alternating
path ending at u′ for some (u′ , v ′ ) ∈ Φ. Let us instead look at this from the perspective of the {α, β}-
type edge (u′ , v ′ ) ∈ Φ. For any (u′ , v ′ ) ∈ Φ and any color γ, there can be at most one uncolored
edge (u, v) ∈/ Φ whose corresponding path Pu is the same {γ, α}-alternating path starting at u′ ;
this is because any vertex belongs to at most one alternating path of a fixed type (here, the type
{γ, α}) and each vertex belongs to at most one uncolored edge (recall that the uncolored edges form
a matching). This is also true for {γ, β}-type edges for the same exact reason. As such, ranging
over all possible choices of γ ∈ [∆ + 1], there are at most O(|Φ|∆) uncolored edges (u, v) whose
alternating paths could damage the set Φ. Therefore, as long as the size of Φ is a o(1/∆) fraction
(or more precisely, an O(1/∆) fraction, for a sufficiently small constant hiding in the O-notation)
of all uncolored edges, the following property holds: a constant fraction of uncolored edges e ∈       /Φ
can be popularized using the above method without damaging Φ. See Figure 2 for an illustration.

                                                      4
         β         λ           β       v             u   α           γ   α           γ     u′ popular v ′

                                   cv = λ       cu = γ                                   cu′ = α     cv′ = β



         λ         β           λ       v   popular   u   γ           α   γ           α     u′          v′

                                   cv = β       cu = α                                   cu′ = γ     cv′ = β


Figure 1: In this picture, we attempt to popularize edge (u, v) by flipping the {α, γ}-alternating
path from u and the {β, λ}-alternating path from v. However, flipping the {α, γ}-alternating path
from u makes a previously popular edge (u′ , v ′ ) unpopular as u′ will not miss color α anymore.

             u1           v1       β                             β

             u2           v2       β                             β

             u3           v3       β                             β              v′              u′

             u4           v4       β                             β

             u5           v5       β                             β


Figure 2: In this picture, (u′ , v ′ ) ∈ Φ with cu′ = α, cv′ = β. For each uncolored edge (ui , vi ), flipping
the {ci , β}-alternating path from vi would damage the property that cv′ = β. Fortunately, there are
at most ∆ many different such (ui , vi ) because each of them is at the end of an {β, ·}-alternating
path starting at v ′ .


    This property resolves both technical issues raised earlier simultaneously. Let λ denote the
number of uncolored edges. For the second issue, as long as |Φ| = o(λ/∆), we can take a random
uncolored edge (u, v) ∈  / Φ and flip Pu and Pv if this does not damage any already popularized
edge in Φ; by the observation above, this happens with constant probability. For the first issue, we
can show that the expected length of alternating paths Pu and Pv , for the random uncolored edge
picked above, is O(m/λ); indeed, this is because the number of edges colored α or β is O(n), hence
the total length of all alternating paths with one color being fixed to α or β is O(m). All in all, the
total time we spend to popularize a single type {α, β} to become a Θ(1/∆) fraction of all uncolored
edges is O(λ/∆ · m/λ) = O(m/∆). Since coloring edges of a single type takes O(n) = O(m/∆)
time by our earlier discussion, we can color in this way a Θ(1/∆) fraction of all uncolored edges in
O(m/∆) expected time. As a direct corollary, we can color all uncolored edges in O(m log n) time,
hence solving the color extension problem, in (near-regular) bipartite graphs, in near-linear time.
The above argument will be detailed in the proof of Lemma 3.2 in Section 3 (see also Lemma 6.3
of Section 8 for the analogous but not identical argument in general graphs).

Collecting U-Fans in General Graphs: We now discuss the generalization of our scheme above
to non-bipartite graphs. Existence of odd cycles in non-bipartite graphs implies that we can no
longer assign a color type {cu , cv } to an uncolored edge (u, v) for cu ∈ missχ (u), cv ∈ missχ (v), and


                                                             5
          u1                                                                u1

                                                                                 α                 w1
                 α            α w1                                                        α
v1                                      x α y                      v1                                      x α y
                 α            α                                                           α
v2                                                                 v2
                                  w2                                             α                 w2

          u2                                                                u2



Figure 3: In this picture, we have two different uncolored edges (u1 , v1 ), (u2 , v2 ) such that α ∈
missχ (u1 ) ∩ missχ (u2 ), and their Vizing chains first intersect at edge (x, y) which currently has
color α under χ. Then we can rotate their Vizing fans and flip part of their Vizing chains to shift
(u1 , v1 ), (u2 , v2 ) to (w1 , x), (w2 , x) respectively to form a u-fan; note that α ∈ missχ (w1 ) ∩ missχ (w2 )
after this shifting procedure.


hope that flipping the {cu , cv }-alternating path from u allows us to color the edge (u, v) with cv
(because the path may end at v, and thus after flipping it cv will no longer be missing from v). This
is where Vizing fans and Vizing chains come into play: in non-bipartite graphs, a color type of an
uncolored edge (u, v) is {cu , γu,v } where cu is an arbitrary color in missχ (u) but γu,v is determined
by the Vizing fan around u and the alternating path that we take for coloring (u, v) (or the Vizing
chain of (u, v)). Thus, while as before we can switch cu with some fixed color α, it is unclear how
to flip alternating paths to change γu,v to β also, in order to popularize the edge (u, v) to be of
some designated type {α, β}, without damaging another popularized edge as a result.
    To address this challenge, we rely on the notion of a u-fan, introduced by [GNK+ 85], which is
the non-bipartite graph analogue of an uncolored edge in bipartite graphs. A u-fan of type {α, β} is
a pair of uncolored edges (u, v), (u, w) such that α ∈ missχ (u) and β ∈ missχ (u)∩missχ (v). Consider
the {α, β}-alternating path starting at u. As at least one of v or w is not the other endpoint of this
alternating path (say v), flipping this path allows us to assign the color β to (u, v). Consequently,
as u-fans are similar to edges in bipartite graphs, we can still essentially (but not exactly) apply our
color type reduction approach if all the uncolored edges are paired as u-fans. Therefore, it suffices
to modify χ to pair all uncolored edges together to form u-fans.
   In order to pair different uncolored edges together and form u-fans, we should first be able
to move uncolored edges around. Such operations already appeared in some previous work on
dynamic edge coloring [DHZ19, Chr23, Chr24]. Basically, for any uncolored edge (u, v), we can
modify χ to shift this uncolored edge to any position on its Vizing chain. This naturally leads to a
win-win situation: if the Vizing chain is short, then (u, v) can be colored efficiently using Vizing’s
procedure; otherwise if most Vizing chains are long, then there must be a pair of Vizing chains
meeting together after a few steps, so we can shift two uncolored edges to form a u-fan efficiently.
    Let us make this a bit more precise. Fix an arbitrary color α and consider the set Uα of all
the uncolored edges (u, v) such that α ∈ missχ (u) and the respective Vizing chain is of type {α, ·}.
Also, assume there are mα edges colored α under χ. If most {α, ·}-Vizing chains have length larger
than Ω(mα /|Uα |), then on average, two Vizing chains should meet within the first O(mα /|Uα |)
steps; in this case, we can repeatedly pick two intersecting Vizing chains and create a u-fan by
shifting their initial uncolored edges to the intersection of these Vizing chains; see Figure 3 for
illustration. Given the length of the chains, this takes O(mα /|Uα |) time. Otherwise, the average

                                                        6
cost of applying Vizing’s color extension procedure is O(mα /|Uα |), and in this case we can directly
color all those edges in O(mα ) time. Summing over all different α ∈ [∆ + 1] gives a near-linear
runtime. The above argument will be detailed in the proof of Lemma 6.2 in Section 7.
    The above discussion leaves out various technical challenges. For instance, moving around un-
colored edges as described above breaks the assumption that the uncolored edges form a matching.
Handling this requires dedicating different colors from missχ (u) for every uncolored edge incident
on a vertex u. This is formalized via the notion of separability in Section 5.1. Additionally, we have
ignored all algorithmic aspects of (efficiently) finding pairs of intersecting Vizing chains, as well as
the corner cases of Vizing fan intersections and fan-chain intersections. We defer the discussions
on these details to the actual proofs in subsequent sections.


2    Basic Notation
Let G = (V, E) be graph on n vertices with m edges and maximum degree ∆ and let χ : E →
[∆ + 1] ∪ {⊥} be a (partial) (∆ + 1)-edge coloring of G. We refer to edges e ∈ E with χ(e) = ⊥
as uncolored. Given a vertex u ∈ V , we denote the set of colors that are not assigned to any edge
incident on u by missχ (u). We sometimes refer to missχ (u) as the palette of u. We say that the
colors in missχ (u) are missing (or available) at u.
    Given a path P = e1 , . . . , ek in G, we say that P is an {α, β}-alternating path if χ(ei ) = α
whenever i is odd and χ(ei ) = β whenever i is even (or vice versa). We say that the alternating
path P is maximal if one of the colors α or β is missing at each of the endpoints of P . We refer
to the process of changing the color of each edge ei ∈ P with color α (resp. β) to β (resp. α) as
flipping the path P . We denote by |P | the length (i.e., the number of edges) of the alternating
path P . We define the length i prefix of the path P to be the path P≤i := e1 , . . . , ei .
     Consider a set U ⊆ E of edges that are uncolored under χ, i.e., χ(e) = ⊥ for all e ∈ U . We use
the phrase “extending χ to U ” to mean the following: Modify χ so as to ensure that χ(e) ̸= ⊥ for
all e ∈ U , without creating any new uncolored edges. When the set U consists of a single edge e
(i.e., when U = {e}), we use the phrase “extending χ to the edge e” instead of “extending χ to U ”.
   Our algorithms will always work by modifying a partial coloring χ; unless explicitly specified
otherwise, every new concept we define (such as u-fans and separable collection in Section 5) will
be defined with respect to this particular partial coloring χ.


3    Showcase: Our Algorithm Instantiated on Bipartite Graphs
In this section, we instantiate our algorithm on bipartite graphs to showcase some of our key
insights, and outline a proof of Theorem 3.1 below.

Theorem 3.1. There is a randomized algorithm that, given a bipartite graph G = (V, E) with
maximum degree ∆, returns a ∆-edge coloring of G in Õ(m) time with high probability.

    As explained in Section 1.2, focusing on bipartite graphs allows us to ignore the technical issues
that arise while dealing with Vizing fans. At the same time, this does not trivialize the main
conceptual ideas underpinning our algorithm. In particular, we prove Theorem 3.1 via Lemma 3.2
below, which gives a specific color-extension algorithm on bipartite graphs. Although near-linear
time ∆-edge coloring algorithms on bipartite graphs existed since the 1980s [CH82, COS01], to the
best of our knowledge there was no known algorithm for Lemma 3.2 prior to our work.

                                                   7
Lemma 3.2. Let χ : E → [∆] ∪ {⊥} be a partial ∆-edge coloring in a bipartite graph G = (V, E),
and let U ⊆ E be a matching of size λ such that every edge e ∈ U is uncolored in χ. Furthermore,
suppose that we have access to an “auxiliary data structure”, which allows us to detect in Õ(1)
time the two least common colors α, β ∈ [∆] in χ.2 Then there is a randomized algorithm that can
extend χ to Ω(λ/∆) many edges of U in Õ(m/∆) time with high probability.

      Let us start by showing that Lemma 3.2 implies Theorem 3.1 easily.

Proof of Theorem 3.1. We follow the strategy outlined in Section 1.2. Given a bipartite graph G,
we first find an Eulerian partition of the graph to partition the edges of G into two subgraphs of
maximum degree ⌈∆/2⌉ each, and color them recursively using different colors. This leads to a
2 · ⌈∆/2⌉ ≤ (∆ + 2) edge coloring of G. We then form a partial ∆ edge coloring χ by uncoloring
the two color classes of this (∆ + 2)-edge coloring with the fewest edges assigned to them, which
leaves us with two edge-disjoint matchings U1 and U2 to color. This is our previously mentioned
color extension problem. To solve this problem, we apply Lemma 3.2 to U1 first to extend χ to
Ω(1/∆) fraction of it, and keep applying this lemma to the remaining uncolored edges of U1 until
they are all colored. We then move to U2 in the same exact way and extend χ to its edges as well,
obtaining a ∆-coloring of the entire G as a result.
    The correctness of the algorithm follows immediately by induction. The runtime can also be
analyzed as follows. When coloring U1 (or U2 ), each invocation of Lemma 3.2 reduces the number
of uncolored edges in U1 (or U2 ) by a (1 − Ω(1/∆)) factor and thus we apply this lemma a total
of O(∆ · log n) time. Moreover, each application of Lemma 3.2 takes Õ(m/∆) time with high
probability. Thus, with high probability, it only takes Õ(m) time to extend the coloring χ to U1
and U2 in the color extension problem. Hence, the runtime of the algorithm, with high probability,
follows the recurrence T (m, ∆) ≤ 2T (m/2, ∆/2) + Õ(m), and thus itself is Õ(m) time.
    Finally, note that with O(m) preprocessing time, we can maintain access to the “auxiliary
data structure” throughout the repeated invocations of Lemma 3.2 above: All we need to do is to
maintain a counter for each color γ ∈ [∆], which keeps track of how many edges in G are currently
assigned the color γ in χ. We maintain these counters in a balanced search tree, and update the
relevant counter whenever we change the color assignment of an edge in G.

      The rest of this section is dedicated to the proof of Lemma 3.2.

3.1      Our Bipartite Color Extension Algorithm in Lemma 3.2
At a high level, our algorithm for Lemma 3.2 consists of the following three steps.

1. Pick the two least common colors α, β ∈ [∆] in ∆. This implies that there are at most O(m/∆)
   edges in G that are colored with α or β in χ.

2. Modify the coloring χ so that Ω(λ/∆) of the edges (u, v) ∈ U either receive a color under χ, or
   have α ∈ missχ (u) and β ∈ missχ (v). While implementing this step, we ensure that the total
   number of edges with colors α or β remains at most O(m/∆).

3. Let Φ denote the set of edges (u, v) ∈ U with α ∈ missχ (u) and β ∈ missχ (v). We call these
   edges popular. We extend χ to a constant fraction of the edges in Φ, by flipping a set of maximal
   {α, β}-alternating paths.
  2
      Specifically, for every γ ∈ [∆] \ {α, β} and γ ′ ∈ {α, β}, we have |{e ∈ E | χ(e) = γ} | ≥ |{e ∈ E | χ(e) = γ ′ }|.


                                                              8
    We now formalize the algorithm; the pseudocode is provided in Algorithm 1. As input, we are
given a bipartite graph G = (V, E), a partial ∆-edge coloring χ of G, and a matching U ⊆ E of
uncolored edges of size λ.
    The algorithm starts by fixing the two least common colors α, β ∈ [∆] in χ. The main part
is the while loop in Line 3, which runs in iterations. In each iteration of the while loop, the
algorithm samples an edge e = (u, v) from U u.a.r. and attempts to either (1) directly extend the
coloring χ to e (see Line 10), which adds e to a set C ⊆ U of colored edges in U or (2) modify
χ so that α ∈ missχ (u) and β ∈ missχ (v)—we refer to this as making the edge (u, v) popular —,
which adds e to a set Φ ⊆ U of popular edges (see Line 20). The attempt to modifying χ is done
by essentially finding a maximal {cu , α}-alternating path Pu starting at u and a {cv , β}-alternating
path Pv starting at v for cu ∈ missχ (u) and cv ∈ missχ (v) (see Line 12 and Lines 14 and 15). The
modification itself is done only if Pu and Pv do not intersect any other popular edge already in Φ.
    We say that the concerned iteration of the while loop FAILS if it chooses an already colored
edge in C (Line 6), or modifying the color leads to an already popular edge in Φ to no longer
remain popular (Line 17); otherwise, we say the iteration succeeds. As stated earlier, the algorithm
maintains a subset Φ ⊆ U of popular edges, and a subset of edges C ⊆ U that got colored since the
start of the while loop. Initially, we have C = Φ = ∅. Thus, the quantity |Φ| + |C| denotes the
number of successful iterations of the while loop that have been performed so far. The algorithm
performs iterations until |Φ| + |C| = Ω(λ/∆), and then it proceeds to extend the coloring χ to at
least a constant fraction of the edges in Φ by finding {α, β}-alternating paths for edges in Φ that
admits such paths (see the for loop in Line 22).

3.2   Analysis of the Bipartite Color Extension Algorithm: Proof of Lemma 3.2
We start by summarizing a few key properties of Algorithm 1.
Claim 3.3. Throughout the while loop in Algorithm 1, there are at most O(m/∆) edges in G that
receive either the color α or the color β, under χ.

Proof. We start with O(m/∆) such edges in G and each successful iteration of the while loop
increases the number of such edges by O(1), and there are O(λ/∆) = O(m/∆) such iterations.

Lemma 3.4. Throughout the execution of the while loop in Algorithm 1, the following conditions
hold: (i) the set C consists of all the edges in U that are colored under χ; (ii) for every edge
(u, v) ∈ Φ, we have α ∈ missχ (u) and β ∈ missχ (v).

Proof. Part (i) of the lemma follows from Line 2 and Line 10. For part (ii), consider an edge
e = (u, v) that gets added to Φ. This happens only after flipping the paths Pu and Pv in Line 19.
Just before we execute Line 19, the following conditions hold:

   • α ̸= β (see Line 1).
   • cu ∈ missχ (u) and cv ∈ missχ (v) (see Line 8).
   • cu ̸= cv , cu ̸= β and cv ̸= α (see Line 12).
   • If α ∈ missχ (u) then Pu = ∅ (see Line 14), and if β ∈ missχ (v) then Pv = ∅ (see Line 15).
   • The path Pu (resp. Pv ) does not end at that a vertex that is incident on some edge in Φ \ {e}
     (see Lines 16 and 17), although it might possibly end at v (resp. u).

                                                     9
 Algorithm 1: BipartiteExtension(G, χ, U )
 1 Let α, β ∈ [∆] be the two least common colors in χ   // We have α ̸= β
 2 Initialize Φ ← ∅, C ← ∅, and set λ ← |U |
 3 while |Φ| + |C| < λ/(10∆) do
 4    Sample an edge e = (u, v) ∼ U independently and u.a.r.
 5    if (u, v) ∈ Φ ∪ C then
 6        The iteration FAILS
 7        go to Line 3
 8    Identify (arbitrarily) two colors cu ∈ missχ (u) and cv ∈ missχ (v)
 9    if cu = cv then
10        Set χ(u, v) ← cu and C ← C ∪ {(u, v)}
11        go to Line 3
12    if cu = β or cv = α then
13        Set (u, v) ← (v, u)    // Now cu ̸= cv , cu ̸= β, cv ̸= α (see Lines 9 and 12)
14    Let Pu be the maximal {cu , α}-alternating path starting at u (Pu = ∅ if α ∈ missχ (u))
15    Let Pv be the maximal {cv , β}-alternating path starting at v (Pv = ∅ if β ∈ missχ (v))
16    if either Pu or Pv ends at a node that is incident on some edge in Φ \ {e} then
17        The iteration FAILS
18    else
19        Modify χ by flipping the alternating paths Pu and Pv
20        Set Φ ← Φ ∪ {(u, v)} // Now α ∈ missχ (u) and β ∈ missχ (v)
21 Φ′ ← Φ
22 for each edge e = (u, v) ∈ Φ′ do
23    Φ′ ← Φ′ \ {e}
24    W.l.o.g., suppose that α ∈ missχ (u) and β ∈ missχ (v)
25    if there exists a color c ∈ {α, β} such that c ∈ missχ (u) ∩ missχ (v) then
26        χ(u, v) ← c
27    else
28        Let Pu⋆ be the maximal {α, β}-alternating path starting at u
29        (Since G is bipartite, α ∈ missχ (u) and β ∈ missχ (v), Pu⋆ does not end at v)
30        Modify χ by flipping the alternating path Pu⋆ , and set χ(u, v) ← β
31        if the path Pu⋆ ends at a node that is incident on some edge in e′ ∈ Φ′ \ {e} then
32            Φ′ ← Φ′ \ {e′ }




                                               10
From these conditions, it follows that the paths Pu and Pv are edge-disjoint, and after we flip them
in Line 19, we have α ∈ missχ (u) and β ∈ missχ (v) in Line 20.
     In subsequent iterations of the while loop, the only places where we change the coloring χ are
Lines 10 and 19. Since the edges in U form a matching, changing the coloring in Line 10 cannot
affect whether or not the edge (u, v) ∈ Φ remains popular (i.e., has α ∈ missχ (u) and β ∈ missχ (v)).
Finally, during a subsequent iteration of the while loop where we sample an edge (u′ , v ′ ) ∼ U , we
flip the paths Pu′ , Pv′ in Line 19 only if their endpoints are not incident on any edges in Φ\{(u′ , v ′ )}
(see Line 16), and in particular, on u or v. Thus, this operation cannot change what colors are
available at u and v, and so cannot change whether or not the edge (u, v) ∈ Φ remains popular.

   We use the following lemma to bound the number of iterations of the while loop in Algorithm 1.

Lemma 3.5. Each iteration of the while loop in Algorithm 1 increases the value of |Φ| + |C| by
an additive one, with probability at least 1/2 and otherwise keep it unchanged.

Proof. Fix any given iteration of the while loop. At the start of this iteration, we sample an edge
from U u.a.r. We say that an edge e ∈ U is bad if the iteration FAILS when we sample e (see Line 6
and Line 17), and good otherwise. Note that if we sample a good edge e ∈ U , then the iteration
either adds one edge to the set Φ (see Line 20), or adds one edge to the set C (see Line 10). In
other words, if we sample a good (resp. bad) edge e ∈ U at the start of the iteration, then this
increases in the value of |Φ| + |C| by one (resp. keeps the value of |Φ| + |C| unchanged). We will
show that at most λ/2 edges in U are bad. Since |U | = λ, this will imply the lemma.
    To see why this claimed upper bound on the number of bad edges holds, first note that there
are (|Φ| + |C|) many bad edges that cause the iteration to FAIL in Line 6. It now remains to bound
the number of bad edges which cause the iteration to FAIL in Line 17.
    Towards this end, note that for each edge (u′ , v ′ ) ∈ Φ, there are at most 4∆ many maximal {α, ·}-
or {β, ·}-alternating paths that end at either u′ or v ′ . Furthermore, each such alternating path has
its other endpoint incident on at most one edge in U since the edges in U form a matching. Thus,
for each edge (u′ , v ′ ) ∈ Φ, there are at most 4∆ many edges f(u′ ,v′ ) ∈ U that satisfy the following
condition: Some alternating path constructed by the algorithm after sampling f(u′ ,v′ ) ends at either
u′ or v ′ (see Line 14 and Line 15). Each such edge f(u′ ,v′ ) is a bad edge which causes the iteration
to FAIL in Line 17, and moreover, only such edges are causing the iteration to FAIL in Line 17.
Thus, the number of such bad edges is at most |Φ| · 4∆.
    To summarize, the total number of bad edges is at most (|Φ| + |C|) + |Φ| · 4∆ < λ/2, where the
last inequality holds since |Φ| + |C| < λ/(10∆) (see Line 3). This concludes the proof.

   Similarly, we can bound the expected runtime of each iteration of the while loop in Algorithm 1.

Lemma 3.6. Each iteration of the while loop in Algorithm 1 takes Õ(m/λ) time in expectation,
regardless of the outcome of previous iterations.

Proof. Alternating path flips can be done in time proportional to the path lengths using standard
data structures, so we only need to analyze the path lengths. Fix any given iteration of the while
loop. At the start of this iteration, we can classify the edges in U into one of the following three
categories: An edge e ∈ U is of “Type I” if the iteration ends at Line 7 when we sample e, is of
“Type II” if the iteration ends at Line 11 when we sample e, and is of “Type III” otherwise. Let
λ1 , λ2 and λ3 respectively denote the total number of Type I, Type II and Type III edges, with
λ1 + λ2 + λ3 = λ. For every Type III edge e = (u, v) ∈ U , we refer to the alternating paths Pu


                                                    11
and Pv (see Line 14 and Line 15) as the “characteristic alternating paths” for e. Let P3 denote
the collection of characteristic alternating paths of all Type III edges. Since the set of Type III
edges is a subset of U , they form a matching, and hence different paths in P3 have different starting
points. Furthermore, every path in P3 is either a maximal {α, ·}-alternating path or a maximal
{β, ·}-alternating path. Accordingly, Claim 3.3 implies that the total length of all the paths in P3
is at most O((m/∆) · ∆) = O(m).
    Now, if at the start of the iteration, we happen to sample either a Type I or a Type II edge
e ∈ U , then the concerned iteration takes O(1) time. In the paragraph below, we condition on the
event that the edge e = (u, v) ∈ U sampled at the start of the iteration is of Type III.
    Using appropriate data structures, the time taken to implement the concerned iteration is
proportional (up to Õ(1) factors) to the lengths of the alternating paths Pu and Pv (see Line 14
and Line 15). The key observation is that for each x ∈ {u, v}, the path Px is sampled almost
uniformly (i.e., with probability Θ(1/λ3 )) from the collection P3 . Since the total length of all the
paths in P3 is O(m), it follows that the expected length of each of the paths Pu , Pv is O(m/λ3 ).
    To summarize, with probability λ3 /λ, we sample a Type III edge at the start of the concerned
iteration of the while loop, and conditioned on this event the expected time spent on that iteration
is Õ(m/λ3 ). In contrast, if we sample a Type I or a Type II edge at the start of the concerned
iteration, then the time spent on that iteration is O(1). This implies that we spend at most
Õ(m/λ3 ) · (λ3 /λ) + O(1) = Õ(m/λ) expected time per iteration of the while loop.

   Finally, we show that in the very last step of the algorithm, the for loop in Line 22, the algorithm
succeeds in coloring a constant fraction of popular edges.

Lemma 3.7. The for loop in Line 22 extends the coloring χ to at least half of the edges in Φ.

Proof. Consider any given iteration of the for loop where we pick an edge e = (u, v) ∈ Φ′ in Line 22,
where w.l.o.g. α ∈ missχ (u) and β ∈ missχ (v). It is easy to verify that during this iteration, we
successfully extend the coloring χ to e, either in Line 26 or in Line 30. In the latter case, we
crucially rely on the fact that the graph G is bipartite (see Line 29), and hence the maximal {α, β}-
alternating path Pu⋆ starting at u cannot end at v; in fact, this is the only place where we rely on
the biparteness of G. Lines 31 and 32 ensure that the following invariant is satisfied: For every
edge e′ = (u′ , v ′ ) ∈ Φ′ , we have α ∈ missχ (u′ ) and β ∈ missχ (v ′ ), i.e., the edge e′ is popular; indeed,
Lemma 3.4 implies that this invariant holds just before the for loop starts (see Line 21), and any
edge e′ ∈ Φ′ that may violate this invariant at a later stage, which may only occur due to flipping
an alternating path that ends at a node incident on e′ (in Line 31), is removed from Φ′ (in Line 32).
    Now, the lemma follows from the observation that each time we successfully extend the coloring
to one edge e in Φ′ , we remove at most one other edge e′ ̸= e from Φ′ (see Line 32), due to the
vertex-disjointness of the edges in U ⊇ Φ ⊇ Φ′ .

We are now ready to conclude the running time analysis of Algorithm 1 and establish the required
lower bound on the number of newly colored edges in U under χ by this algorithm.

Lemma 3.8. Algorithm 1 takes Õ(m/∆) time in expectation and extends χ to Ω(λ/∆) new edges.

Proof. We start with the runtime analysis:

• Line 1 can be implemented in Õ(1) time using the auxiliary data structure, and Lines 2 and 21
  take constant time.


                                                      12
• Next, we bound the running time of the while loop (Line 3). For any integer k ≥ 0, let
  T (k) denote the expected runtime of while loop if we start the loop under the condition that
  |Φ| + |C| = k. We are interested in T (0) and we know that T (λ/10∆) = O(1) by the termination
  condition of the loop. By Lemma 3.5 and Lemma 3.6, for any 0 < k < λ/10∆, we have,
                                                  1          1
                              T (k) ≤ Õ(m/λ) +     · T (k) + · T (k + 1),
                                                  2          2
  where we additionally used the monotonicity of T (·), as well as the fact that each while-loop
  of Algorithm 1 has expected runtime Õ(m/λ) regardless of previous iterations, according to
  Lemma 3.6. Thus, T (k) ≤ T (k + 1) + Õ(m/λ) and hence T (0) ≤ λ/10∆ · Õ(m/λ) = Õ(m/∆).
• Finally, since the total number of edges with colors α and β just before Line 22 is O(m/∆) (see
  Claim 3.3), the for loop can be implemented in Õ(m/∆) time deterministically in a straightfor-
  ward manner.

Thus, the total runtime is Õ(m/∆) in expectation.
    We now establish the bound on the number of newly colored edges. When the while loop
terminates, we have |Φ| + |C| ≥ λ/(10∆) (see Line 3), and all the edges in C are colored under χ
(see Lemma 3.4). Next, by Lemma 3.7, the for loop in Line 22 further extends the coloring χ to
a constant fraction of the edges in Φ, by only flipping {α, β}-alternating paths. Consequently, we
get at least Ω(λ/∆) newly colored edges in U under χ. This concludes the proof.

    We can now conclude the proof of Lemma 3.2. To achieve the algorithm in this lemma, we simply
run Algorithm 1 in parallel Θ(log n) times and use the coloring of whichever one finishes first (and
terminate the rest at that point). This ensures the high probability guarantee of Lemma 3.2 still
in Õ(m/∆) runtime. This concludes the entire proof.

3.3   Extension to General Graphs: Roadmap for the Rest of the Paper
In our Lemma 3.2, we crucially need the graph G to be bipartite while executing Lines 28 to 30
in Algorithm 1. Otherwise, if G contains odd cycles, then the maximal {α, β}-alternating path Pu⋆
starting from u can end at v. In that case, the color β will no longer be missing at v once we flip
the path Pv⋆ , and so we will not be able to extend the coloring χ to the edge (u, v) via χ(u, v) ← β.
We shall emphasize that this is not a minor technicality, but rather the key reason general graphs
are not necessarily ∆ edge colorable and rather require (∆ + 1) colors.
    The standard machinery to deal with this issue is the Vizing fan (see Section 4). However, if we
try to use Vizing fans inside the framework of Algorithm 1 in a naive manner, then we lose control
over one of the colors in the alternating path being flipped while extending the coloring to an edge,
leading to a weaker averaging argument and a running time of Õ(∆m) instead of Õ(m).
    To address this bottleneck, one of our key conceptual contributions is to focus on Vizing fans
with respect to an object called a separable collection of u-components (see Section 5). Using this
concept, in Section 6 we present our algorithmic framework in general graphs. Our main result (see
Theorem 6.1) relies upon two fundamental subroutines. The second subroutine (see Lemma 6.3)
generalizes Algorithm 1 presented in this section. In contrast, the first subroutine (see Lemma 6.2)
either efficiently extends the current coloring to a constant fraction of the uncolored edges, or
changes the colors of some edges in the input graph so as to create a situation whereby we can
invoke Lemma 6.3. We devote Sections 7 and 8 towards proving Lemmas 6.2 and 6.3, respectively.
For clarity of presentation, we defer the details on supporting data structures to Section 9, such

                                                  13
as the use of hash tables or binary search tree data structures for basic operations like picking a
missing color at a vertex. Finally, Section 10 shows how to optimize our algorithmic framework
further and achieve an O(m log n) time with high probability.


4      Preliminaries: Vizing Fans and Vizing Chains
We now define the notion of Vizing fans, which has been used extensively in the edge coloring
literature [Viz64, GNK+ 85, Sin19].
Definition 4.1 (Vizing fan). A Vizing fan is a sequence F = (u, α), (v1 , c1 ), . . . , (vk , ck ) where
u, v1 , . . . , vk are distinct vertices and c1 , . . . , ck are colors such that

     1. α ∈ missχ (u) and ci ∈ missχ (vi ) for all i ∈ [k].

     2. v1 , . . . , vk are distinct neighbours of u.

     3. χ(u, v1 ) = ⊥ and χ(u, vi ) = ci−1 for all i > 1.

     4. Either ck ∈ missχ (u) or ck ∈ {c1 , . . . , ck−1 }.

We say that the Vizing fan F = (u, α), (v1 , c1 ), . . . , (vk , ck ) is α-primed, has center u and leaves
v1 , . . . , vk . We refer to ci as the color of vi within F. A crucial property is that we can rotate
colors around the Vizing fan F by setting χ(u, v1 ) ← c1 , . . . , χ(u, vi−1 ) ← ci−1 , χ(u, vi ) ← ⊥ for
any i ∈ [k]. We say that F is a trivial Vizing fan if ck ∈ missχ (u). Note that, if F is trivial, we
can immediately extend the coloring χ to (u, v1 ) by rotating all the colors around F and setting
χ(u, vk ) ← ck .
   Algorithm 2 describes the standard procedure used to construct Vizing fans. As input, it takes
a vertex u and a color α ∈ missχ (u), and returns an α-primed Vizing fan with center u.
    Algorithm 2: Vizing-Fan(u, v, α)
 1 For each x ∈ V , let clr(x) ∈ missχ (x)
 2 k ← 1 and v1 ← v
 3 c1 ← clr(v1 )
 4 while ck ∈
            / {c1 , . . . , ck−1 } and ck ∈
                                          / missχ (u) do
 5    Let (u, vk+1 ) be the edge with color χ(u, vk+1 ) = ck
 6    ck+1 ← clr(vk+1 )
 7    k ←k+1
 8 return (u, α), (v1 , c1 ), . . . , (vk , ck )


The Algorithm Vizing: We now describe the algorithm Vizing that, given a Vizing fan F =
(u, α), (v1 , c1 ), . . . , (vk , ck ) as input, extends the coloring χ to the edge (u, v1 ) by building a Vizing
chain. Algorithm 3 gives a formal description of this procedure.
    Algorithm 3: Vizing(F)
 1 if F is trivial then
 2    χ(u, vi ) ← ci for all i ∈ [k]
 3    return
 4 Let P denote the maximal {α, ck }-alternating path starting at u
 5 Extend χ to (u, v1 ) by flipping the path P and rotating colors in F (details in Lemma 4.2)



                                                         14
Thus, running Vizing(F) extends the coloring χ to the uncolored edge (u, v1 ) by rotating colors
in the Vizing fan F and flipping the colors of the alternating path P . We sometimes refer to the
process of running Vizing(F) as activating the Vizing fan F.

Lemma 4.2. Algorithm 3 extends the coloring χ to the edge (u, v1 ) in time O(∆ + |P |).

Proof. To see that the path P is well-defined, note that α ∈ missχ (u) and ck ∈
                                                                              / missχ (u), so there
is an {α, ck }-alternating path starting at u.

Extending the coloring χ: By the definition of a Vizing fan, ck is the first repetition of some color
in {c1 , . . . , ck−1 }, so there is a unique index j ∈ [k − 1] such that cj = ck , and P has (u, vj+1 ) as
its first edge with color cj . We consider the cases where the path P does or does not have vj as an
endpoint. If P does not end at vj , then we can rotate the colors of the first j + 1 edges in the fan
by setting χ(u, v1 ) ← c1 , . . . , χ(u, vj ) ← cj and flip the colors of the alternating path P . If P does
end at vj , then we flip the colors of the alternating path P , rotate the colors of the fan by setting
χ(u, v1 ) ← χ(u, v2 ), . . . , χ(u, vk−1 ) ← χ(u, vk ), and set χ(u, vk ) ← ck . Note that, while rotating the
fan in this last step, we have χ(u, vj+1 ) = α ̸= ck .

Using standard data structures (see Section 9), we can implement this algorithm to run in time
proportional to the number of edges that change their colors, which is O(∆ + |P |).

Given a Vizing fan F, we denote the path P considered by Algorithm 3 by Vizing-Path(F). If the
Vizing fan F is trivial, then Vizing-Path(F) denotes an empty path ∅.


5        Basic Building Blocks
In this section, we introduce the notation of u-fans, u-edges and separable collections, which are
the definitions that work as the basic building blocks for our algorithms.

5.1        U-Fans and U-Edges
We begin by defining the notion of u-fans that was used by [GNK+ 85].3

Definition 5.1 (u-fan, [GNK+ 85]). A u-fan is a tuple f = (u, v, w, α, β) where u, v and w are
distinct vertices and α and β are distinct colors such that:

    1. (u, v) and (u, w) are uncolored edges.

    2. α ∈ missχ (u) and β ∈ missχ (v) ∩ missχ (w).

We say that u is the center of f and that v and w are the leaves of f . We also say that the u-fan
f is {α, β}-primed and define cf (u) := α, cf (v) := β and cf (w) := β (i.e. given a vertex x ∈ f ,
cf (x) is the available color that f ‘assigns’ to x).

Activating U-Fans: Let f be an {α, β}-primed u-fan with center u and leaves v and w. The
key property of u-fans is that at most one of the {α, β}-alternating paths starting at v or w ends
at u. Say that the {α, β}-alternating path starting at v does not end at u. Then, after flipping
    3
        The term ‘u-fan’ was originally introduced by [GNK+ 85] as an abbreviation for ‘uncolored fan’.




                                                            15
                                     u1                              u2




                     x1                               x                               x2


Figure 4: This picture shows two u-fans (u1 , x1 , x, ∗, β1 ) and (u2 , x2 , x, ∗, β2 ) sharing a common
vertex x. The separable condition requires that β1 ̸= β2 ; for instance β1 , β2 could be magenta and
cyan as shown here.

this {α, β}-alternating path, both u, v are missing color α. Thus, we can extend the coloring χ by
assigning χ(u, v) ← α. We refer to this as activating the u-fan f .
We also define the notion of a u-edge similarly to u-fans.
Definition 5.2 (u-edge). A u-edge is a tuple e = (u, v, α) where (u, v) is an uncolored edge and α
is a color such that α ∈ missχ (u).

We say that u is the center of e and that α is the active color of e. For notational convenience,
we also say that the u-edge e is α-primed and define ce (u) := α and ce (v) = ⊥.4
Collections of U-Components: While working with both u-fans and u-edges simultaneously,
we sometimes refer to some g that is either a u-fan or a u-edge as a u-component. Throughout this
paper, we often consider collections of u-components U. We only use the term ‘collection’ in this
context, so we abbreviate ‘collection of u-components’ by just ‘collection’. We will be particularly
interested in collections satisfying the following useful property, which we refer to as separability.
Definition 5.3 (Separable Collection). Let χ be a partial (∆ + 1)-edge coloring of G and U be a
collection of u-components (i.e. u-fans and u-edges). We say that the collection U is separable if
the following holds:

   1. All of the u-components in U are edge-disjoint.
   2. For each x ∈ V , the colors in the multi-set CU (x) := {cg (x) | g ∈ U, x ∈ g} are distinct.

    We remark that the second property of this definition is rather important because we need to
ensure that different u-components are not interfering with each other when they share common
vertices. Check Figure 4 for an illustration.
Damaged U-Components: Suppose we have a partial (∆ + 1)-edge coloring χ and a separable
collection U w.r.t. χ. Now, suppose that we modify the coloring χ. We say that a u-component
g ∈ U is damaged if it is no longer a u-component w.r.t. the new coloring χ.
    We note that this can only happen for one of the following two reasons: (1) one of the uncolored
edges in g is now colored, or (2) there is a vertex x ∈ g such the color cg (x) that g assigns to x is
no longer missing at x.
   The following lemma shows that flipping the colors of an alternating path cannot damage many
u-components in a separable collection U.
  4
    Whenever we refer to an “uncolored edge e”, we are referring to an edge e ∈ E such that χ(e) = ⊥, whereas a
‘u-edge e’ always refers to the object from Definition 5.2 and is denoted in bold.


                                                      16
Lemma 5.4. Let χ be a partial (∆ + 1)-edge coloring of a graph G, U a separable collection and
P a maximal alternating path in χ. Then flipping the colors of the alternating path P damages at
most 2 u-components in U (corresponding to the two endpoints of the path).

Proof. Let x and y be the endpoints of the path P . Since only the palettes of x and y change after
flipping the colors of P , only u-components g ∈ U that contain x and y may be damaged. Since
the palette of x changes by 1 and the collection U is separable, only one g ∈ U containing x may
be damaged by the color cg (x) no longer being available at x. The same holds for the vertex y.
Thus, at most 2 u-components in U are damaged by this operation.

5.2      Data Structures
In Section 9, we describe the data structures that we use to implement our algorithm. On top of
the standard data structures used to maintain the edge coloring χ, which allows us to implement
Algorithms 2 and 3 efficiently, we also use data structures that allow us to efficiently maintain a
separable collection U. More specifically, the data structures that we use to implement a separable
collection U support the following queries.

      • InsertU (g): The input to this query is a u-component g. In response, the data structure
        adds g to U if U ∪{g} is separable and outputs fail otherwise.

      • DeleteU (g): The input to this query is a u-component g. In response, the data structure
        removes g from U if g ∈ U and outputs fail otherwise.

      • Find-ComponentU (x, c): The input to this query is a vertex x ∈ V and a color c ∈ [∆ + 1].
        In response, the data structure returns the u-component g ∈ U with cg (x) = c if such a
        u-component exists and outputs fail otherwise.

      • Missing-ColorU (x): The input to this query is a vertex x ∈ V . In response, the data
        structure returns an arbitrary color from the set missχ (x) \ CU (x).5

The following claim shows that it is always possible to answer a Missing-Color query.

Claim 5.5. For each x ∈ V , the set missχ (x) \ CU (x) is non-empty.

Proof. Let d denote the number of uncolored edges incident on x. Since the collection U is separable,
we have that |CU (x)| ≤ d. Since |missχ (x)| ≥ d + 1, it follows that |CU (x)| < |missχ (x)| and so
missχ (x) \ CU (x) ̸= ∅.

Furthermore, the data structure supports the following initialization operation.

      • Initialize(G, χ): Given a graph G and an edge coloring χ of G, we can initialize the data
        structure with an empty separable collection U = ∅.

In Section 9, we show how to implement the initialization operation in O(m) time and each of
these queries in O(1) time with the appropriate data structures. These queries provide the
‘interface’ via which our algorithm will interact with the u-components.
  5
      Note that, since |CU (x)| < |missχ (x)|, such a color always exists.




                                                             17
A Remark on Randomization: In order to get good space complexity and running time si-
multaneously, we implement the standard data structures (used for Algorithms 2 and 3) and the
data structure supporting the preceding queries using hashmaps (see Section 9). The following
proposition describes the hashmaps used by our data structures.6

Proposition 5.6 ([DadH90]). There exists a dynamic dictionary that, given a parameter k, can
handle k insertions, deletions, and map operations, uses O(k) space and takes O(1) worst-case time
per operation with probability at least 1 − O(1/k 7 ).

This implementation gives us the guarantee that across the run of the entire algorithm, every
query made to one of these data structures takes O(1) time and each initialization operation takes
O(m) time, with high probability. Since the randomization used for the hashmaps is independent
of the randomization used in the rest of the algorithm, we implicitly condition on the high
probability event that, every operation performed using a hashmap runs in O(1) time
throughout the rest of the paper.7

5.3    Vizing Fans in Separable Collections
Within our algorithm, we only construct Vizing fans and Vizing chains in a setting where there
is some underlying separable collection U. To ensure that activating and rotating colors around
Vizing fans does not damage too many u-components, we choose the missing colors involved in
Vizing fan constructions so that they ‘avoid’ the colors assigned to the u-components in U.

Definition 5.7. Let U be a separable collection and F = (u, α), (v1 , c1 ), . . . , (vk , ck ) be a Vizing
fan. We say that the Vizing fan F is U-avoiding if ci ∈ missχ (vi ) \ CU (vi ) for each leaf vi ∈ F.

We say that a Vizing fan F is a Vizing fan of the u-edge e = (u, v, α) if F is α-primed, has center
u and its first leaf is v. The following lemma shows that we can always find a U-avoiding Vizing
fan for a u-edge.

Lemma 5.8. Given a u-edge e ∈ U, there exists a U-avoiding Vizing fan F of e. Furthermore, we
can compute such a Vizing fan in O(∆) time.

Proof. By Claim 5.5, we can always find a collection of colors {clr(x)}x∈V such that clr(x) ∈
missχ (x) \ CU (x) for each x ∈ V . If we construct a Vizing fan F by calling Vizing-Fan and using
such a collection of colors within the algorithm, then it follows that the Vizing fan F is U-avoiding.
By combining the standard implementation of Algorithm 2 with the queries described in Section 5.2
we can do this in O(∆) time.

The following lemma describes some crucial properties of U-avoiding Vizing fans.

Lemma 5.9. Let χ be a (∆ + 1)-edge coloring of a graph G and U be a separable collection. For
any u-edge e = (u, v, α) ∈ U with a U-avoiding Vizing fan F, we have the following:

   1. Rotating colors around F does not damage any u-component in U \{e}.
   6
     We could have even used the construction of [Kus22] to obtain exponentially small error probability in this case,
but since we do not need this stronger guarantee we stick with the simpler work of [DadH90].
   7
     Alternatively, one can replace these hashmaps with balanced search trees to make these data structures deter-
ministic, incurring O(log n) overhead in the running time of each operation.



                                                         18
    2. Calling Vizing(F) damages at most one u-component in U \{e}. Furthermore, we can identify
       the damaged u-component in O(1) time.

Proof. Let F = (u, α), (v1 , c1 ), . . . , (vk , ck ). Rotating colors around F will only remove the color ci
from the palette of a leaf vi appearing in F. Since ci ∈          / CU (vi ), these changes to the palettes will
not damage any u-component in U. However, rotating colors around F will color the edge (u, v),
damaging the u-edge e.
    Now, suppose we call Vizing(F). If the Vizing fan F is trivial, then the algorithm rotates all the
colors in F and sets χ(u, vk ) ← ck . Rotating the colors can only damage e ∈ U, and removing ck
from the palette of u can damage at most one other u-component in U since U is separable. Using
the queries from Section 5.2, we can check if such a u-component exist and return it in O(1) time.
If F is not trivial, then the algorithm flips the {α, ck }-alternating path P = Vizing-Path(F) and
rotates colors around F. By similar arguments, rotating colors around F can only damage e ∈ U.
Let x denote the endpoint of P that is not u. Flipping the path P might remove either the color
α or ck from the palette of x, which might damage at most one u-component in U \{e} since U is
separable. We can again identify this u-component in O(1) time using the queries from Section 5.2.
It also removes the color α from the palette of u, but this cannot damage any u-component other
than e (again, since U is separable).


6     The Main Algorithm
We are now ready to present our main technical result, which is a slightly weaker version of The-
orem 1.1, and focuses on achieving a near-linear time algorithm for (∆ + 1) edge coloring (instead
of the exact O(m log n) time in Theorem 1.1; see the remark after that theorem). We will then use
this theorem in Section 10 to conclude the proof of Theorem 1.1.

Theorem 6.1. There is a randomized algorithm that, given any simple undirected graph G = (V, E)
on n vertices and m edges with maximum degree ∆, finds a (∆ + 1)-edge coloring of G in Õ(m)
time with high probability.

    Our main algorithm consists of two main components. The first component is an algorithm
called Construct-U-Fans that takes a partial (∆ + 1)-edge coloring χ with λ uncolored edges and
either extends χ to Ω(λ) of these edges or modifies the coloring to construct a separable collection
of Ω(λ) u-fans. Lemma 6.2 summarizes the behavior of this algorithm.

Lemma 6.2. Given a graph G, a partial (∆ + 1)-edge coloring χ of G and a set of λ uncolored
edges U , the algorithm Construct-U-Fans does one of the following in O(m + ∆λ) time:

    1. Extends the coloring to Ω(λ) uncolored edges.

    2. Modifies χ to obtain a separable collection of Ω(λ) u-fans U.

    The second component is an algorithm called Color-U-Fans that takes a collection of λ u-fans
and extends the coloring to Ω(λ) of the edges in the u-fans. Lemma 6.3 summarizes the behavior
of this algorithm. The reader may find it helpful to keep in mind that the algorithm for Lemma 6.3
is a generalization of algorithm for Lemma 3.2 in Section 3.

Lemma 6.3. Given a graph G, a partial (∆ + 1)-edge coloring χ of G and a separable collection
of λ u-fans U, the algorithm Color-U-Fans extends χ to Ω(λ) edges in O(m log n) time w.h.p.

                                                      19
   In Sections 7 and 8, we prove Lemmas 6.2 and 6.3 respectively. Using these lemmas, we now
show how to efficiently extend an edge coloring χ to the remaining uncolored edges.

Lemma 6.4. Given a graph G and a partial (∆ + 1)-edge coloring χ of G with λ uncolored edges
U , we can extend χ to the remaining uncolored edges in time O((m + ∆λ) log2 n) w.h.p.

Proof. Let U denote the set of edges that are uncolored by χ. We can then apply Lemma 6.2
to either extend χ to a constant fraction of the edges in U or construct a separable collection of
Ω(λ) u-fans U in O(m + ∆λ) time. In the second case, we can then apply Lemma 6.3 to color
Ω(λ) of the edges contained in these u-fans in O(m log n) time w.h.p. Thus, we can extend χ to
a constant fraction of the uncolored edges in O(m log n + ∆λ) time w.h.p. After repeating this
process O(log λ) ≤ O(log n) many times, no edges remain uncolored. Thus, we can extend the
coloring χ to the entire graph in O((m + ∆λ) log2 n) time w.h.p.

We now use this lemma to prove Theorem 6.1.

Proof of Theorem 6.1. We prove this by applying Lemma 6.4 to the standard Euler partition frame-
work [GNK+ 85, Sin19]. Given a graph G, we partition it into two edge-disjoint subgraphs G1 and
G2 on the same vertex set such that ∆(Gi ) ≤ ⌈∆/2⌉ for each Gi , where ∆(Gi ) denotes the maximum
degree of Gi . We then recursively compute a (∆(Gi ) + 1)-edge coloring χi for each Gi . Combining
χ1 and χ2 , we obtain a (∆ + 3)-edge coloring χ of G. We then uncolor the two smallest color classes
in χ, which contain O(m/∆) edges, and apply Lemma 6.4 to recolor all of the uncolored edges in
χ using only ∆ + 1 colors in O(m log2 n) time w.h.p.
    To show that the total running time of the algorithm is O(m log3 n), first note that the depth
of the recursion tree is O(log ∆). Next, consider the ith level of the recursion tree, for an arbitrary
i = O(log ∆): we have 2i edge-disjoint subgraphs G1 , . . . , G2i such that ∆(Gj ) ≤ O(∆/2i ) and
P2i                                                                                2
   j=1 |E(Gj )| = m. Since the total running time at recursion level i is O(m log n) and the depth
of the recursion tree is O(log ∆), it follows that the total running time is O(m log3 n) w.h.p.


7    The Algorithm Construct-U-Fans: Proof of Lemma 6.2
As input, the algorithm Construct-U-Fans is given a graph G and a partial (∆ + 1)-edge coloring
χ of G with λ uncolored edges. It begins by taking the λ uncolored edges in χ and using them to
construct a separable collection U of λ u-edges in an obvious way, which we describe in Lemma 7.14.
It then uses two subroutines, Prune-Vizing-Fans and Reduce-U-Edges, to reduce the number of u-
edges in U, either by coloring them or modifying χ to turn them into u-fans. More specifically, for
each color α ∈ [∆ + 1], the algorithm considers the collection E α (U) of u-edges that are α-primed
and (1) calls the subroutine Prune-Vizing-Fans to ensure that the u-edges in E α (U) have vertex-
disjoint Vizing fans by either coloring the u-edges with overlapping Vizing fans or using them to
create u-fans, and (2) calls the subroutine Reduce-U-Edges which either extends the coloring to
the u-edges in E α (U) or uses them to create u-fans. Algorithm 4 gives the pseudocode for this
algorithm.




                                                  20
 Algorithm 4: Construct-U-Fans
 1 Construct a separable collection of λ u-edges U
 2 for each α ∈ [∆ + 1] do
 3    F ← Prune-Vizing-Fans(U, α)
 4    Reduce-U-Edges(U, α, F)
 5 return U


Organization of Section 7: We first describe and analyze the subroutines Prune-Vizing-Fans and
Reduce-U-Edges used by the algorithm Construct-U-Fans before proving Lemma 6.2 in Section 7.3.

7.1         The Subroutine Prune-Vizing-Fans
As input, the subroutine Prune-Vizing-Fans is given a graph G, a partial (∆ + 1)-edge coloring χ
of G, a color α and a separable collection U with λα α-primed u-edges. We let E(U) ⊆ U denote
the set of u-edges in U and E α (U) ⊆ E(U) denote the set of u-edges in U that are α-primed. The
subroutine then modifies the coloring χ and collection U to ensure that the Vizing fans of the
u-edges in E α (U) are vertex-disjoint.

Algorithm Description: The subroutine Prune-Vizing-Fans scans through the u-edges in E α (U)
and constructs U-avoiding Vizing fans at these u-edges (see Lemma 5.8), maintaining a vertex-
disjoint subset F of these Vizing fans throughout the process. After constructing a Vizing fan
F for a u-edge e ∈ E α (U), the subroutine checks if F intersects some other fan in F. If not, it
adds F to F. Otherwise, it uses these intersecting Vizing fans to either extend the coloring χ or
construct a u-fan, removing the corresponding Vizing fans and u-edges from F and U respectively.
See Figure 5 for an illustration.
     More formally, let e1 , . . . , eℓ denote the u-edges in E α (U) at the time when the subroutine is
called and let ei = (ui , vi , α). Initialize an empty set of Vizing fans F ← ∅. Then, for each i ∈ [ℓ],
the subroutine constructs a U-avoiding Vizing fan Fi of ei and checks if any of the vertices in Fi
(i.e. any of the leaves or the center of Fi ) are contained in any of the Vizing fans in F. Note
that, since U is separable and u1 , . . . , uℓ are the centers of α-primed u-edges, these vertices are all
distinct. If Fi is vertex-disjoint from the Vizing fans in F, we add Fi to F. Otherwise, consider
the following 3 cases.

         • If ui is a leaf in Fj for some Fj ∈ F: Then we rotate the fan Fj so that (ui , uj ) is uncolored,
           set χ(ui , uj ) ← α, remove Fj from F and remove ei and ej from E α (U).

If this is not the case, then some leaf of Fi appears in a Vizing fan in F. Let w be the first such
leaf in Fi .

         • If w = uj for some Fj ∈ F: Then we rotate the fan Fi so that (ui , w) is uncolored, set
           χ(ui , w) ← α, remove Fj from F and remove ei and ej from E α (U).

         • If w is a leaf in Fj ∈ F: Then we rotate the fans Fi and Fj so that (ui , w) and (uj , w) are
           uncolored, create a u-fan f = (w, ui , uj , β, α) for an arbitrary β ∈ missχ (w) \ CU (w), remove
           Fj from F, add f to U and remove ei and ej from E α (U).8

The subroutine then returns the set F of vertex-disjoint Vizing fans.
     8
         Note that, since the vertices u1 , . . . , uℓ are all distinct, these cases are exhaustive.


                                                                   21
                                 u1                           u2




                   v1                           w                            v2




                                 u1                           u2




                   v1                           w                            v2


Figure 5: In this picture, we look at the Vizing fan around an uncolored edge (u1 , v1 ), and it
intersects with the Vizing fan of another edge (u2 , v2 ) currently residing in F at vertex w. Then,
we can rotate both Vizing fans and pair these two uncolored edges as a u-fan.




                                                22
Analysis of Prune-Vizing-Fans

The following lemmas summarize the main properties of the subroutine Prune-Vizing-Fans.

Lemma 7.1. Prune-Vizing-Fans returns a set F of vertex-disjoint U-avoiding Vizing fans for all of
the u-edges in E α (U).

Proof. We can show by induction that, after the subroutine finishes scanning the u-edge ei , the set
F consists of vertex-disjoint U-avoiding Vizing fans for the u-edges in {e1 , . . . , ei } ∩ E α (U).9 This
is true trivially for i = 1. Now, suppose that this is true for i − 1. If the U-avoiding Vizing fan Fi
of ei does not intersect any of the Vizing fans in F, then we add Fi to F and are done.
    Thus, we assume this is not the case. Let x0 denote ui and x1 , . . . , xk denote the leaves of the
Vizing fan Fi (in order). Let xp be the first vertex in the sequence x0 , x1 , . . . , xk which appears in
some Vizing fan Fj ∈ F. It follows from our algorithm that we remove ei and ej from E α (U) while
only affecting the palettes of vertices in Fj and x0 , . . . , xp by rotating the fans Fi and Fj . Since all
of the Vizing fans in F were vertex-disjoint, changing the palettes of vertices in Fj cannot affect the
palettes of vertices in any of the Vizing fans in F \ {Fj }, and hence they are still valid. Similarly,
none of the vertices x0 , . . . , xp are contained in any of the Vizing fans in F \ {Fj }, so changing
their palettes does not affect the validity of any other Vizing fans in F. Thus, after removing
Fj from F, removing ei and ej from E α (U) and making the changes to χ, the set F consists of
vertex-disjoint Vizing fans of the u-edges in {e1 , . . . , ei } ∩ E α (U). To see why all of these Vizing
fans also remain U-avoiding, note that any u-fan f added to U is vertex-disjoint from these Vizing
fans by an analogous argument.

Lemma 7.2. The subroutine Prune-Vizing-Fans maintains the invariant that U is separable.

Proof. We can show by induction that, after the subroutine finishes scanning the u-edge ei , the
collection U is separable. This is true trivially for i = 1, since the subroutine does not change
χ or U while scanning this u-edge. Now, suppose that this is true for i − 1. It follows from the
inductive argument in Lemma 7.1 that the Vizing fans currently in F are all U-avoiding. If we do
not change χ or U while scanning ei , then we are done. Thus, we assume that this is not the case.
The subroutine can modify χ and U in the following ways:

      • Rotating colors around some Fj so that (uj , ui ) is uncolored, setting χ(uj , ui ) ← α and
        removing ei and ej from U.

      • Rotating colors around Fi so that (ui , uj ) is uncolored for some Fj ∈ F, setting χ(uj , ui ) ← α
        and removing ei and ej from U.

      • Rotating colors around Fi and some Fj ∈ F so that (ui , uj ) is uncolored, removing ei and ej
        from U and adding a u-fan f to U.

In the first case, it follows from Lemma 5.9 (since the Vizing fans are all U-avoiding) that rotating
colors around Fj does not damage any u-component in U \{ej }. Furthermore, since U is separable,
setting χ(uj , ui ) ← α does not damage any u-component in U \{ei , ej }. Since we remove ei and ej
from U, we ensure that U remains separable. The same argument extends to the second case.
   Finally, for the third case, it follows from Lemma 5.9 that rotating colors around Fi and Fj
does not damage an u-component in U \{ei , ej }, so after removing ei and ej from U we have that
  9
      Since we remove edges from E α (U) throughout this process, {e1 , . . . , ei−1 } might not be contained in E α (U).


                                                             23
U is separable. Since we have that α ∈ CU (ui ) ∩ CU (uj ) after removing ei and ej from U, we can
see that f is indeed a u-fan and that U ∪{f } is separable.10

Lemma 7.3. Each time Prune-Vizing-Fans modifies the coloring χ, it removes at most 2 u-edges
from E α (U) and either (1) extends the coloring χ to one more edge, or (2) adds a u-fan to U.

Proof. This follows immediately from the description of the subroutine.

The following lemma shows that this subroutine can be implemented efficiently.

Lemma 7.4. We can implement Prune-Vizing-Fans to run in O(∆λα ) time.

Proof. By Lemma 5.8, we can construct each of the U-avoiding Vizing fans Fi in O(∆) time. As
we scan through the u-edges e1 , . . . , eℓ , we can maintain the set of vertices S that are contained in
the Vizing fans in F, along with pointers to the corresponding Vizing fan. This allows us to either
verify that Fi is vertex-disjoint from the Vizing fans in F or to find the first vertex in Fi that is
contained in one of these fans in O(∆) time. We can then appropriately update the coloring χ and
the collection U in O(∆) time using the data structures outlined in Section 5.2. It follows that the
entire subroutine takes O(∆λα ) time.

7.2      The Subroutine Reduce-U-Edges
As input, the subroutine Reduce-U-Edges is given a graph G, a partial (∆ + 1)-edge coloring χ of
G, a color α, a separable collection U with λα α-primed u-edges and a set F of vertex-disjoint
U-avoiding Vizing fans for all of the u-edges in E α (U). Let mα denote the number of edges with
color α in the input coloring χ. For each u-edge e ∈ E α (U), let Fe denote its Vizing fan in F.
Similarly, for each F ∈ F, let eF denote its corresponding u-edge in E α (U). The subroutine then
either extends the coloring to the u-edges in E α (U) or uses them to construct u-fans.

Algorithm Description: The subroutine Reduce-U-Edges begins by activating all of the trivial
Vizing fans in F and then removing them from F. It then constructs Vizing chains at each of
the remaining Vizing fans in F and explores the alternating paths Vizing-Path(F) ‘in parallel’
(meaning, one edge at a time across all these paths). The subroutine continues this process until it
either (1) reaches the end of one of these paths, or (2) two of these paths intersect. In the first case,
it identifies the Vizing fan F ∈ F whose Vizing chain is fully explored and calls Vizing(F) before
removing F from F. In the second case, it identifies the Vizing fans F, F′ ∈ F corresponding to the
intersecting Vizing chains and either extends the coloring χ to one more uncolored edge or creates
a u-fan by shifting uncolored edges down these Vizing chains, before removing F and F′ from F.
It repeats this process until at least half of the Vizing fans have been removed from F.
   More formally, the subroutine begins by scanning through all of the trivial Vizing fans F ∈ F
and calling Vizing(F), removing the F from F and eF from U, and removing any other u-component
damaged by calling Vizing(F) from U (see Lemma 5.9). Let L ← 0, S ← ∅ and PF ← ∅ for all
F ∈ F. The subroutine then proceeds in rounds which consist of updating each of the paths PF
while maintaining the following invariant.

Invariant 7.5. After updating a path PF , the following hold:
 10
      Note that, before removing ei and ej from U, cei (ui ) = cej (uj ) = α.




                                                            24
  1. For each F ∈ F, PF is the length |PF | prefix of the alternating path Vizing-Path(F), which
     we denote by Vizing-Path(F)≤|PF | .

  2. The prefix paths {PF }F∈F are all edge-disjoint.

  3. The set S is the union of all the edges in the prefix paths {PF }F∈F .

  4. The collection U is separable and |F| = |Eα (U)|.

  5. The Vizing fans in F are U-avoiding and vertex-disjoint.

The prefix paths {PF }F∈F maintained by the algorithm also satisfy the following invariant.

Invariant 7.6. At the start of each round, we have that |PF | = L for all F ∈ F.

Immediately after starting a round, the subroutine increases the value of L by 1, invalidating
Invariant 7.6. To restore this invariant, we need to update each of the prefix paths PF , which in
turn may require us to update the other objects maintained by the subroutine to ensure that the
other conditions of Invariant 7.5 are satisfied. The subroutine does this by scanning through each
F ∈ F and calling Update-Path(F), which we describe formally in Algorithm 5. The subroutine
performs these rounds until |F| ≤ λα /2, at which point it terminates.

Analysis of Reduce-U-Edges

The following lemmas summarize the main properties of the subroutine Reduce-U-Edges.

Lemma 7.7. The subroutine Reduce-U-Edges satisfies Invariant 7.5.

Proof. We show that, as long as Invariant 7.5 is satisfied, calling Update-Path(F) for some F ∈ F
maintains the invariant. We first note that Invariant 7.5 is trivially satisfied immediately after
initializing L, S and {PF }F∈F .
     After activating and removing all of the trivial Vizing fans in F, Conditions 1-3 are clearly
still satisfied (since the paths are all empty). For Condition 4, note that the subroutine removes
any damaged u-components from U, removes e from E α (U) if it removes Fe from F, and that
activating some trivial Fe ∈ F cannot damage any e′ ∈ E α (U) \ {e} since their Vizing fans are
vertex-disjoint. For Condition 5, note that the subroutine does not add any u-component to U, so
the vertex-disjoint Vizing fans remain U-avoiding. This establishes the base case of the induction.
   Now, assume that Invariant 7.5 is satisfied and suppose that we call Update-Path(F) for some
F ∈ F. We now argue that each condition of Invariant 7.5 is satisfied after handling this call.

Conditions 2 and 3: After extending PF by one more edge, we check if there is some other path
PF′ that intersects the updated path PF at this new edge. If so, we remove both F and F′ from
F, ensuring that the remaining paths are edge-disjoint. Otherwise, the paths are all edge-disjoint.
It’s straightforward to verify that S is updated correctly in each case.

Condition 4: We first show that the collection U remains separable. If this call to Update-Path
does not change χ or U, then clearly U remains separable. Note that the Vizing fans in F are all
U-avoiding and recall Lemma 5.9. We now consider the following three cases.




                                                25
 Algorithm 5: Update-Path(F)
 1 PF ← Vizing-Path(F)≤|PF |+1
 2 Let (x, y) be the Lth edge in P
                                 F and y be an endpoint of PF
 3 if (x, y) ∈ S then
 4     Let F′ ∈ F be the Vizing fan such that (x, y) ∈ PF′
 5     if (x, y) appears in the same orientation in PF and PF′ then
 6         Let (z, x) and (z ′ , x) be the edges appearing before (x, y) in PF and PF′ respectively
 7         β ← χ(z, x)
 8         χ(z, x) ← ⊥ and χ(z ′ , x) ← ⊥
 9         Vizing(F)
10         Vizing(F′ )
11         Remove F and F′ from F and eF and eF′ from U
12         Add the u-fan f = (x, z, z ′ , β, α) to U
13     else
14         χ(x, y) ← ⊥
15         Vizing(F)
16         Vizing(F′ )
17         Remove F and F′ from F and eF and eF′ from U
18     S ← S \ (PF ∪ PF′ )
19     return
20 S ← S ∪ {(x, y)}
21 if PF is maximal then
22     Vizing(F)
23     Remove F from F and eF from U
24     S ← S \ PF
25     if there exists g ∈ U with cg (y) = χ(x, y) then
26         Remove g from U (see Lemma 5.9)
27     if there exists F′ ∈ F such that y ∈ F then
28         Remove F′ from F and eF′ from U // eF ′ might also get removed in Line 26
29         S ← S \ PF′




                                                 26
(1) If we enter the if statement on Line 5, then calling Vizing(F) and Vizing(F′ ) after uncoloring
    edges on their Vizing chains does not damage any u-component in U apart from eF and eF′ ,
    which are both removed from U. This is because shifting colors around the Vizing fans F and F′
    does not damage any u-components apart from eF and eF′ (see Lemma 5.9) and truncating an
    alternating path before flipping it ensures that we do not remove any colors from the palettes
    of the vertices on that path. Finally, note that the u-fan f that we add to U only uses colors
    that were previously unavailable. See the example Figure 6 for an illustration.

(2) If we enter the else statement on Line 13, then the argument is completely analogous to the
    previous case, except that we do not add any u-components to U (see Lemma 5.9). See Figure 7
    for an illustration.

(3) If we enter the if statement on Line 21, then calling Vizing(F) damages at most one u-component
    in U apart from eF (see Lemma 5.9). In particular, if there is such a u-component g, it will
    contain the vertex y and have cg (x) = χ(x, y). Thus, we check if such a u-component exists
    and remove it from U.

Finally, to see that |F| = |Eα (U)|, we can verify that we remove e from E α (U) if and only if we
remove Fe from F. It’s clear that whenever we remove some F′ from F, we also remove eF′ from
E α (U). Similarly, if we remove a u-edge e from Eα (U) in Line 26, then we can see that we then
remove Fe from F in Line 28 immediately afterwards.

Condition 5: Let F′ be some Vizing fan that remains in F after handling this call. Then, for any
Vizing fan F′′ that was activated during the call, we know that its corresponding Vizing chain did
not end at F′ , otherwise F′ would have been removed from F (see Line 29). Thus, activating F′′
(which is vertex-disjoint from F′ ) does not change the palettes of any vertices in F′ . Note that,
since the center of F′ is missing α, it is not possible for the Vizing chain of F′′ to contain any edges
in F′ without ending at F′ . Hence, the Vizing fans remaining in F are still valid Vizing fans and
remain vertex-disjoint. To see that they also remain U-avoiding, note that if we add a u-fan f to
U (see Line 12), the color that f assigns to a vertex x ∈ f was previously not available at x before
the start of the call to Update-Path. Thus, if x is contained in the Vizing fan F′ , its color within
F′ is not cf (x).

Condition 1: Let F′ be some Vizing fan that remains in F after handling this call. We now show
that the first |PF′ | edges of Vizing-Path(F′ ) do not change throughout this call. It follows from
the argument for Condition 5 that neither the colors of the edges nor the palettes of the vertices
in F′ change during this call. It remains to show that none of the edges in PF′ change color during
this call. We can see that PF′ and PF are edge-disjoint since otherwise F′ would have been removed
from F. Consequently, PF′ is edge-disjoint from any Vizing chain that is activated during the call,
and hence none of the edges in PF′ change colors.

Lemma 7.8. The subroutine Reduce-U-Edges satisfies Invariant 7.6.

Proof. This follows from the fact that initially |PF | = 0 for all F ∈ F and that the subroutine calls
Update-Path(F) for each F ∈ F during each round, which increases |PF | by 1.

Lemma 7.9. Each time Reduce-U-Edges modifies the coloring χ, it removes at most 2 u-edges
from E α (U) and either (1) extends the coloring χ to one more edge and removes at most one other
u-component from U, or (2) adds a u-fan to U.



                                                  27
                Vizing fan F around u1

                         u1




                                                                 z
           v1
                                    α                      α
                                                                            x           y
                                                                                  α
           v2
                                    α                      α
                                                                 z′

                                         A prefix of the Vizing chain PF′ starting at u2

                         u2




                         u1


                              α
                                                                 z
           v1
                                                α
                                                                            x           y
                                                                                  α
           v2
                                                α
                                                                 z′
                              α


                         u2

Figure 6: In this picture, α is blue, and two uncolored edges eF = (u1 , v1 ), eF′ = (u2 , v2 ) generate
Vizing fans F and F′ , and the two Vizing chains are joining at (x, y) for the first time and in the
same direction. Then, we can rotate both Vizing fans and partially flip PF and PF′ to make a u-fan




                                                    28
                               u1




                                                                                  x
                 v1
                                          α                      α
                                                                                      α

                 v2
                                          α                      α                y



                               u2




                               u1


                                    α
                                                                                  x
                 v1
                                                       α                     α


                 v2
                                                       α                     α    y
                                    α


                               u2

Figure 7: In this picture, α is blue, and two uncolored edges eF = (u1 , v1 ), eF′ = (u2 , v2 ) generate
Vizing fans F and F′ , and the two Vizing chains are joining at (x, y) for the first time and in the
opposite direction. Then, we can uncolor (x, y) and color both eF , eF′ .




                                                  29
Proof. While activating some trivial Vizing fan Fe ∈ F at the start, we remove e from U (and thus
also from E α (U)) along with at most one other u-component in U that is damaged by this operation
(see Lemma 5.9). Now, suppose that the subroutine calls Update-Path(F) for some F ∈ F. By
considering each case, we can verify that the subroutine either makes no changes to χ and U or it
removes at most 2 u-edges from E α (U) and either (1) extends the coloring χ to one more edge (see
Lines 12-14 and Line 22) and removes at most one other u-component from U (see Line 26), or (2)
adds a u-fan to U (see Line 12).

Running Time: We now show how to implement this subroutine and analyse its running time.
We begin with the following claim which shows that Update-Path can be implemented efficiently.

Claim 7.10. Each call to Update-Path(F) for some F ∈ F that does not remove F from F takes
O(1) time. Otherwise, it takes O(∆ + L) time.

Proof. By Invariants 7.5 and 7.6, we know that PF = Vizing-Path(F)≤L when the subroutine
calls Update-Path(F). Thus, updating PF to Vizing-Path(F)≤L+1 only requires computing the
next edge in the path, which can be done in O(1) time using our data structures (see Section 9).
    If this call does not remove F from F, then we know that PF is not maximal and is also edge-
disjoint from the other prefix paths maintained by the subroutine. In this case, we update S in
O(1) time and do not modify χ or U.
    On the other hand, if this call does remove F from F, then we need to activate O(1) many
Vizing chains of length O(L), which can be done in O(∆ + L) time. Removing the edges of the
corresponding paths from S can also be done in O(L) time. Finally, using the data structures
outlined in Section 5.2, we can update F and U in O(1) time.

Let F ⋆ denote the subset of Vizing fans that get removed from F by the subroutine Reduce-U-Edges
before it terminates. For each F ∈ F ⋆ , let LF denote the value of L at the time that F is removed
from F.

Claim 7.11. The total time spent handling calls to Update-Path is at most
                                                    X
                                  O(∆λα ) + O(1) ·      LF .                                    (1)
                                                     F∈F ⋆

Proof. Let F ∈ F ⋆ . We can observe that the subroutine calls Update-Path(F) at most LF times.
It follows from Claim 7.10 that the last call takes O(∆ + LF ) time while the rest take O(1) time.
Thus, the total time spent handling calls to Update-Path(F) is at most O(∆ + LF ). Summing over
each F ∈ F ⋆ , we get that the total time spent handling calls to Update-Path is at most
                            X                                     X
                                 O(∆ + LF ) ≤ O(∆λα ) + O(1) ·       LF .
                          F∈F ⋆                                F∈F ⋆


Recall that mα denotes the number of edges with color α when we first call the subroutine.

Claim 7.12. For each F ∈ F ⋆ , we have that LF ≤ O(mα /λα ).

Proof. Let Lmax denote the value of L at the start of the final round performed by the subroutine.
We now show that Lmax ≤ 8(mα + 4)/λα , which implies the claim. At the start of the final round,
we know that |F| > λα /2, otherwise the subroutine would terminate. Furthermore, it follows


                                                30
from Invariants 7.5 and 7.6 that, at the start of this round, the alternating paths {PF }F∈F form a
collection of at least λα /2 edge-disjoint {α, ·}-alternating paths of length Lmax in G. Let T denote
the total length of these paths. We can observe that T is at most 3m′α , where m′α is the number
of edges that currently have color α, since at least a third of the edges in each of these paths has
color α (note that any {α, ·}-alternating path of length k has at least ⌊k/2⌋ edges with color α).
We can also observe that m′α ≤ mα + 2λα since the number of edges with color α increases by at
most 2 each time we activate a Vizing fan in F. Thus, it follows that
                                λα
                                   · Lmax ≤ T ≤ 3m′α ≤ 3(mα + 2λα ),
                                 2
and so Lmax ≤ 6mα /λα + 12.

Lemma 7.13. We can implement Reduce-U-Edges to run in O(mα + ∆λα ) time.

Proof. Activating the trivial Vizing fans in F when the subroutine is first called can be done in
O(∆λα ) time. The running time of the rest of the subroutine is dominated by the time taken to
handle the call to Update-Path. Combining Claims 7.11 and 7.12, it follows that the total time
spent handling calls to Update-Path is at most
                                                                  
                              X                                 mα
            O(∆λα ) + O(1) ·        LF ≤ O(∆λα ) + O(λα ) · O        ≤ O(mα + ∆λα ).
                                  ⋆
                                                                λα
                               F∈F


7.3   Analysis of Construct-U-Fans: Proof of Lemma 6.2
The algorithm begins by constructing a separable collection of λ u-components U. The following
lemma shows that this can be done efficiently.
Lemma 7.14. Given a graph G and a partial (∆ + 1)-edge coloring χ of G with λ uncolored edges,
we can construct a separable collection of λ u-edges U in O(m) time.

Proof. The algorithm first initializes an empty separable collection U = ∅. It then scans through
the edges of the graph and retrieves the λ edges e1 , . . . , eλ that are uncolored by χ. The algorithm
then scans through each of these uncolored edges and, for each ei = (ui , vi ), picks a missing color
αi ∈ missχ (ui ) \ CU (ui ) (see Claim 5.5) and adds the u-edge ei := (ui , vi , αi ) to U. It’s easy to
verify that the resulting collection U is separable and contains λ u-edges. Using the data structures
outlined in Section 5.2, finding such a color and adding a u-edge to U can be done in O(1) time.
Thus, this entire process can be implemented in O(m) time.

Let mα denote the number of edges that have color α w.r.t. the initial coloring χ when we first call
Construct-U-Fans. The following claim shows that we cannot create too many more edges with the
color α throughout this sequence of calls to the subroutines Prune-Vizing-Fans and Reduce-U-Edges
for each color α ∈ [∆ + 1].
Claim 7.15. For each color α ∈ [∆ + 1], we have that at most mα + O(λ) edges have color α
throughout the entire run of the algorithm Construct-U-Fans.

Proof. This follows from the fact that the number of edges with color α can only increase when
we extend the coloring χ to some uncolored edge. Furthermore, it can only increase by at most 2
every time this happens. Thus, for any α ∈ [∆ + 1], the number of edges with color α can increase
by at most λ throughout the entire run of the algorithm Construct-U-Fans.

                                                  31
Let λα denote the number of α-primed u-edges in the initial collection U. Note that the number of
such u-edges can only decrease throughout the run of this algorithm. It follows from Lemmas 7.4
and 7.13 that the total running time of the algorithm Construct-U-Fans across all of these calls to
the subroutines Prune-Vizing-Fans and Reduce-U-Edges is at most
                        X
                             (O(λα ∆) + O(mα + λ + λα ∆)) ≤ O(m + ∆λ).
                      α∈[∆+1]

Lemma 7.16. The algorithm Construct-U-Fans either extends the coloring χ to at least λ/18 more
edges or returns a separable collection of at least λ/18 u-fans U.

Proof. To see why this lemma holds, consider the following three quantities and how they evolve
over time throughout the execution of the algorithm Construct-U-Fans: The number of edges that
the algorithm has extended the coloring to so far, Ψc , the number of u-edges in U, Ψe := |E|, and
the number of u-fans in U, Ψf := |U \ E|. Immediately after constructing the separable collection
of λ u-edges U, it holds that Ψc = 0, Ψe = λ and Ψf = 0. We next argue that by the time the
execution of the algorithm has finished, either Ψc ≥ λ/18 or Ψf ≥ λ/18 must hold. To this end,
we employ the following potential function argument.
Claim 7.17. The quantity 3(Ψf +2Ψc )+Ψe is non-decreasing throughout the algorithm’s execution.

Proof. Consider first a call made by algorithm Construct-U-Fans to the Prune-Vizing-Fans subrou-
tine. By Lemma 7.3, the subroutine repeatedly removes at most 2 u-edges from U and either (1)
adds a u-fan to U, or (2) extends the coloring to another edge. This may decrease Ψe by at most
2 while increasing one of Ψf or Ψc by at least 1, so 3(Ψf + 2Ψc ) + Ψe increases in this case.
    Next, consider a call to the Reduce-U-Edges subroutine. By Lemma 7.9, during any iteration,
the subroutine removes at most 3 u-edges from U and either (1) adds a u-fan to U, or (2) extends
the coloring to another edge while removing at most one u-fan from U. This may decrease Ψe by at
most 3 while increasing Ψf + 2Ψc by at least 1, so 3(Ψf + 2Ψc ) + Ψe cannot decrease in value.

Immediately after constructing the separable collection of λ u-edges in U, we have 3(Ψf + 2Ψc ) +
Ψe = λ. Claim 7.17 implies that 3(Ψf + 2Ψc ) + Ψe ≥ λ holds at all times afterwards. Since we
have that |E| ≤ λ/2 when the algorithm terminates, we know that Ψe ≤ λ/2 at this time, so at
that moment we have 3(Ψf + 2Ψc ) ≥ λ/2. Hence, either Ψf ≥ λ/18 or Ψc ≥ λ/18 must hold.


8    The Algorithm Color-U-Fans: Proof of Lemma 6.3
As input, the algorithm Color-U-Fans is given a graph G, a partial (∆ + 1)-edge coloring χ of G, and
a collection of separable u-fans U of size λ. It then uses two further subroutines, Prime-U-Fans and
Activate-U-Fans, to prime u-fans with the same colors and then activate them. More specifically,
the algorithm first identifies the two least common colors α, β ∈ [∆ + 1]. It calls the subroutine
Prime-U-Fans which proceeds to prime Ω(λ/∆) of the u-fans in U with the colors α and β (i.e. to
modify the coloring χ and these u-fans so that they are {α, β}-primed). It then calls Activate-U-Fans
which extends the coloring χ to Ω(λ/∆) of the edges in these u-fans. The algorithm repeats this
process until it has colored Ω(λ) edges. Algorithm 6 gives the pseudocode for Color-U-Fans.




                                                 32
 Algorithm 6: Color-U-Fans(U)
 1 for ∆/2 iterations do
 2     Let α, β ∈ [∆ + 1] be the two least common colors in χ
 3     Prime-U-Fans(U, α, β)
 4     Activate-U-Fans(U, α, β)

Organization of Section 8: We begin by describing and analyzing the subroutines Prime-U-Fans
and Activate-U-Fans used by Color-U-Fans before proving Lemma 6.3 in Section 8.3.

8.1   The Subroutine Prime-U-Fans
As input, this subroutine is given a graph G, a partial (∆ + 1)-edge coloring χ of G, a collection of
separable u-fans U of size λ, and the two least common colors α, β ∈ [∆+ 1] (see Claim 8.1). It then
proceeds in iterations, where in each iteration it samples a u-fan f from U uniformly at random
and attempts to prime f with the colors α and β. The subroutine maintains a subset Φ ⊆ U
of {α, β}-primed u-fans. The subroutine performs iterations until |Φ| = Ω(λ/∆), after which we
proceed to call the subroutine Activate-U-Fans to extend the coloring χ to edges in the u-fans in Φ.
The pseudocode in Algorithm 7 gives a formal description of the subroutine Prime-U-Fans.
 Algorithm 7: Prime-U-Fans(U)
 1 Φ ← ∅ and λ ← |U|
 2 while |Φ| < λ/(48∆) do
 3    Sample f = (u, v, w, γ, δ) ∼ U independently and u.a.r.
 4    (α′ , β ′ ) ← (α, β)       // α′ and β ′ ensure we never flip an {α, β}-alt. path
 5    if γ = β or δ = α then
 6         (α′ , β ′ ) ← (β, α)
 7    Let Pu be the {α′ , γ}-alternating path starting at u
 8    Let Pv and Pw be the {β ′ , δ}-alternating paths starting at v and w respectively
 9    cost(f ) := |Pu | + |Pv | + |Pw |
10    if cost(f ) ≤ 128m/λ then
11         Let S denote the set of endpoints of Pu , Pv and Pw
12         if the vertices in S ∪ {u, v, w} are not in any u-fan in Φ then
13               Flip the alternating paths Pu , Pv and Pw
14               Remove f and any damaged u-fans from U
15               Add the u-fan (u, v, w, α′ , β ′ ) to U and Φ
16 return (α, β)



Analysis of Prime-U-Fans

We say that an iteration of (the while loop in) Prime-U-Fans is successful if it adds a u-fan to
the set Φ. Otherwise, we say that the iteration fails (see Figure 8 for an illustration). Note that
the algorithm repeatedly performs iterations until it has performed at least λ/(48∆) successful
iterations. The following lemmas summarize the main properties of the subroutine Prime-U-Fans.

Claim 8.1. The total number of edges with color α or β is O(m/∆) at any point throughout the
run of Prime-U-Fans.



                                                 33
                                 u-fan f
                                             w     β          δ         β

       α          γ          α     u

                                             v     β          δ         β          δ    v′

                                                                                                   u′

                                                                                        w′


                                                                                             u-fan f ′ ∈ Φ



                                 u-fan f
                                             w     δ          β          δ
       γ          α          γ     u

                                             v     δ          β          δ         β    v′

                                                                                                   u′
                                                                  violating cv′ = β
                                                                                        w′


                                                                                             u-fan f ′ ∈ Φ

Figure 8: In this picture, α is red, β is orange, γ is blue, and δ is green. (u′ , v ′ , w′ ) ∈ Φ is an existing
{α, β}-primed u-fan in Φ. If we flip the {α, γ}-alternating path at u and the {β, δ}-alternating paths
at v and w, then we would destroy the property that β ∈ missχ (v ′ ).




                                                       34
Proof. Since α and β are initially the two least common colors, there are initially O(m/∆) such
edges. Since each successful iteration increases the number of such edges by O(1) and there are at
most O(λ/∆) ≤ O(m/∆) such iterations, the claim follows.

Claim 8.2. |U| ≥ (1 − 1/(2∆))λ at any point throughout the run of Prime-U-Fans.

Proof. Consider a successful iteration of the algorithm where we sample a u-fan f ∈ U. The
algorithm removes f from U, along with any other damaged u-fans in U. It follows from Lemma 5.4
that flipping the colors of the alternating paths Pu , Pv and Pw damages at most 6 u-fans in U.
Since our algorithm runs for at most λ/(48∆) iterations, it follows that |U| ≥ λ − 7 · λ/(48∆).

Lemma 8.3. The u-fans in Φ are all vertex-disjoint and {α, β}-primed at any point throughout
the run of Prime-U-Fans.

Proof. If the algorithm samples a u-fan f that shares a vertex x with a u-fan f ′ ∈ Φ, then the
iteration fails. Thus, the u-fans in Φ are all vertex-disjoint.
    Now, note that whenever we add a u-fan (u, v, w, α′ , β ′ ) to Φ, we do this immediately after
flipping the paths Pu , Pv , and Pw . Thus, we have that α′ ∈ missχ (u) and β ′ ∈ missχ (v) ∩ missχ (w),
so the u-fan is {α, β}-primed when we add it to Φ. Since we only flip the paths in Line 13 if their
endpoints do not touch any u-fans in Φ, this operation cannot change what colors are available at
vertices in u-fans in Φ, and hence cannot change whether or not any u-fan in Φ is {α, β}-primed.

The following standard claim bounds the total length of all maximal {c, ·}-alternating paths.

Claim 8.4. For any color c ∈ [∆ + 1], the total length of all maximal {c, ·}-alternating paths in χ
is at most 4m.

Proof. Let Pc denote the set of all such alternating paths. First note that the total length of all
alternating paths in Pc with length 1 is at most m. Now, let c′ ∈ [∆ + 1] \ {c} and let P be a
maximal {c, c′ }-alternating path with |P | ≥ 2. We can observe that at least a third of the edges in
P have color c′ , and that each edge with color c′ only appears in one path in Pc . Thus, we have
that                        X
                                 |P | ≤ m + 3 · |{e ∈ E | χ(e) ̸= c}| ≤ 4m.
                            P ∈Pc

Lemma 8.5. Each iteration of Prime-U-Fans succeeds with probability at least 1/4.

Proof. Consider the state of the algorithm at the start of some iteration. For each u-fan f ′ ∈ U,
let Px (f ′ ) denote the alternating path starting at x ∈ f ′ that is considered by the algorithm if it
samples f ′ ∈ U. We define the cost of the u-fan f ′ to be
                                                     X
                                       cost(f ′ ) :=    |Px (f ′ )|.
                                                   x∈f ′

Let U ⋆ ⊆ U denote the subset of u-fans f ′ ∈ U such that none of the alternating paths {Px (f ′ )}x∈f ′
have endpoints at some u-fan in Φ. We can see that the iteration is successful if and only if the
u-fan f ∈ U sampled during the iteration satisfies cost(f ) ≤ 128m/λ (see Line 10) and f ∈ U ⋆
(see Line 12). Thus, we now show that this happens with probability at least 1/4.
   We first begin with the following claim.



                                                  35
                   β                   β
                   β                   β             v′
                                                                             α        α
                   β                   β
                                                                u′           α        α
                   β                   β
                                                                             α        α
                   β                   β
                   β                   β             w′ u-fan f ′ ∈ Φ


Figure 9: In this picture, α is red, β is orange, and f ′ = (u′ , v ′ , w′ , α, β) ∈ Φ is a u-fan, and we
have drawn 9 different {α, ·}- or {β, ·}-alternating paths starting from f ′ .


Claim 8.6. Let P be an {α, ·}- or {β, ·}-alternating path in χ. Then there are at most 2 u-fans
f ′ ∈ U that, if sampled during the iteration, might cause the algorithm to flip P .

Proof. Suppose that the path P is a {c, c′ }-alternating path for some colors c ∈ {α, β} and c′ ∈  /
                               11
{α, β} with endpoints x and y. Since the collection of u-fans U is separable, we know that at
most one u-fan in U containing x (resp. y) has c′ assigned as the color missing at x (resp. y). Thus,
we must sample one of these 2 u-fans for the algorithm to flip the path P .

Let P denote the set of all maximal {α, ·}- and {β, ·}-alternating paths in χ. We can see that the
expected cost of the u-fan f sampled during the iteration is
                                            1 XX              2 X         32m
                         E[cost(f )] =         · |Px (f )| ≤     · |P | ≤     ,
                                           |U|               |U|           λ
                                                f ∈U x∈f                   P ∈P

where we are using the facts that (1) each path in P ∈ P appears at most twice while summing
over the paths |Px (f )| by Claim 8.6, (2) that |U| ≥ λ/2 by Claim 8.2, and (3) that that the total
length of all paths in P is 8m by Claim 8.4. Applying Markov’s inequality, it follows that
                                                        
                                                    128m     1
                                    Pr cost(f ) ≥          ≤ .
                                                      λ      4

    Since each u-fan in Φ has 3 vertices and at most 2∆ {α, ·}- or {β, ·}-alternating paths end
at each of these vertices, we know that there are at most 6∆|Φ| alternating paths that could be
considered during some iteration that end at a u-fan in Φ. By Claim 8.6, there are at most 12∆|Φ|
u-fans in U that could cause the algorithm to consider one of these paths (see Figure 9 for an
illustration). Thus, at least |U| − 12∆|Φ| of the u-fans in U are contained in U ⋆ . It follows that

                                                                        λ   |U|
                                     |U ⋆ | ≥ |U | − 12∆|Φ| ≥ |U | −      ≥     ,
                                                                        4    2
where we are using the facts that |Φ| ≤ λ/(48∆) and |U| ≥ λ/2 from Claim 8.2. Since f is sampled
uniformly at random from U, it follows that Pr[f ∈ U ⋆ ] ≥ 1/2. The lemma follows by applying a
union bound.

 11
      Note that the algorithm never flips {α, β}-alternating paths.


                                                           36
Lemma 8.7. Each iteration of Prime-U-Fans takes time O(m/λ).

Proof. Using standard data structures, each iteration can be implemented in time proportional to
the length of the alternating paths Pu , Pv and Pw considered by the algorithm during the iteration.
We can check if |Pu | + |Pv | + |Pw | ≤ 128m/λ in O(m/λ) time by traversing these paths and aborting
if we notice that their total length exceeds 128m/λ. If their total length is at most 128m/λ, then
we can flip these paths and update the collection U in O(m/λ) time.

Lemma 8.8. The subroutine Prime-U-Fans runs in time O(m log n/∆) with high probability.

Proof. Since each iteration of Prime-U-Fans succeeds with probability at least 1/4 by Lemma 8.5
and the subroutine performs iterations until it succeeds λ/(48∆) times, it follows that it performs
at most O(λ log n/∆) iterations with high probability. Since each iteration takes O(m/λ) time by
Lemma 8.7, it follows that the total running time is O(m log n/∆) with high probability.

8.2   The Subroutine Activate-U-Fans
As input, this subroutine is given a graph G, a partial (∆ + 1)-edge coloring χ of G, and a subset
Φ ⊆ U of µ vertex-disjoint {α, β}-primed u-fans such that at most O(m/∆) edges have color α or
β. The subroutine repeatedly picks any f ∈ Φ and proceeds to activate the u-fan. It repeats this
process until Φ = ∅. The pseudocode in Algorithm 8 gives a formal description of the subroutine.
 Algorithm 8: Activate-U-Fans(U, α, β)
 1 while Φ ̸= ∅ do
 2     Let f ∈ Φ
 3     Let P and P ′ be the {α, β}-alternating paths starting at the leaves of f
 4     Activate the u-fan f by flipping the path P or P ′
 5     Remove f and any damaged u-fans from U and Φ


Analysis of Activate-U-Fans

The following lemmas summarise the key properties of the subroutine Activate-U-Fans.

Lemma 8.9. The subroutine Activate-U-Fans extends the coloring χ to at least µ/2 more edges.

Proof. During each iteration of the while loop, we activate a u-fan f and extend the coloring to
an uncolored edge in f . By Lemma 5.4, this process damages at most 2 u-fans in U (including f ).
Thus, in each iteration |Φ| decreases by at most 2, and hence we extend the coloring to at least
µ/2 more edges across all the iterations performed by the subroutine.

Lemma 8.10. The subroutine Activate-U-Fans has a running time of O(m/∆).

Proof. Let P denote the collection of maximal {α, β}-alternating paths in the coloring χ when we
first call Activate-U-Fans. Since at most O(m/∆) edges have color α and β in this coloring, we have
that the total length of the paths in P is O(m/∆) since the paths in P are all vertex-disjoint.
    Throughout the run of Activate-U-Fans, we only modify χ by flipping the colors of {α, β}-
alternating paths (see Line 4). Thus, the structure of the paths in P does not change (but their
colors might be flipped). Furthermore, each time the subroutine flips an {α, β}-alternating path,


                                                37
it removes any u-fan from Φ that contains an endpoint of this alternating path (see Line 5). Thus,
each path in P is only flipped at most once.
    Using standard data structures, each iteration of the while loop can be implemented in time
proportional to the length of the alternating path that is flipped during the iterationP
                                                                                       (see Line 4). It
follows that the total running time of the subroutine across all iterations is at most P ∈P O(|P |) ≤
O(m/∆).

Claim 8.11. |U| decreases by at most 2µ throughout the run of Activate-U-Fans.

Proof. During each iteration, we flip an alternating path and remove the damaged u-fans from U
and Φ. It follows from Lemma 5.4 that each iteration removes at most 2 u-fans from these sets.
Since we perform at most µ iterations, the claim follows.

8.3         Analysis of Color-U-Fans: Proof of Lemma 6.3
Given a separable collection of λ u-fans U, the algorithm Color-U-Fans repeatedly calls Prime-U-Fans
and Activate-U-Fans with the set U as described in Algorithm 6. It repeats this process for ∆/2
iterations. It follows from Claims 8.2 and 8.11 that |U| decreases by at most a (1 − 1/∆) factor
during each iteration. Thus, by Bernoulli’s inequality, we get that
                                                             ∆/2
                                                         1               λ
                                           |U| ≥ λ · 1 −             ≥
                                                         ∆               2

throughout the entire run of the algorithm. In each iteration of Algorithm 6, we extend the coloring
to Ω(λ/∆) edges in O(m log n/∆) time w.h.p. Thus, in total, we extend the coloring to Ω(λ) edges
in O(m log n) time w.h.p.


9         Implementation and Data Structures
In this section, we describe the key data structures that we use to implement an edge coloring χ
and a separable collection U, allowing us to efficiently implement the operations performed by our
algorithms. We first describe the data structures and then show how they can be used to efficiently
implement the queries described in Section 5.2.

Implementing an Edge Coloring: Let G = (V, E) be a graph of maximum degree ∆ and let
C := [∆ + 1] ∪ {⊥}. We implement an edge coloring χ : E −→ C of G using the following:

         • The map ϕ : E −→ C where ϕ(e) := χ(e) for all e ∈ E.

         • The map ϕ′ : V × C −→ E where ϕ′u (c) := {e ∋ u | χ(e) = c}.

         • The set χ−1 (c) := {e ∈ E | χ(e) = c}, for all c ∈ C.

         • The set missχ (u) ∩ [degG (u) + 1], for all u ∈ V .12
    12
   We take this intersection with [degG (u) + 1] instead of maintaining missχ (u) directly to ensure that the space
complexity and initialization time of the data structures are Õ(m) and not Ω(∆n).




                                                         38
We implement all of the maps and sets using hashmaps, allowing us to perform membership queries,
insertions and deletions in O(1) time (see Proposition 5.6).13 The map ϕ′ allows us to check if a
color c ∈ [∆ + 1] is available at a vertex u ∈ V in O(1) time, and if it is not, to find the edge
e ∋ u with χ(e) = c. The sets {χ−1 (c)}c∈C allow us to easily return all edges with a specific
color (including ⊥). Furthermore, we can determine which color classes are the least common in
O(1) time.14 Each time an edge e changes color under χ, we can easily update all of these data
structures in O(1) time. Furthermore, given O(1) time query access to an edge coloring χ, we can
initialize these data structures in O(m) time. We note that χ is a proper edge coloring if and only
if |ϕ′u (c)| ≤ 1 for all u ∈ V, c ∈ [∆ + 1].
    Since the hashmap used to implement ϕ stores m elements, it follows that it can be implemented
with O(m) space. Similarly, the map ϕ′ stores 2m elements (if {e ∋ u | χ(e) = c} = ∅, then we
do not store anything for ϕ′u (c)) and thus can be implemented with O(m) space since each element
has size O(1) (recall that |ϕ′u (c)|P
                                    ≤ 1 since the coloring is proper). Since each set χ−1 (c) can be
stored in space O(|χ −1 (c)|) and c |χ−1 (c)| = m, these sets can be implemented in space O(m).
                 P
Similarly, since u |missχ (u) ∩ [degG (u) + 1]| = O(m), the sets missχ (u) ∩ [degG (u) + 1] can also
be implemented in O(m) space.

Implementing a Separable Collection: We implement a separable collection U in a similar
manner using the following:

       • The map ψ : V × [∆ + 1] −→ U where ψu (c) := {g ∈ U | u ∈ g, cg (u) = c}.

       • The set CU (u) := {cg (u) | g ∈ U, u ∈ g}, for all u ∈ V .

       • The set C U (u) := (missχ (u) ∩ [degG (u) + 1]) \ CU (u), for all u ∈ V .

We again implement all of the maps and sets using hashmaps, allowing us to access and change
entries in O(1) time. We note that, since U is separable, |ψu (c)| ≤ 1 for all u ∈ V, c ∈ [∆ + 1].
Thus, we can determine the size of U and also sample from U uniformly at random in O(1) time.15
Each time we remove a color c ∈ [∆ + 1] from the palette missχ (u) of a vertex u ∈ V , we can
update C U (u) in O(1) time and check ψu (c) in O(1) time to find any u-component that has been
damaged. Each time we add or remove a u-component from U, we can update all of these data
structures in O(1) time. Furthermore, we can initialize these data structures for an empty collection
in O(m) time by creating an empty map ψ, empty sets CU (u) for each u ∈ V and copying the sets
C U (u) = missχ (u) ∩ [degG (u) + 1] for each u ∈ V which are maintained by the data structures for
the edge coloring χ. Since U is separable, we can see that C U (u) ̸= ∅. Thus, whenever we want a
color from the set missχ (u) \ CU (u), it suffices to take an arbitrary color from C U (u).
    For each g ∈ U, we can see that g is contained at most 3 times in ψ. Thus, the total space
required to store the hashmap that implements ψ is O(| U |). Since U is separable, the u-components
in U are edge-disjoint, and thus |U| ≤ m. It follows that the map ψ can be stored with O(m) space.
For each u ∈ V , we can observe that |CU (u)| ≤ degG (u) since at most degG (u) many u-components
in U contain the vertex u, and |C U (u)| ≤ degG (u) + 1 since C U (u) ⊆ [degG (u) + 1]. Thus, the total
space required to store the sets {CU (u)}u∈V and {C U (u)}u∈V is O(m).
  13
     By using balanced search trees instead of hashmaps, we can make the data structures deterministic while increas-
ing the time taken to perform these operations to O(log n).
  14
     For example, we can then maintain a list of colors c ∈ C sorted by the values of |χ−1 (c)|.
  15
     For example, we can sample a number r ∼ [|U|] u.a.r. and then return the rth element in the hashmap that
implements ψ.



                                                         39
9.1    Implementing the Operations from Section 5.2
We now describe how to implement each of the operations from Section 5.2.

Implementing Initialize(G, χ): Suppose that we are given the graph G and O(1) time query
access to an edge coloring χ of G. We can initialize the data structures used to maintain the
maps ϕ and ϕ′ in O(m) time. We can then scan through the edges e ∈ E and initialize the sets
χ−1 (c) in O(m) time. Finally, we can scan through the vertices u ∈ V and initialize the sets
missχ (u) ∩ [degG (u) + 1] in O(m) time. We can then initialize the data structures for an empty
separable collection in O(m) time by creating an empty map ψ and, for each u ∈ V , initializing
the sets CU (u) ← ∅ and C U (u) ← missχ (u) ∩ [degG (u) + 1].

Implementing InsertU (g): By performing at most 3 queries to the map ψ, we can check if U ∪{g}
is separable. If so we can update ψ and the sets CU (x) and C U (x) for x ∈ g in O(1) time in order
to insert g into U. Otherwise, we return fail.

Implementing DeleteU (g): We can first make a query to ψ to ensure that g ∈ U. If so, we can
update ψ and the sets CU (x) and C U (x) for x ∈ g in O(1) time to remove g from U.

Implementing Find-ComponentU (x, c): We make a query to ψ by checking if there is an element
ψx (c). If no such element is contained in ψ, then return fail. Otherwise, return the unique u-
component in the set ψx (c).

Implementing Missing-ColorU (x): Return an arbitrary color from the set C U (x).


10      The Final Algorithm: Proof of Theorem 1.1
Up until this point in the presentation of our algorithm and its analysis, we made no attempt in
optimizing the logarithmic runtime factors, which led to an algorithm with O(m log3 n) time for
finding a (∆ + 1)-edge coloring with high probability in Theorem 6.1. We now show that we can
further optimize the algorithm and obtain an O(m log n) time algorithm and conclude the proof
of Theorem 1.1.
     Let us first start by listing where the log(n)-terms come from in the proof of Theorem 6.1:

  1. In Lemma 6.3, when coloring u-fans, we are losing an O(log n) factor to ensure the probabilistic
     guarantees of the algorithm hold with high probability in each step (see Lemma 8.8).

  2. Our application of Lemmas 6.2 and 6.3 can color a constant fraction of remaining uncolored
     edges, hence, we need to run them O(log n) times to color all uncolored edges (see Lemma 6.4).

  3. And finally, the entire framework of reducing (∆ + 1) coloring to extending the coloring to
     O(m/∆) uncolored edges using Eulerian partition technique (in the proof of Theorem 6.1)
     leads to a recursion depth of O(log n) leading to another O(log n) overhead in the runtime.

All in all, these factors led to the O(m log3 n) bound of our algorithm in Theorem 6.1.
    We now show how these log factors can be reduced to a single one. The two main ideas are: (1)
relaxing the requirement of Lemma 6.3 so that its runtime holds in expectation, plus a suitable tail
bound. Then, instead of maintaining high probability bound on each invocation of this lemma, we
only bound the runtime of all invocations of this lemma together with high probability; (2) relaxing


                                                 40
the Eulerian partition technique to color most of the graph recursively, instead of the entire graph,
and then taking care of the remaining uncolored edges at the end.
    We start by presenting a more fine-grained version of two of our main technical lemmas in the
proof of Theorem 6.1 (Lemma 6.3 and Lemma 6.4; recall that Lemma 6.2 already runs in linear
time deterministically), which correspond to part (1) above, and then use these to present part (2)
of above and conclude the proof.

10.1     Fine-Grained Variants of Lemma 6.3 and Lemma 6.4
We remove the O(log n)-term of Lemma 6.3 by focusing on the expected runtime of the algorithm
(plus a crucial tail inequality). In the next part, we show how to recover the final result even from
this weaker guarantee.

Lemma 10.1 (A slight modification of Lemma 6.3). There is an algorithm that, given a graph G,
a partial (∆ + 1)-edge coloring χ and a separable collection of λ u-fans, extends χ to Ω(λ) in T
(randomized) time such that for some T0 = O(m), we have,
                                                                                
                                                                             δ·λ
              E [T ] ≤ T0 and for all δ > 0 Pr [T ≥ (1 + δ) · 2T0 ] ≤ exp −        .
                                                                             100

Proof. The amortized runtime for coloring each single edge in Lemma 6.3 is O(m/λ) by Lemma 8.7
assuming the coloring succeeds, which happens with probability at least 1/4 by Lemma 8.5. We
emphasize that this is in an average sense: the algorithm Color-U-Fans first uses Prime-U-Fans to
prime Ω(λ/∆) u-fans in expected O(m/λ) time per u-fan and thus O((λ/∆) · (m/λ)) = O(m/∆)
expected total time, and then deterministically colors them in O(m/∆) time using Activate-U-Fans.
It then repeats this process ∆/2 times to color Ω(λ) edges, implying that E [T ] = O(m) time.
    We now prove the desired tail inequality on T as well. Let us number all iterations of the
while-loop in Prime-U-Fans from 1 to t, across all ∆/2 times Prime-U-Fans is called in Color-U-Fans:
for the ith iteration, define an indicator random variable Xi which is 1 iff this iteration of the
while-loop is successful
                     P in priming a u-fan (i.e., increasing the size of Φ). Additionally, for every
such i, define Si := ij=1 Xj which denotes the number of successful iterations after the algorithm
has done i iterations in total. Thus, t is the smallest integer where St = γ · λ for some integer
γ ∈ (0, 1) which is the fraction of u-fans the algorithm colors (precisely, γ = 1/48 · 1/2 ≥ 1/100).
   For any i ≥ 1, Si stochastically dominates the binomial distribution with parameters i and
p = 1/4 (by Lemma 8.5 each iteration is successful with probability at least 1/4). Hence, for δ > 0,
by concentration results for the binomial distribution,

                                                (2(1 + δ))2
                                                                                
                                                                               δ·λ
             Pr [t ≥ (1 + δ) · 8γ · λ] ≤ exp −              · 4γ · λ ≤ exp −         ,
                                               2 + 2(1 + δ)                    100

using a loose upper bound in the last step. Given that the runtime of the algorithm is t · O(m/λ),
the tail inequality follows.

    We note Lemma 10.1 guarantees that as long as λ = ω(log n), the O(m) runtime of the algorithm
also holds with high probability. For smaller values of λ (which happens only as a corner case in
the algorithm16 ), we need the specific tail inequality proven in the lemma instead.
 16
    The only case in our algorithm where this tail bound cannot be replaced with a high probability bound is when
m/∆ = o(log n) and at the same time ∆ log2 ∆ = ω(m log ∆), which means m = o(n log n) and yet ∆ = ω(n/ log n).


                                                       41
    Furthermore, we provide a similarly fine-grained version of Lemma 6.4 that will be used in the
last step of the argument to bypass the O(log n) factor loss of the original lemma.

Lemma 10.2 (A slight modification of Lemma 6.4). There is an algorithm that given a graph G,
a partial (∆ + 1)-edge coloring χ of G with λ uncolored edges, and an integer λ0 ≤ λ, extends χ to
all but λ0 uncolored edges in T (randomized) time such that for some T0 = O(m · log (λ/λ0 ) + ∆λ),
we have,                                                                           
                                                                             δ · λ0
              E [T ] ≤ T0 and for all δ > 0 Pr [T ≥ (1 + δ) · 2T0 ] ≤ exp −           .
                                                                              200

Proof. The algorithm is verbatim as in Lemma 6.4 by replacing Lemma 6.3 with Lemma 10.1:
repeatedly color a constant fraction of remaining edges as long as λ0 uncolored edges remain.
Since we start with λ uncolored edges and finish with λ0 many, and reduce uncolored edges by
a constant factor each time, we apply Lemma 6.2 and Lemma 10.1 for a total of O(log (λ/λ0 ))
times. Moreover, the ∆λ terms in the runtime of Lemma 6.2 form a geometric series and thus their
contribution to the runtime is O(∆λ) in total. This proves the expected bound on the runtime,
i.e., E [T ] ≤ T0 = O(m · log (λ/λ0 ) + ∆λ).
   As for the tail bound, in each application of Lemma 10.1 with an intermediate value λ′ ∈ [λ0 , λ],
and T0′ = O(m) being the T0 -parameter of Lemma 10.1, we have,

                                                                                    δ · λ′
                                                                                          
      Pr runtime of Lemma 10.1 with λ′ uncolored edges ≥ (1 + δ) · 2T0′ ≤ exp −
                                                                        
                                                                                             .
                                                                                     100

Thus, by union bound,

                                                                  δ · λ′
                                                        X               
                        Pr [total runtime ≥ k · T0 ] ≤     exp −
                                                         ′
                                                                   100
                                                       λ
                                                       ∞
                                                                   δ · λ0 · γ i
                                                       X                       
                                                     ≤     · exp −
                                                                       100
                                                        i=0
      (as number of uncolored edges drops by some constant factor, say, γ = 1 + Θ(1) each time)
                                                              
                                                        δ · λ0
                                               ≤ exp −           ,
                                                         200

by (loosely) upper bounding the sum of the geometric series using its first term.

    Before moving on from this subsection, we mention the following specialized concentration
inequality that we need in order to be able to exploit the tail bounds proven in Lemma 10.1
and Lemma 10.2. The proof follows a standard moment generating function argument and is
postponed to Appendix D.

Proposition 10.3. Let {Xi }ni=1 be n independent non-negative random variables associated with
parameters {αi }ni=1 and {βi }ni=1 such that for each i ∈ [n], αi , βi ≥ 1, and for every δ > 0,

                                Pr [Xi ≥ (1 + δ) · αi ] ≤ exp (−δ · βi ) ;

then, for every t ≥ 0,
                    " n      n
                                     #                                      n
                                                                                         !!
                     X       X                  n   βi                        X αi
                 Pr     Xi ≥   αi + t ≤ exp − min        ·              t−2                   .
                                               i=1 2αi                              βi
                    i=1        i=1                                            i=1


                                                   42
10.2    Proof of Theorem 1.1
We are now ready to use the more fine-grained versions of our main technical lemmas to (∆ + 1)
edge color the graph in O(m log n) randomized time, and conclude the proof of Theorem 1.1.
    The first part is based on a modification to the Eulerian partition approach used in the proof
of Theorem 6.1. Notice that the runtime of this algorithm is even O(m log ∆) and not O(m log n)
(the distinction at this point is irrelevant for us in proving Theorem 1.1, but we state the result
this way so we can use it later in Appendix C as well).
Lemma 10.4. There is an algorithm that given a graph G, finds a (∆ + 1) edge coloring of all but
(exactly) m/∆ edges in O(m log ∆) time with high probability.

Proof. Consider the following recursive algorithm. Find an Eulerian tour of G and partition G into
two edge-disjoint subgraphs G1 and G2 on the same vertex set such that ∆(G1 ), ∆(G2 ) ≤ ⌈∆/2⌉
and m(G1 ), m(G2 ) ≤ ⌈m/2⌉. For i ∈ {1, 2}, recursively, find a (∆(Gi ) + 1)-edge coloring χi of
Gi that leaves m(Gi )/∆(Gi ) = O(m/∆) edges uncolored. Combining χ1 and χ2 and uncoloring
the two smallest color classes, gives a (∆ + 1) coloring χ of all but λ := O(m/∆) edges. We then
run Lemma 10.2 to reduce the number of uncolored edges to λ0 := m/∆ edges. The correctness of
the algorithm thus follows immediately.
   For the runtime, we have k := O(∆) sub-problems in total, with 2d subproblems on m/2d
                                                                                                

edges and maximum degree ∆/2d for d ≤ ⌈log ∆⌉. Let X1 , . . . , Xk denote the runtime of these
subproblems and thus X := ki=1 Xi is the total runtime. We have,
                             P

                                      ⌈log ∆⌉                                        
                      X                X
                                                d          m                 ∆
            E [X] =       E [Xi ] =             2 ·O          · log (λ/λ0 ) + d · λ       = O(m log ∆)
                                                           2d                2
                      i                d=1

by Lemma 10.2, since λ = O(m/∆), λ0 = m/∆, and thus log (λ/λ0 ) = O(1) and ∆ · λ = O(m).
Moreover, by Lemma 10.2, for the sub-problem corresponding to Xi at some level d of the recursion,
T0d = O(m/2d ), and every δ ≥ 0,
                                                       
                    h
                                     d
                                       i            λ0                m 
                 Pr Xi ≥ (1 + δ) · 2T0 ≤ exp −δ ·         = exp −δ ·          .
                                                    200               200∆

For this Xi , define αi := 2T0d and βi := m/200∆. We can now apply Proposition 10.3 and for
                                  k                        ⌈log ∆⌉
                                  X                         X
                      t := 1000         αi = 1000 ·                  2d · 2T0d = O(m log ∆),
                                  i=1                       d=1

have,
                                                           n
                                                              !!
                                       n   βi             X  αi
                 Pr [X ≥ t] ≤ exp − min          · t−2
                                      i=1 2αi                β
                                                          i=1 i
                                                           
                                    Θ(1)            t · 200∆
                            = exp −      · t−2·
                                     ∆              1000 · m
                                                 
                                    Θ(1)
                            ≤ exp −      · (3t/5)                  (as ∆ ≤ m always trivially)
                                     ∆
                                                  
                                           m log ∆
                            = exp −Θ(1) ·             ,
                                              ∆

                                                            43
where we can make the constant in Θ(1) arbitrarily large without changing the asymptotic run-
                                                                                         √
time of the algorithm. This probability is always   at most 1/ poly(n) because either ∆ < m, in
                                              √            √
which case, the probability is at most 1/2Θ( m) , or ∆ > m, and thus log ∆ = Θ(log n) and the
probability is 1/ poly(n) at the very least. This concludes the proof.

    To conclude the proof, we have the following lemma that colors the remaining m/∆ edges left un-
colored by the algorithm of Lemma 10.4. The proof is a straightforward application of Lemma 10.2,
plus the original Vizing’s Fan and Chain approach stated in Lemma 4.2.
Lemma 10.5. There is an algorithm that, given a graph G, a partial (∆ + 1)-edge coloring χ with
λ = m/∆ uncolored edges, extends the coloring to all edges in O(m log n) time with high probability.

Proof. We first run our Lemma 10.2 with the given parameters λ = m/∆ and λ0 = 100 log n.
By Lemma 10.2, the expected runtime will be some T0 = O(m log (λ/λ0 ) + ∆λ) = O(m log n) since
λ ≤ m and λ0 ≥ 1. Moreover, by the same lemma,

    Pr [runtime of the algorithm ≥ 2000 · T0 ] ≤ exp (−10λ0 ) ≤ exp (−1000 log n) = 1/ poly(n),

since λ0 ≥ 100 log n. Thus, after running this part, in O(m log n) time, with high probability, we
will be left with only O(log n) uncolored edges. We can then color each remaining uncolored edge
in O(n) time using the original Vizing’s Fans and Chains approach of Lemma 4.2, thus obtaining
a (∆ + 1) coloring of the entire graph in O(m log n) time with high probability.

   Theorem 1.1 now follows immediately from Lemma 10.4 and Lemma 10.5.


Acknowledgements
Part of this work was conducted while Sepehr Assadi and Soheil Behnezhad were visiting the Simons
Institute for the Theory of Computing as part of the Sublinear Algorithms program.
    Sepehr Assadi is supported in part by a Sloan Research Fellowship, an NSERC Discovery Grant
(RGPIN-2024-04290), and a Faculty of Math Research Chair grant from University of Waterloo.
Soheil Behnezhad is funded by an NSF CAREER award CCF-2442812 and a Google Faculty Re-
search Award. Martı́n Costa is supported by a Google PhD Fellowship. Shay Solomon is funded by
the European Union (ERC, DynOpt, 101043159). Views and opinions expressed are however those
of the author(s) only and do not necessarily reflect those of the European Union or the European
Research Council. Neither the European Union nor the granting authority can be held responsible
for them. Shay Solomon is also funded by a grant from the United States-Israel Binational Science
Foundation (BSF), Jerusalem, Israel, and the United States National Science Foundation (NSF).
Tianyi Zhang is supported by funding from the starting grant “A New Paradigm for Flow and Cut
Algorithms” (no. TMSGI2 218022) of the Swiss National Science Foundation.


References
[Alo03]     Noga Alon. A simple algorithm for edge-coloring bipartite multigraphs. Information
            Processing Letters, 85(6):301–302, 2003.

[Arj82]     Eshrat Arjomandi. An efficient algorithm for colouring the edges of a graph with ∆ + 1
            colours. INFOR: Information Systems and Operational Research, 20(2):82–101, 1982.

                                                44
[Ass25]    Sepehr Assadi. Faster Vizing and Near-Vizing Edge Coloring Algorithms. In Annual
           ACM-SIAM Symposium on Discrete Algorithms (SODA), 2025.

[BBKO22] Alkida Balliu, Sebastian Brandt, Fabian Kuhn, and Dennis Olivetti. Distributed edge
         coloring in time polylogarithmic in ∆. In Proceedings of the 2022 ACM Symposium on
         Principles of Distributed Computing, pages 15–25, 2022.

[BCC+ 24] Sayan Bhattacharya, Din Carmon, Martı́n Costa, Shay Solomon, and Tianyi Zhang.
                                                      √
          Faster (∆ + 1)-Edge Coloring: Breaking the m n Time Barrier. In 65th IEEE Sym-
          posium on Foundations of Computer Science (FOCS), 2024.

[BCHN18] Sayan Bhattacharya, Deeparnab Chakrabarty, Monika Henzinger, and Danupon
         Nanongkai. Dynamic Algorithms for Graph Coloring. In Proceedings of the Twenty-
         Ninth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1–20.
         SIAM, 2018.

[BCPS24a] Sayan Bhattacharya, Martı́n Costa, Nadav Panski, and Shay Solomon. Arboricity-
          Dependent Algorithms for Edge Coloring. In 19th Scandinavian Symposium and Work-
          shops on Algorithm Theory (SWAT), volume 294 of LIPIcs, pages 12:1–12:15, 2024.

[BCPS24b] Sayan Bhattacharya, Martı́n Costa, Nadav Panski, and Shay Solomon. Density-
          Sensitive Algorithms for (∆ + 1)-Edge Coloring. In 32nd Annual European Symposium
          on Algorithms, ESA 2024, volume 308 of LIPIcs, pages 23:1–23:18, 2024.

[BCPS24c] Sayan Bhattacharya, Martı́n Costa, Nadav Panski, and Shay Solomon. Nibbling at
          Long Cycles: Dynamic (and Static) Edge Coloring in Optimal Time. In Proceedings of
          the ACM-SIAM Symposium on Discrete Algorithms (SODA). SIAM, 2024.

[BCSZ25]   Sayan Bhattacharya, Martı́n Costa, Shay Solomon, and Tianyi Zhang. Even Faster
           (∆ + 1)-Edge Coloring via Shorter Multi-Step Vizing Chains. In Annual ACM-SIAM
           Symposium on Discrete Algorithms (SODA), 2025.

[BD23]     Anton Bernshteyn and Abhishek Dhawan. Fast algorithms for vizing’s theorem on
           bounded degree graphs. CoRR, abs/2303.05408, 2023.

[BD24]     Anton Bernshteyn and Abhishek Dhawan. A linear-time algorithm for (1 + ϵ)∆-edge-
           coloring. arXiv preprint arXiv:2407.04887, 2024.

[BDH+ 19] Soheil Behnezhad, Mahsa Derakhshan, MohammadTaghi Hajiaghayi, Marina Knittel,
          and Hamed Saleh. Streaming and massively parallel algorithms for edge coloring. In
          27th Annual European Symposium on Algorithms (ESA), volume 144 of LIPIcs, pages
          15:1–15:14, 2019.

[Ber22]    Anton Bernshteyn. A fast distributed algorithm for (∆ + 1)-edge-coloring. J. Comb.
           Theory, Ser. B, 152:319–352, 2022.

[BGW21]    Sayan Bhattacharya, Fabrizio Grandoni, and David Wajc. Online edge coloring algo-
           rithms via the nibble method. In Proceedings of theACM-SIAM Symposium on Discrete
           Algorithms (SODA), pages 2830–2842. SIAM, 2021.

[BM17]     Leonid Barenboim and Tzalik Maimon. Fully-dynamic graph algorithms with sublinear
           time inspired by distributed computing. In International Conference on Computational
           Science (ICCS), volume 108 of Procedia Computer Science, pages 89–98. Elsevier, 2017.

                                              45
[BSVW24] Joakim Blikstad, Ola Svensson, Radu Vintan, and David Wajc. Online edge coloring
         is (nearly) as easy as offline. In Proceedings of the Annual ACM Symposium on Theory
         of Computing (STOC). ACM, 2024.

[BSVW25] Joakim Blikstad, Ola Svensson, Radu Vintan, and David Wajc. Deterministic Online
         Bipartite Edge Coloring. In Annual ACM-SIAM Symposium on Discrete Algorithms
         (SODA), 2025.

[CH82]     Richard Cole and John Hopcroft. On edge coloring bipartite graphs. SIAM Journal
           on Computing, 11(3):540–546, 1982.

[CHL+ 20] Yi-Jun Chang, Qizheng He, Wenzheng Li, Seth Pettie, and Jara Uitto. Distributed
          Edge Coloring and a Special Case of the Constructive Lovász Local Lemma. ACM
          Trans. Algorithms, 16(1):8:1–8:51, 2020.

[Chr23]    Aleksander Bjørn Grodt Christiansen. The Power of Multi-step Vizing Chains. In
           Proceedings of the 55th Annual ACM Symposium on Theory of Computing (STOC),
           pages 1013–1026. ACM, 2023.

[Chr24]    Aleksander B. G. Christiansen.     Deterministic dynamic edge-colouring.      CoRR,
           abs/2402.13139, 2024.

[CK08]     Richard Cole and Lukasz Kowalik. New linear-time algorithms for edge-coloring planar
           graphs. Algorithmica, 50(3):351–368, 2008.

[CMZ24]    Shiri Chechik, Doron Mukhtar, and Tianyi Zhang. Streaming edge coloring with sub-
           quadratic palette size. In 51st International Colloquium on Automata, Languages, and
           Programming, ICALP 2024, July 8-12, 2024, Tallinn, Estonia, volume 297 of LIPIcs,
           pages 40:1–40:12. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2024.

[CN90]     Marek Chrobak and Takao Nishizeki. Improved edge-coloring algorithms for planar
           graphs. Journal of Algorithms, 11(1):102–116, 1990.

[COS01]    Richard Cole, Kirstin Ost, and Stefan Schirra. Edge-Coloring Bipartite Multigraphs in
           O(E log D) Time. Comb., 21(1):5–12, 2001.

[CPW19]    Ilan Reuven Cohen, Binghui Peng, and David Wajc. Tight bounds for online edge col-
           oring. In 60th IEEE Annual Symposium on Foundations of Computer Science (FOCS),
           pages 1–25. IEEE Computer Society, 2019.

[CRV24]    Aleksander B. G. Christiansen, Eva Rotenberg, and Juliette Vlieghe. Sparsity-
           parameterised dynamic edge colouring. In 19th Scandinavian Symposium and Work-
           shops on Algorithm Theory (SWAT), volume 294 of LIPIcs, pages 20:1–20:18, 2024.

[CY89]     Marek Chrobak and Moti Yung. Fast algorithms for edge-coloring planar graphs. Jour-
           nal of Algorithms, 10(1):35–51, 1989.

[DadH90]   Martin Dietzfelbinger and Friedhelm Meyer auf der Heide. A new universal class of
           hash functions and dynamic hashing in real time. In Mike Paterson, editor, Automata,
           Languages and Programming, 17th International Colloquium, ICALP90, Warwick Uni-
           versity, England, UK, July 16-20, 1990, Proceedings, volume 443 of Lecture Notes in
           Computer Science, pages 6–19. Springer, 1990.


                                              46
[Dav23]    Peter Davies. Improved distributed algorithms for the lovász local lemma and edge col-
           oring. In Proceedings of the ACM-SIAM Symposium on Discrete Algorithms (SODA),
           pages 4273–4295. SIAM, 2023.

[DGS25]    Aditi Dudeja, Rashmika Goswami, and Michael Saks. Randomized Greedy Online Edge
           Coloring Succeeds for Dense and Randomly-Ordered Graphs. In Annual ACM-SIAM
           Symposium on Discrete Algorithms (SODA), 2025.

[Dha24a]   Abhishek Dhawan. Edge-coloring algorithms for bounded degree multigraphs. In
           David P. Woodruff, editor, Proceedings of the 2024 ACM-SIAM Symposium on Discrete
           Algorithms, SODA 2024, Alexandria, VA, USA, January 7-10, 2024, pages 2120–2157.
           SIAM, 2024.

[Dha24b]   Abhishek Dhawan. A simple algorithm for near-vizing edge-coloring in near-linear time.
           arXiv preprint arXiv:2407.16585, 2024.

[DHZ19]    Ran Duan, Haoqing He, and Tianyi Zhang. Dynamic edge coloring with improved ap-
           proximation. In 30th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA),
           2019.

[EK24]     Michael Elkin and Ariel Khuzman. Deterministic Simple (1 + ϵ)-Edge-Coloring in
           Near-Linear Time. arXiv preprint arXiv:2401.10538, 2024.

[EPS14]    Michael Elkin, Seth Pettie, and Hsin-Hao Su. (2∆ − 1)-Edge-Coloring is Much Easier
           than Maximal Matching in the Distributed Setting. In Proceedings of the Twenty-Sixth
           Annual ACM-SIAM Symposium on Discrete Algorithms, pages 355–370. SIAM, 2014.

[FGK17]    Manuela Fischer, Mohsen Ghaffari, and Fabian Kuhn. Deterministic distributed edge-
           coloring via hypergraph maximal matching. In 2017 IEEE 58th Annual Symposium on
           Foundations of Computer Science (FOCS), pages 180–191. IEEE, 2017.

[GKK10]    Ashish Goel, Michael Kapralov, and Sanjeev Khanna. Perfect matchings in o(n log n)
           time in regular bipartite graphs. In Proceedings of the Forty-second ACM Symposium
           on Theory of Computing, pages 39–46, 2010.

[GKMU18] Mohsen Ghaffari, Fabian Kuhn, Yannic Maus, and Jara Uitto. Deterministic dis-
         tributed edge-coloring with fewer colors. In Proceedings of the 50th Annual ACM
         SIGACT Symposium on Theory of Computing, pages 418–430, 2018.

[GNK+ 85] Harold N Gabow, Takao Nishizeki, Oded Kariv, Daneil Leven, and Osamu Terada.
          Algorithms for edge coloring. Technical Rport, 1985.

[GP20]     Jan Grebı́k and Oleg Pikhurko. Measurable versions of vizing’s theorem. Advances in
           Mathematics, 374:107378, 2020.

[GS24]     Prantar Ghosh and Manuel Stoeckl. Low-memory algorithms for online edge color-
           ing. In 51st International Colloquium on Automata, Languages, and Programming,
           ICALP 2024, July 8-12, 2024, Tallinn, Estonia, volume 297 of LIPIcs, pages 71:1–
           71:19. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2024.

[Hol81]    Ian Holyer. The np-completeness of edge-coloring.       SIAM Journal on computing,
           10(4):718–720, 1981.


                                               47
[KLS+ 22]   Janardhan Kulkarni, Yang P. Liu, Ashwin Sah, Mehtaab Sawhney, and Jakub Tar-
            nawski. Online edge coloring via tree recurrences and correlation decay. In 54th Annual
            ACM SIGACT Symposium on Theory of Computing (STOC), pages 104–116. ACM,
            2022.

[Kow24]     Lukasz Kowalik. Edge-Coloring Sparse Graphs with ∆ Colors in Quasilinear Time. In
            32nd Annual European Symposium on Algorithms, ESA 2024, volume 308 of LIPIcs,
            pages 81:1–81:17, 2024.

[KS87]      Howard J Karloff and David B Shmoys. Efficient parallel algorithms for edge coloring
            problems. Journal of Algorithms, 8(1):39–52, 1987.

[Kus22]     William Kuszmaul. A hash table without hash functions, and how to get the most out
            of your random bits. In 63rd IEEE Annual Symposium on Foundations of Computer
            Science, FOCS 2022, Denver, CO, USA, October 31 - November 3, 2022, pages 991–
            1001. IEEE, 2022.

[PR01]      Alessandro Panconesi and Romeo Rizzi. Some simple distributed algorithms for sparse
            networks. Distributed computing, 14(2):97–100, 2001.

[SB24]      Mohammad Saneian and Soheil Behnezhad. Streaming edge coloring with asymptot-
            ically optimal colors. In 51st International Colloquium on Automata, Languages, and
            Programming, ICALP 2024, July 8-12, 2024, Tallinn, Estonia, volume 297 of LIPIcs,
            pages 121:1–121:20. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2024.

[Sha49]     Claude E Shannon. A theorem on coloring the lines of a network. Journal of Mathe-
            matics and Physics, 28(1-4):148–152, 1949.

[Sin19]     Corwin Sinnamon. Fast and simple edge-coloring algorithms.             arXiv preprint
            arXiv:1907.03201, 2019.

[SW21]      Amin Saberi and David Wajc. The greedy algorithm is not optimal for on-line edge
            coloring. In 48th International Colloquium on Automata, Languages, and Programming
            (ICALP), volume 198 of LIPIcs, pages 109:1–109:18, 2021.

[Viz64]     V. G. Vizing. On an estimate of the chromatic class of a p-graph. Discret Analiz,
            3:25–30, 1964.

[Viz65]     Vadim G Vizing. The chromatic class of a multigraph. Cybernetics, 1(3):32–41, 1965.




                                                48
A      Vizing’s Theorem for Multigraphs in Near-Linear Time
We now show how to extend our arguments to multigraphs. A generalization of Vizing’s theorem
to multigraphs shows that any multigraph G with maximum degree ∆ and maximum multiplicity
µ can be edge colored with ∆ + µ colors [Viz64] (and this is worst-case optimal for µ ≤ ∆/2 in the
sense that not every multigraph admits an edge coloring with fewer colors). We prove the following
theorem, which shows how to compute such a coloring in near-linear time.
Theorem A.1. There is a randomized algorithm that, given an undirected multigraph G = (V, E)
on n vertices and m edges with maximum degree ∆ and maximum multiplicity µ, finds a (∆+µ)-edge
coloring of G in O(m log n) time with high probability.

   Similar to our results for (∆ + 1) edge coloring, we first focus on only obtaining an Õ(m)
time algorithm, and then use almost exactly the same argument we used to extend Theorem 6.1
to Theorem 1.1, to optimize this runtime to O(m log n) time.

Notation for Multigraphs: We refer to edges in G with the same endpoints as parallel. Whenever
we refer to an edge e of the multigraph G, we are referring to a specific edge and say that this edge
is distinct from its parallel edges.

A.1      Vizing Fans and Separable Collection in Multigraphs
Let G = (V, E) be an undirected multigraph on n vertices and m edges with maximum degree ∆
and maximum multiplicity µ, and χ be a partial (∆ + µ)-edge coloring of G. We now describe how
to generalize the objects used by our algorithm for simple graphs to deal with multigraphs. We
begin by defining a generalization of Vizing fans for multigraphs [Viz64]. For convenience, we still
refer to them as Vizing fans.
Definition A.2 (Vizing fan for multigraphs). A Vizing fan for a multigraph is a sequence F =
(u, α), (v1 , e1 , C1 ), . . . , (vk , ek , Ck ) where u, v1 , . . . , vk are distinct vertices, C1 , . . . , Ck ⊆ [∆ + µ] are
subsets of colors and e1 , . . . , ek are edges such that

    1. α ∈ missχ (u) and Ci ⊆ missχ (vi ) of size µ for all i ∈ [k].

    2. v1 , . . . , vk are distinct neighbours of u and ei is an edge with endpoints u and vi for all i ∈ [k].

    3. χ(e1 ) = ⊥ and, for all i > 1, there exists p(i) ∈ [i − 1] such that χ(ei ) ∈ Cp(i) .

    4. The sets C1 , . . . , Ck−1 , missχ (u) are mutually disjoint.

    5. Either Ck ∩ missχ (u) ̸= ∅ or Ck ∩ (C1 ∪ · · · ∪ Ck−1 ) ̸= ∅.

We say that the Vizing fan F = (u, α), (v1 , e1 , C1 ), . . . , (vk , ek , Ck ) is α-primed, has center u and
leaves v1 , . . . , vk . We refer to Ci as the colors of vi within F. A crucial property is that we can
rotate colors around the Vizing fan F: given any i ∈ [k], there is a sequence 1 = i1 < · · · < iℓ = i
(where iℓ−j = p(j) (i)) such that we can set χ(ei1 ) ← χ(ei2 ), . . . , χ(eiℓ−1 ) ← χ(eiℓ ), χ(eiℓ ) ← ⊥. We
say that F is a trivial Vizing fan if Ck ∩ missχ (u) ̸= ∅. Note that, if F is trivial, we can immediately
extend the coloring χ to (e1 ) by rotating colors around F to leave (e1 ) uncolored and setting χ(ek )
to be any color in Ck ∩ missχ (u).
   For completeness, we prove the following lemma, which shows how to efficiently construct such
a Vizing fan.

                                                             49
Lemma A.3 ([Viz64]). Given an uncolored edge e = (u, v) and a color α ∈ missχ (u), we can
construct an α-primed Vizing fan F with center u in O(∆) time.

Proof. Let v1 = v, e1 = e and C1 ⊆ missχ (v) be a subset of size µ. If C1 ∩ missχ (u) ̸= ∅,
then we can return (u, α), (v1 , e1 , C1 ) as a Vizing fan. Otherwise, we assume inductively that
we have constructed a sequence (u, α), (v1 , e1 , C1 ), . . . , (vi , ei , Ci ) satisfying Conditions 1 to 4 of
Definition A.2. In this case, we have that |C1 ∪ · · · ∪ Ci | = iµ. Since there are at most iµ − 1 many
colored edges that have u as one endpoint and one of v1 , . . . , vi as the other, it follows that there
must be some edge ei+1 = (u, vi+1 ) such that χ(ei+1 ) ∈ C1 ∪ · · · ∪ Ci and vi+1 ∈          / {v1 , . . . , vi }. Let
Ci+1 ⊆ missχ (vi+1 ) be a subset of size µ. If either Ci+1 ∩ missχ (u) ̸= ∅ or Ci+1 ∩ (C1 ∪ · · · ∪ Ci ) ̸=
∅, then (u, α), (v1 , e1 , C1 ), . . . , (vi+1 , ei+1 , Ci+1 ) is a Vizing fan and we are done. Otherwise, we
continue the induction on this longer sequence. Since this process must terminate within at most
∆/µ steps, this process always returns a Vizing fan. Furthermore, using standard data structures,
we can implement this process in O(∆) time.

Vizing Chains in Multigraphs: Let F = (u, α), (v1 , e1 , C1 ), . . . , (vk , ek , Ck ) be a non-trivial
α-primed Vizing fan, let c be a color in Ck ∩ (C1 , . . . , Ck−1 ) and let P denote the maximal {α, c}-
alternating path starting at u. Then, by a similar argument to the case of simple graphs, we
can extend the coloring χ to the edge e1 by flipping the path P and rotating colors around F.
Furthermore, this can be done in time O(∆ + |P |). We denote a call to the algorithm that extends
the coloring in the manner by Vizing(F).

Separable Collections in Multigraphs: We define a separable collection of u-components U
in a multigraph in the exact same way as simple graphs. We emphasise that, given a separable
collection U, any distinct u-components g 1 , g 2 ∈ U are edge-disjoint but may contain parallel edges.
In other words, it is possible for two distinct parallel uncolored edges e1 and e2 to be contained in
u-components within the separable collection U.
       We define U-avoiding Vizing fans for multigraphs in the following analogous way.

Definition A.4. Let U be a separable collection and F = (u, α), (v1 , e1 , C1 ), . . . , (vk , ek , Ck ) be a
Vizing fan. We say that the Vizing fan F is U-avoiding if Ci ⊆ missχ (vi ) \ CU (vi ) for each vi ∈ F.

    Using these new definitions, we can directly extend Lemmas 5.8 and 5.9 to multigraphs by
slightly changing the proofs, giving us the following lemmas.

Lemma A.5. Given a u-edge e ∈ U, there exists a U-avoiding Vizing fan F of e. Furthermore,
we can compute such a Vizing fan in O(∆) time.17

Lemma A.6. Let χ be a (∆ + µ)-edge coloring of a graph G and U be a separable collection. For
any u-edge e = (u, v, α) ∈ U with a U-avoiding Vizing fan F, we have the following:

   1. Rotating colors around F does not damage any u-component in U \{e}.

   2. Calling Vizing(F) damages at most one u-component in U \{e}. Furthermore, we can identify
      the damaged u-component in O(1) time.
  17
       Recall that we say that F is a Vizing fan of e = (u, v, α) if F is α-primed, has center u, and its first leaf is v.




                                                              50
A.2    Proof of Theorem A.1
Our main algorithm for multigraphs is completely analogous to our main algorithm for simple
graphs, which we describe in Section 6. In particular, the algorithm also consists of two main
components that are combined in the same way, which we summarize in the following lemmas.
Lemma A.7 (A modification of Lemma 6.2 for multigraphs). There is an algorithm that, given a
multigraph G, a partial (∆ + µ)-edge coloring χ of G and a set of λ uncolored edges U , does one of
the following in O(m + ∆λ) time:

  1. Extends the coloring to Ω(λ) uncolored edges.

  2. Modifies χ to obtain a separable collection of Ω(λ) u-fans U.

The proof of this lemma is similar to the proof of Lemma 6.2 but with some minor changes. In
Appendix A.3, we sketch how to modify the proof of Lemma 6.2 to extend it to Lemma A.7.
Lemma A.8 (A modification of Lemma 6.3 for multigraphs). There is an algorithm that, given
a multigraph G, a partial (∆ + µ)-edge coloring χ of G and a separable collection of λ u-fans U,
extends χ to Ω(λ) edges in O(m log n) time w.h.p.

The proof of this lemma is almost identical to the proof of Lemma 6.3, thus we omit the details.
    Using Lemmas A.7 and A.8, we get the following lemma which allows us to extend an edge
coloring χ to a set of uncolored edges.
Lemma A.9 (A modification of Lemma 6.4 for multigraphs). There is an algorithm that, given a
multigraph G and a partial (∆ + µ)-edge coloring χ of G with λ uncolored edges U , extends χ to
the remaining uncolored edges in time O((m + ∆λ) log2 n) w.h.p.

The proof of this lemma is verbatim the same as the proof of Lemma 6.4, and hence is omitted.
    Finally, in the same way as in the proof of Theorem 1.1, we can apply standard Euler partitioning
using Lemma A.9 to merge the colorings. We only note that in multigraphs, splitting an m-edge
multigraph G via the Euler partition leads to two multigraphs, each with ⌈m/2⌉ edges, maximum
degree ⌈∆/2⌉, and maximum multiplicity ⌈µ/2⌉. Thus, the number of colors used to color the
graph recursively will be 2 · (⌈∆/2⌉ + ⌈µ/2⌉) = ∆ + µ + O(1) colors. Hence, we again only need to
uncolor O(m/∆) edges to obtain a (∆ + µ) coloring and then extend this coloring to the remaining
O(m/∆) edges using Lemma A.9. This leads to an O(m log3 n) time algorithm for finding a (∆+µ)
edge coloring with high probability.
   We can also optimize this algorithm to run in O(m log n) time with high probability exactly as in
our (∆ + 1) edge coloring algorithm. Specifically, using exactly the same argument as in Section 10,
we have the following analogues of Lemma 10.4 and Lemma 10.5 for multigraphs.
Lemma A.10 (A modification of Lemma 10.4 for multigraphs). There is an algorithm that given
a multigraph G, finds a (∆ + µ) edge coloring of all but (exactly) m/∆ edges in O(m log ∆) time
with high probability.
Lemma A.11 (A modification of Lemma 10.5 for multigraphs). There is an algorithm that, given
a multigraph G, a partial (∆ + µ)-edge coloring χ with λ = m/∆ uncolored edges, extends the
coloring to all edges in O(m log n) time with high probability.

   Theorem A.1 now follows immediately from Lemma A.10 and Lemma A.11.

                                                 51
A.3       Proof Sketch of Lemma A.7
The algorithm for Lemma A.7 is almost identical to the algorithm for Lemma 6.2, except that we
replace the relevant definitions with their generalizations for multigraphs, leading to a few extra
cases. The overall structure of the algorithm is still the same as Algorithm 4, but using modifications
of Prune-Vizing-Fans and Reduce-U-Edges for multigraphs.

Initializing the Seperable Collection U: Given a set of λ uncolored edges U , we can construct
a separable collection of λ u-edges U in the exact same way as simple graphs. However, there may
exist distinct u-edges e, e′ ∈ U that correspond to parallel (but distinct) uncolored edges.

Modifying Prune-Vizing-Fans for Multigraphs: Suppose that we want to run Prune-Vizing-Fans
on a separable collection U with color α. In this call to the subroutine, we consider all of the
α-primed u-edges in E α (U) ⊆ U. We implement this subroutine for multigraphs in the same way
as for simple graphs, but with the following additional preprocessing step to ensure that none of the
u-edges in E α (U) are parallel.
   Suppose that we have distinct e, e′ ∈ E α (U) that correspond to parallel uncolored edges with
endpoints u and v. Then, by the definition of a separable collection, one of these u-edges must
have u as a center and the other must have v as a center. Thus, α ∈ missχ (u) ∩ missχ (v), so we
can extend the coloring to one of these u-edges by coloring it with α and then remove them both
from U. We can scan through all of the α-primed u-edges and perform this operation whenever
we encounter parallel u-edges. This maintains the invariant that U is a separable collection and
ensures that none of the u-edges in E α (U) are parallel.
   After performing this preprocessing step, we can implement the rest of Prune-Vizing-Fans in the
exact same way as described in Section 7.1, except that we must use the generalization of Vizing
fans for multigraphs. In particular, all of the lemmas in Section 7.1 describing the properties of
Prune-Vizing-Fans still hold.

Modifying Reduce-U-Edges for Multigraphs: Suppose that we want to run Reduce-U-Edges on
a separable collection U with color α and have obtained a collection of vertex-disjoint U-avoiding
Vizing fans F corresponding to the u-edges in E α (U) from calling Prune-Vizing-Fans. We implement
the subroutine Reduce-U-Edges in the same way as described in Section 7.2 but with the following
modification to Algorithm 5 to account for an additional case.
    Suppose that we call Update-Path(F) and take one more step along the path Vizing-Path(F)
and observe the edge e = (x, y). If the edge e has already been seen (i.e. is contained in S) then
we proceed as normal.18 If neither the edge e nor any edge parallel to e is contained in S, then
we again proceed as normal. However, if we observe that e is not already contained in S but some
edge e′ that is parallel to e is contained in S, then we do the following: Let F′ ∈ F be the Vizing
fan such that e′ ∈ PF′ , set χ(e) ← ⊥ and χ(e′ ) ← ⊥, call Vizing(F) and Vizing(F′ ), set χ(e) ← α,
remove F and F′ from F and eF and eF′ from U, and set S ← S \ (PF ∪ PF′ ); see Figure 10 for
an illustration. We can verify that, if this does happen, the edges e and e′ must appear in different
orientations in the paths PF and PF′ . Consequently, this operation removes 2 u-edges from U and
extends the coloring to one more edge.
   After modifying Algorithm 5 with this additional case, we strengthen Invariant 7.5 so that no
two edges contained in distinct prefix paths in {PF }F∈F are parallel. This is sufficient to ensure
that the analysis of Reduce-U-Edges in Section 7.2 extends to multigraphs. In particular, all of the
lemmas in Section 7.2 describing the properties of Reduce-U-Edges still hold.
 18
      That is, this exact copy of the edge is in S, not just an edge parallel to e.


                                                            52
                        Vizing fan F around u1

                                  u1




                   v1
                                               α                       α
                                                                           e    e′
                                               α                       α
                   v2




                                  u2

                        Vizing fan F′ around u2




                                  u1


                                       α

                   v1
                                                           α
                                                                           e α e′
                                                           α
                   v2

                                       α


                                  u2

Figure 10: If PF and PF′ meet in the middle at a pair of parallel edges e, e′ , then we can shift the
uncolored edge from F′ to e′ and apply Vizing(F ) which assigns χ(e) ← α; here α is blue.




                                                   53
B      Shannon’s Theorem for Multigraphs in Near-Linear Time
A classical theorem by Shannon shows that any multigraph G with maximum degree ∆ can be edge
colored with ⌊3∆/2⌋ colors [Sha49]. Unlike Vizing’s theorem for multigraphs, this bound does not
depend on the maximum multiplicity of the graph. We show that we can also extend our approach
to a near-linear time randomized algorithm for computing such a coloring. Previously, only the
original O(mn) time algorithm of [Sha49] and a recent O(n · ∆18 ) time algorithm of [Dha24a] were
known for this problem.
Theorem B.1. There is a randomized algorithm that, given an undirected multigraph graph G =
(V, E) on n vertices and m edges with maximum degree ∆, finds a ⌊3∆/2⌋-edge coloring of G in
O(m log n) time with high probability.

   Yet again, we first focus on obtaining a near-linear time algorithm, and postpone optimizing
the runtime to O(m log n) to the end of the proof.

B.1     Shannon Fans
Let G = (V, E) be an undirected multigraph on n vertices and m edges with maximum degree ∆
and χ be a partial ⌊3∆/2⌋-edge coloring of G. We begin by defining the notion of a Shannon fan,
which is conceptually very similar to a u-fan.
Definition B.2 (Shannon fan). A Shannon fan is a tuple S = (u, v, w, α, β, γ) where u, v and w
are distinct vertices and α, β and γ are distinct colors such that:

    1. (u, v) is an uncolored edge and (u, w) is an edge with color γ.
    2. α ∈ missχ (u), β ∈ missχ (v) ∩ missχ (w) and γ ∈ missχ (v).

We say that u is the center of S and that v and w are the leaves of S. We also say that the
Shannon fan S is {α, β}-primed, and that γ is the auxiliary color. We say that S is the Shannon
fan of the uncolored edge (u, v).
Activating Shannon Fans: Let S be an {α, β}-primed Shannon fan with center u and leaves
v and w. The key property of Shannon fans is that at most one of the {α, β}-alternating paths
starting at v or w ends at u. Suppose that the {α, β}-alternating path starting at v does not end
at u. Then, after flipping this {α, β}-alternating path, both u, v are missing color α. Thus, we can
extend the coloring χ by assigning χ(u, v) ← α. If the path does not end a w instead, we can set
χ(u, w) ← ⊥ and χ(u, v) ← γ and carry out an analogous argument. We refer to this as activating
the Shannon fan S.
Constructing Shannon Fans: The following lemma shows how to construct a Shannon fan.
Lemma B.3. Given an uncolored edge e, we can either construct a Shannon fan S of the edge e
or extend χ to e in O(∆) time.

Proof. Let u and v be the endpoints of e and let γ be any color in missχ (v). If γ ∈ missχ (u), then
we can set χ(u, v) ← γ and we are done. Otherwise, let (u, w) be the edge incident on u with color
γ (note that w ̸= v since γ ∈ missχ (v)). Now, suppose that missχ (u) is disjoint from both missχ (v)
and missχ (w). Otherwise, we can extend χ to e by shifting the color γ between (u, v) and (u, w)
and directly coloring the uncolored edge. The following claim shows that we can find a color in
missχ (v) ∩ missχ (w).

                                                   54
Claim B.4. We have that missχ (v) ∩ missχ (w) ̸= ∅.

Proof. Suppose that missχ (v) and missχ (v) are disjoint. Then we have that
                                                                                  
           3∆                    3∆                                                     3∆
     2·         − (∆ − 1) +           − ∆ ≥ |missχ (u)| + |missχ (v)| + |missχ (w)| ≥        ,
            2                     2                                                      2

which implies that ⌊3∆/2⌋ ≥ (3∆/2) − 1, giving a contradiction.

Now, we can find colors α ∈ missχ (u) and β ∈ missχ (v) ∩ missχ (w) in O(∆) time. It follows that
S = (u, v, w, α, β, γ) is a Shannon fan of the edge e.

Pre-Shannon Fans: For notational convenience, we introduce the notion of a pre-Shannon fan.

Definition B.5 (pre-Shannon fan). A pre-Shannon fan is a tuple S ⋆ = (u, v, w, γ) where u, v and
w are distinct vertices and γ is the auxiliary color such that (u, v) is an uncolored edge, (u, w) is
an edge with color γ and γ ∈ missχ (v).

By slightly modifying the proof of Lemma B.3, we get the following lemma which shows that we can
either convert a pre-Shannon fan S ⋆ into a Shannon fan S or extend the coloring to the uncolored
edge in S ⋆ .

Lemma B.6. Given a pre-Shannon fan S ⋆ = (u, v, w, γ), we can do one of the following in O(∆)
time:

  1. Find colors α and β such that S = (u, v, w, α, β, γ) is a Shannon fan.

  2. Extend the coloring to (u, v) while only changing the color of (u, w).

B.2    Proof of Theorem B.1
Our final algorithm for Shannon’s theorem is completely analogous to our final algorithm for simple
graphs, which we describe in Section 6. In particular, the algorithm also consists of two main
components that are combined in the same way, which we summarize in the following lemmas.

Lemma B.7 (A modification of Lemma 6.2 for Shannon fans). There is an algorithm that, given
a multigraph G, a partial ⌊3∆/2⌋-edge coloring χ and λ = O(m/∆) uncolored edges that form a
matching, does one of the following in O(m) time deterministically:

  1. Extends the coloring to Ω(λ) uncolored edges.

  2. Returns a set of Ω(λ) vertex-disjoint Shannon fans of the uncolored edges that all share the
     same auxiliary color.

We defer the proof of this lemma to Appendix B.3.

Lemma B.8 (A modification of Lemma 6.3 for Shannon fans). There is an algorithm that, given a
multigraph G, a partial ⌊3∆/2⌋-edge coloring χ and set of λ vertex-disjoint Shannon fans U sharing
the same auxiliary color, extends χ to Ω(λ) edges in O(m log n) time with high probability.



                                                 55
The proof of this lemma is almost identical to the proof of Lemma 6.3, thus we omit the details. In
particular, the set of vertex-disjoint Shannon fans U in this lemma has the same key properties as
the separable collection of u-fans in Lemma 6.3, so the proof extends directly by simply replacing
the u-fans with Shannon fans. We remark that the additional condition that the Shannon fans
must share the same auxiliary color γ is crucial. In particular, it ensures that we can prime these
Shannon fans by flipping alternating paths which will not contain the color γ, and thus will not
interfere with the auxiliary colors of the other Shannon fans in U.
    Using Lemmas B.7 and B.8, we get the following lemma which allows us to extend an edge
coloring χ to a matching of uncolored edges.
Lemma B.9 (A modification of Lemma 6.4 for Shannon fans). There is an algorithm that, given
a multigraph G, a partial ⌊3∆/2⌋-edge coloring χ and a set of λ = O(m/∆) uncolored edges that
form a matching, extends χ to these uncolored edges in O(m log2 n) time with high probability.

The proof of this lemma is almost identical to the proof of Lemma 6.4, with the observation that
repeatedly applying Lemmas B.7 and B.8 does not change the fact that the remaining uncolored
edges form a matching.19
    Finally, in the same way as in the proof of Theorem A.1, we can apply standard Euler parti-
tioning using Lemma B.9 to merge the colorings, applying it separately for each color class that we
uncolor. This allows for obtaining an O(m log3 n) time algorithm.
   The extension to an O(m log n) time algorithm is verbatim as in Theorem A.1 and Theorem 1.1
and we only mention the relevant lemmas below without proving them.
Lemma B.10 (A modification of Lemma 10.4 for Shannon’s Theorem). There is an algorithm that,
given a multigraph G, finds a ⌊3∆/2⌋-edge coloring of all but (exactly) m/∆ edges in O(m log ∆)
time with high probability.
Lemma B.11 (A modification of Lemma 10.5 for Shannon’s Theorem). There is an algorithm that,
given a multigraph G, a partial ⌊3∆/2⌋-edge coloring χ with λ = m/∆ uncolored edges, extends the
coloring to all edges in O(m log n) time with high probability.

       This concludes the proof of Theorem B.1.

B.3       Proof of Lemma B.7
Let U be the set of λ = O(m/∆) uncolored edges forming a matching. We first scan through the
edges (u, v) ∈ U and check if missχ (u) ∩ missχ (v) ̸= ∅ in O(∆) · λ = O(m) time. If this holds for at
least λ/2 of these edges, we can extend the coloring to these edges in O(m) time and we are done.
Otherwise, let U ′ ⊆ U be the subset of uncolored edges where this does not hold.
       Now, for each color c, we compute
                                                   X
                                    freq(c) :=               1[c ∈ missχ (v)]
                                                 (u,v)∈U ′

by scanning over each (u, v) ∈ U ′ , taking O(λ∆) time in total. We now define an auxiliary color

                                           γ := arg max freq(c)
                                                       c
  19
    Note that this is not true when we repeatedly apply Lemmas 6.2 and 6.3 in our algorithm for simple graphs,
since Lemma 6.2 changes the locations of the uncolored edges.


                                                      56
and proceed to construct a vertex-disjoint collection of pre-Shannon fans U ′ with auxiliary color γ
as described in Algorithm 9.
  Algorithm 9: Construct-Pre-Shannon-Fans(U ′ )
       ′
 1 U       ← ∅ and S ← ∅
 2 if (u, v) ∈ U ′ then
 3    if γ ∈ missχ (v) then
 4       Let (u, w) be the edge with χ(u, w) = γ
 5       if {u, v, w} ∩ S = ∅ then
 6            U ′ ← U ′ ∪{(u, v, w, γ)}
 7            S ← S ∪ {u, v, w}
            ′
 8 return U

We can observe that the set U ′ produced by Algorithm 9 consists of vertex-disjoint pre-Shannon
fans with auxiliary color γ, and that this can be computed in O(λ) time. The following lemma
shows that U ′ contains Ω(λ) Shannon fans.

Lemma B.12. We have that |U ′ | ≥ λ/24.

Proof. We begin by proving the following claim, which shows that γ is available at the endpoints
of Ω(λ) many edges in U ′ .
Claim B.13. We have that freq(γ) ≥ λ/12.

Proof. We can first observe that
             X               X X                                   X X
                  freq(c) =      1[c ∈ missχ (v)] =                              1[c ∈ missχ (v)]
                  c                 c (u,v)∈U ′                  (u,v)∈U ′   c

                            X  3∆     
                                            ∆−1
                                                
                                                             ∆            λ∆
                      ≥               −∆ ≥        · |U ′ | ≥   · |U ′ | ≥    ,
                              ′
                                  2          2               4             8
                          (u,v)∈U

where we are using the facts that each vertex has at least ⌊3∆/2⌋ − ∆ missing colors, that ∆ > 1
(otherwise, the problem is trivial), and that |U ′ | ≥ λ/2. Thus, we have that

                                            1    X            2 λ∆  λ
                            freq(γ) ≥          ·   freq(c) ≥    ·  = .
                                         ⌊3∆/2⌋ c            3∆ 8   12

    It follows from this claim that we can create pre-Shannon fans with γ as an auxiliary color with
at least λ/12 of the uncolored edges in U ′ . Since the uncolored edges in U ′ are vertex-disjoint, two
pre-Shannon fans with auxiliary color γ on these edges, (ui , vi , wi , γ) and (uj , vj , wj , γ), intersect if
and only if ui = wj and wi = uj . Thus, any such fan can intersect at most one other such fan. It
follows that the algorithm produces a collection U ′ of size at least λ/24.

   Finally, we can take the set U ′ of vertex-disjoint pre-Shannon fans and apply Lemma B.6 to
each S ⋆ ∈ U ′ to compute a set U of at least |U ′ |/2 = Ω(λ) vertex-disjoint Shannon fans or extend
χ to at least |U ′ |/2 = Ω(λ) edges in |U ′ | · O(∆) ≤ λ · O(∆) ≤ O(m) time.




                                                      57
C      Edge Coloring Algorithms for Small Values of ∆
The main contribution and the primary message of our work is presenting the first near-linear time
algorithm for (∆ + 1)-edge coloring and polynomial time factor improvements over prior work.
However, it turns out that a simple corollary of our main results, plus the prior work of [BD23],
is also enough to obtain quite efficient algorithms in the regime when ∆ = no(1) , namely, an
algorithm with only O(m log ∆) runtime with high probability (basically, replacing log n-term in
our main results with a log ∆-term instead). In this regime, previously, the best known bounds
where O(m∆4 log ∆) time randomized algorithm of [BD24] for (∆+1)-edge coloring and O(m log ∆)
time randomized algorithm of [Ass25] for (∆ + O(log n))-edge coloring.

Corollary C.1. There are randomized algorithms that, given any undirected multigraph G on n
vertices and m edges with maximum degree ∆ and maximum edge multiplicity µ, in O(m log ∆)
time, with high probability, output:

    1. a (∆ + 1)-edge coloring of any simple graph G (Vizing’s theorem for simple graphs);

    2. a (∆ + µ)-edge coloring of any multigraph G (Vizing’s theorem for multigraphs);

    3. a ⌊3∆/2⌋-edge coloring of any multigraph G (Shannon’s theorem for multigraphs).

    Corollary C.1 improves the runtime dependence of Theorem 1.1, Theorem A.1, and Theorem B.1
from O(m log n) time to O(m log ∆) time instead. In the following, we prove the first part of this
result, namely, for (∆+1)-edge coloring of simple graphs, and then show how this argument extends
to the other two parts as well.

C.1     Proof of Corollary C.1 for Simple Graphs
Recall that by Lemma 10.4, we already have an algorithm that, with high probability, in O(m log ∆)
time, (∆ + 1) colors all but m/∆ edges of the graph. Thus, our task is now to extend the coloring
to these edges as well. Previously, in Lemma 10.5, we did this by coloring a constant fraction of
the edges repeatedly (using Lemma 10.2); since, we need Θ(log (m/∆)) attempts to color all edges
and each one requires O(m) time, this lead to an O(m log n) time algorithm.
    To prove Corollary C.1, we split this final task: we will again run the algorithm of Lemma 10.2,
but this time to reduce the number of uncolored edges from m/∆ to m/ poly(∆) for some arbitrarily
large polynomial in ∆; thus, we need Θ(log ((m/∆)/(m/ poly(∆)))) = Θ(log ∆) steps of coloring,
and hence O(m log ∆) time in total. Then, to handle the very last m/ poly(∆) remaining edges, we
will run the algorithm of [BD24], which extends the coloring to each uncolored edge in essentially
(but not formally) poly(∆) time per uncolored edge for some polynomial in ∆. Thus, the second
part of the algorithm only takes O(m) time now, leaving us with an O(m log ∆) time algorithm
all in all. We formalize this approach in the following (which specifically requires extra care for
obtaining a high probability guarantee).

Lemma C.2. Let c ≥ 1 be any arbitrary fixed constant and λ0 := max{m/∆2c , m log n/n}. There
is an algorithm that, given a graph G, a partial (∆ + 1)-edge coloring χ with λ = m/∆ uncolored
edges, extends the coloring to all but Θ(λ0 ) uncolored edges in O(m log ∆) time with high probability.

Proof. We simply run our Lemma 10.2 with the given parameters λ and λ0 . By Lemma 10.2, the
expected runtime will be some T0 = O(m log (λ/λ0 ) + ∆λ) = O(m log ∆) since λ0 ≥ m/∆2c for


                                                  58
some absolute constant c ≥ 1. Moreover, by the same lemma,
  Pr [runtime of the algorithm ≥ 2000 · T0 ] ≤ exp (−10λ0 ) ≤ exp (−Θ(m log n)/n) = 1/ poly(n),
since λ0 ≥ m log n/n. Thus, the algorithm finishes in O(m log ∆) time with high probability.

    We now present the final part for coloring the last remaining edges. For this step, we use the
following result from [BD23] (itself based on [Ber22, Chr23]) that allows for coloring a random
uncolored edge in essentially (but not exactly) poly(∆) time. We note that a recent work of [BD24]
presents a further improvement to these bounds (in terms of the exponent of ∆) but as our de-
pendence on ∆ is only logarithmic, this improvement is inconsequential and thus we stick with the
simpler work of [BD23].
Proposition C.3 ([BD23]). There is an algorithm that, given a graph G, a partial (∆ + 1)-edge
coloring χ, and λ uncolored edges, extends the coloring to a randomly chosen uncolored edge in
expected poly(∆) · log (n/λ) time.

We use this result to obtain the following lemma for our purpose.
Lemma C.4. Let c be the poly(∆)-exponent in Proposition C.3, and λ0 := max{m/∆2c , m log n/n}.
There is an algorithm that, given a graph G, a partial (∆ + 1)-edge coloring χ and λ = Θ(λ0 ) un-
colored edges, extends the coloring to all uncolored edges in O(m log ∆) time with high probability.

Proof. Suppose first that the max-term of λ0 is the second one, i.e., λ0 = m log n/n. This implies
that ∆ ≥ (n/ log n)1/2c and since c is a constant, we have log(∆) = Θ(log n) in this case. We can
use the standard Vizing’s fan and Vizing’s chain approach in Lemma 4.2 to color the remaining
edges deterministically in O(λ0 · n) = O(m log n/n · n) = O(m log ∆) time in this case.
    Now consider the case when λ0 = m/∆2c . We run the algorithm of Proposition C.3 with the
following modification: when coloring a random edge, if the runtime becomes two times more than
the expected time guarantee of Proposition C.3, we terminate the coloring and repeat from the
beginning. Thus, each step takes O(∆c · log (n/i)) time deterministically and succeeds in extending
the coloring by one edge with probability at least half.
   For i ∈ [λ], define αi := 4 log (n/i) and Xi as the random variable denoting ℓi · αi where ℓi is
the number of times the algorithm attempts to color the next random edge until it succeeds when
there are only i edges left uncolored. We first prove that
                                                     λ
                                                     X
                                                           αi = o(m/∆c ),                                             (2)
                                                     i=1

and then show that
                                              " λ        #
                                               X      2m
                                            Pr    Xi ≥ c ≪ 1/ poly(m),                                                (3)
                                                      ∆
                                               i=1

which concludes
       Pλ       the proof in this case as well since the runtime of coloring the remaining edges is
    c
O(∆ · i=1 Xi ) as stated earlier.
     0



Proof of Equation (2). We have,
            λ            λ                                 λ
                                                                     !
                                                                                  nλ
                                                                                                           
            X            X             n                 Y n                                        e · n λ
                  αi =         4 log         = 4 log                      = 4 log          ≤ log
                                        i                        i                λ!                    λ
            i=1          i=1                               i=1
                                                                          (using the inequality x! ≥ (x/e)x for all x ≥ 1)

                                                                     59
                                       e · n · ∆2c
                                                    
                        m
                  = 4 · 2c · log                         = o(m/∆c ).      (as λ = m/∆2c and m ≥ n)
                       ∆                    m

Proof of Equation (3). Note that for any i ∈ [λ] and δ > 0,
                 Pr [Xi ≥ (1 + δ) · αi ] = Pr [ℓi ≥ (1 + δ)4] ≤ 2−(3+4δ) ≤ exp (−δ) ,
where the first inequality is because the probability that ℓi increases by one each time is at most
1/2 by Markov’s inequality.
    While Xi ’s are technically not independent (the choice of some edge being colored may change
the structure of the graph for coloring next uncolored edge), they are stochastically dominated by
independent random variables with above tail bounds (given the guarantee of Proposition C.3 for
each uncolored edge). Thus, we can apply Proposition 10.3 with parameters {αi }λi=1 as above and
βi = 1 for all i ∈ [λ] to obtain that for t := m/∆c ,
         " λ              #      " λ          λ
                                                      #                    
           X          2m          X          X                     1     m
       Pr      Xi ≥ c ≤ Pr            Xi ≥      αi + t ≤ exp −         ·      ≪ 1/ poly(m),
                      ∆                                         4 log n 2∆c
          i=1                   i=1           i=1

where the first two inequalities use Equation (2) for i αi and the third uses ∆c ≤ (n/ log n)c/2c <
                                                      P
√
  n in this case (when λ0 is the first-term of its maximum). This concludes the proof.

    The proof of Corollary C.1 for (∆ + 1) edge coloring is now as follows: run Lemma 10.4 to color
all but m/∆ edges. Then, let c be the constant in the exponent of poly(∆) in Proposition C.3
and run Lemma C.2 to extend the coloring to all but λ0 := max{m/∆2c , m log n/n} edges. Finally,
run Lemma C.4 to extend the coloring to all these remaining λ0 edges. Since each of these algorithms
take O(mlo∆) time and succeed with high probability, we obtain the final algorithm.

C.2    Proof of Corollary C.1 for Multigraphs
To extend the proof to multigraphs, recall that by Lemma A.10 and Lemma B.10, we can, respec-
tively, (∆ + µ) color or ⌊3∆/2⌋ color all but m/∆ edges of a multigraph in O(m log ∆) time. Thus,
we can follow the same exact strategy as in (∆ + 1) coloring approach outlined in the previous
subsection, and we only need an analogue of the result of [BD23] to conclude the proof. Such
results are also already established by [Dha24a] and we do not need to reinvent the wheel here.
Proposition C.5 (A subroutine in [Dha24a, Theorem 1.6; part 1]). There is an algorithm that,
given a multigraph G, a partial (∆ + µ)-edge coloring χ, and λ uncolored edges, extends the coloring
to a randomly chosen uncolored edge in expected poly(∆) · log (n/λ) time.
Proposition C.6 (A subroutine in [Dha24a, Theorem 1.5; part 1]). There is an algorithm that,
given a multigraph G, a partial ⌊3∆/2⌋-edge coloring χ, and λ uncolored edges, extends the coloring
to a randomly chosen uncolored edge in expected poly(∆) · log (n/λ) time.

The rest of the proof is verbatim as in simple graphs in Corollary C.1 and we omit it here.


D     A Specialized Concentration Inequality
Throughout the paper, for optimizing log factors in the runtime of our algorithms, we needed the
following specialized concentration inequality. For completeness, we now prove this inequality using
a standard argument based on moment generating functions.

                                                          60
 Proposition (Restatement of Proposition 10.3). Let {Xi }ni=1 be n independent non-negative ran-
 dom variables associated with parameters {αi }ni=1 and {βi }ni=1 such that for each i ∈ [n], αi , βi ≥ 1,
 and for every δ > 0,
                                Pr [Xi ≥ (1 + δ) · αi ] ≤ exp (−δ · βi ) ;
 then, for every t ≥ 0,
                     " n      n
                                      #                                             n
                                                                                                 !!
                      X       X                  n   βi                               X αi
                  Pr     Xi ≥   αi + t ≤ exp − min        ·                     t−2                    .
                                                i=1 2αi                                     βi
                      i=1            i=1                                              i=1
                                                                                                            Pn
 Proof. Let s := 1/2 · mini βi /αi and for i ∈ [n], Yi := max(Xi − αi , 0). Letting Y :=                        i=1 Yi , leads
                                    " n         n
                                                          #
                                     X         X
                                 Pr      Xi ≥       αi + t ≤ Pr [Y ≥ t] ,
                                           i=1       i=1

 for all t ≥ 0. We further have,
                                                                  " n        #           n
                                                                   Y                     Y
                                       −s·t               −s·t                    −s·t
                       s·Y    s·t               s·Y                    s·Yi
                                                                                           E es·Yi
                                                                                               
       Pr [Y ≥ t] = Pr e    ≥e       ≤e     ·E e       =e      ·E     e        =e      ·                                  (4)
                                                                          i=1                      i=1

 where the inequality is by Markov inequality and the final equality is by the independence of the
 variables. Now, for every i ∈ [n],
                        Z ∞
               s·Y 
            E e    i
                      =      Pr [Yi = y · αi ] · exp (s · y · αi ) dy
                         y=0
                        Z ∞
                      =      Pr [Xi = (1 + y) · αi ] · exp (s · y · αi ) dy
                            y=0
                                                           (by the connection between Xi and Yi when y ≥ 0)
                          Z ∞
                      ≤           exp (−y · βi ) · exp (s · y · αi ) dy
                            y=0
                                               (by the tail inequality of Xi in the proposition statement)
                                    1
                      =
                         βi · (1 − s · (αi /βi ))
(since the integral of e−Kx = −e−Kx /K and s < βi /αi thus the value of this function in ∞ is zero)
                                 1
                      ≤                                                                         (as βi ≥ 1)
                         1 − s · (αi /βi )
                      ≤ exp (2s · αi /βi ) .            (as s · αi /βi < 1/2 and 1 − x ≥ e−2x for x < 1/2)

 Plugging this bounds in Equation (4), implies that
                                    n                                                            n
                                               !                                                              !!
                                   X        αi             n   βi                                X αi
         Pr [Y ≥ t] ≤ exp −s · t +     2s ·      = exp − min        ·                  t−2                           ,
                                            βi            i=1 2αi                                          βi
                                            i=1                                                  i=1

 by the choice of s, concluding the proof.




                                                           61
