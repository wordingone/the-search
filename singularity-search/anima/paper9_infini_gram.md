# paper9 infini gram

> **Original PDF Size:** 5.5M
> **Text Size:** 84K
> **Conversion Date:** 2026-01-22 07:34:35

---

                                            I NFINI - GRAM MINI: Exact n-gram Search at the Internet Scale
                                                                   with FM-Index

                                        Hao Xu♡ Jiacheng Liu♡♠ Yejin Choi♢ Noah A. Smith♡♠ Hannaneh Hajishirzi♡♠
                                         ♡
                                           Paul G. Allen School of Computer Science & Engineering, University of Washington
                                                             ♠
                                                               Allen Institute for AI ♢ Stanford University



                                                          Abstract

                                        Language models are trained mainly on mas-
arXiv:2506.12229v5 [cs.CL] 6 Jan 2026




                                        sive text data from the Internet, and it becomes
                                        increasingly important to understand this data
                                        source. Exact-match search engines enable
                                        searching in large text corpora – counting string
                                        appearances and retrieving the enclosing docu-
                                        ments – yet the high storage overhead hinders
                                        their application on Internet-scale data. We
                                        present INFINI - GRAM MINI, an efficient and
                                        scalable system that can make petabyte-level
                                        text corpora searchable. Based on the FM-
                                        index data structure (Ferragina and Manzini,
                                        2000), which simultaneously indexes and com-
                                        presses text, our system creates indexes with
                                        size only 44% of the corpus. I NFINI - GRAM         Figure 1: Overview of INFINI - GRAM MINI. Based on
                                        MINI greatly improves upon the best existing        the FM-index data structure, INFINI - GRAM MINI sup-
                                        implementation of FM-index in terms of index-       ports efficient exact-match search in massive text cor-
                                        ing speed (18×) and memory use during both          pora (n ≃ 1015 bytes) while reducing the index size
                                        indexing (3.2× reduction) and querying (down        down to 7% compared to a canonical suffix array in-
                                        to a negligible amount). We index 83TB of           dex. Searching naively in the corpus would have time
                                        Internet text in 99 days with a single CPU node     complexity of O(n) and is thus impractical; with INFINI -
                                        with 128 vCPUs (or 19 hours if using 137 such       GRAM MINI , the search time complexity is independent
                                        nodes). We show one important use case of           of n. |Q| is the length of query string and can be arbi-
                                        INFINI - GRAM MINI in a large-scale analysis        trarily long, and H0 ≈ 2.1 is the zeroth-order entropy
                                        of benchmark contamination. We find several         of the text corpus.
                                        core LM evaluation benchmarks to be heavily
                                        contaminated in Internet crawls (up to 74.2% in
                                        GSM8K), which could lead to overestimating          1   Introduction
                                        the capabilities of language models if trained
                                        on such data. We host a benchmark contamina-
                                                                                            Modern language models (LMs) are trained mainly
                                        tion bulletin to share the contamination rate of
                                        many core and community-contributed bench-          on text datasets downsampled from massive,
                                        marks. We also release a web interface and an       petabyte-level text corpora like Common Crawl
                                        API endpoint to serve general search queries        (Common Crawl Foundation, 2025). As these LMs
                                        on INFINI - GRAM MINI indexes.                      are deployed more broadly, it becomes more press-
                                                                                            ing to understand the training data and its effects on
                                                                                            model behavior (Liu et al., 2025; Han and Tsvetkov,
                                          Project Home infini-gram-mini.io                  2022). As a starting point, we want to make these
                                          Web Interface infini-gram-mini.io/demo            text corpora searchable; in particular, searching
                                          API Endpoint api.infini-gram-mini.io              for exact matches of long sequences has gained in-
                                         Documentation infini-gram-mini.io/docs             creasing interest (Elazar et al., 2024; Merrill et al.,
                                          Source Code infini-gram-mini.io/code              2024; Lu et al., 2024). The size of these corpora
                                         Contam Bulletin infini-gram-mini.io/bulletin       makes this problem extremely challenging, creat-
ing demand for more efficient indexing techniques.      tamination of benchmarks widely used in state-
   Prior systems for exact-match search build in-       of-art LM evaluations (§4). I NFINI - GRAM MINI
dexes several times as large as the text corpora.       allows us to do the analysis on larger corpora than
Merrill et al. (2024) index 1.3TB of text with a suf-   prior works and on new benchmarks uploaded.
fix automaton (storage multiplier 29×); Liu et al.      This would be much more expensive if using other
(2024) index 12TB of text with a suffix array (stor-    indexing methods. We find several core bench-
age multiplier 6×); Elazar et al. (2024) index 35TB     marks to be heavily contaminated (§4.2). I NFINI -
of text with the proprietary ElasticSearch engine       GRAM MINI detects exact overlap for question of
(storage multiplier about 2×). The size of these        contaminated entries in text corpora, among which
indexes renders it impractical to apply them to         a large majority of questions appear together with
petabyte-level corpora.                                 the correct answer. This reveals a dire evalua-
   To address this challenge, we index text corpora     tion crisis: as benchmarks get increasingly con-
with FM-index (Ferragina and Manzini, 2000), a          taminated by Internet crawls and consequently
compact data structure frequently used in bioinfor-     LM training data, evaluation results may give in-
matics (Guo, 2025; Depuydt et al., 2023; Wang           flated estimates of true model capabilities. As
et al., 2018; Li, 2014), but not yet used for natural   such, we host a benchmark contamination bulletin
language data at scale. Prior work only applies this    to continually monitor contamination of core and
data structure to 13.4 GB dataset (Bevilacqua et al.,   community-contributed benchmarks on new Inter-
2022). We explain FM-index in detail in §2. For         net snapshots, and we call for more community
natural text, the size of FM-index can be made as       attention on this matter.
small as 0.26× of the corpus; practically, to ensure       Beyond contamination analysis, INFINI - GRAM
low query latency, we build FM-index with a stor-       MINI opens up more impactful use cases such as
age multiplier of 0.44×, or 7% compared to the          task-specific dataset construction and pretraining
canonical suffix array.                                 data curation. We release a web interface and
   I NFINI - GRAM MINI is the system we developed       API endpoint of INFINI - GRAM MINI, so that ev-
for efficiently constructing the FM-index at scale      eryone can search in the text corpora that we have
and answering search queries. For indexing, we          indexed. We plan to continue indexing new cor-
extend and combine components from Liu et al.           pora and share regular updates. We also release our
(2024), Gog et al. (2014), and Labeit et al. (2017)     source code. We hope this tool can enable more
into a highly parallelized program, which achieves      insightful analysis and better use of massive text
a 18× speedup and uses only 32% as much RAM             corpora.
compared to the best existing implementation by
SDSL (Gog et al., 2014).1 We use INFINI - GRAM
                                                        2     Background: FM-index
MINI to construct FM-index for two LM pretrain-         The FM-index (Ferragina and Manzini, 2000) is a
ing datasets – the Pile and DCLM-baseline – and 7       full-text index that supports efficient pattern match-
recent Common Crawl dumps from January to July          ing, counting, and text retrieval on a highly com-
2025. Altogether, we indexed 83TB of text in 99         pressed representation of the text corpus. Com-
days with a single CPU node with 128 vCPUs, and         pared with a canonical suffix array, FM-index
this could have been done in 19 hours if embarrass-     stores a compact variation of a suffix array and
ingly parallelized across 137 such nodes. We esti-      the text corpus, greatly reducing storage overhead.
mate that indexing the full Common Crawl would
take 1,200 node-days, or 19 hours if parallelized       2.1    Data Structures and Implementation
across 1,500 nodes. For answering queries, we ex-       On a high level, FM-index has two core compo-
tend SDSL to work with on-disk index, reducing          nents: (1) a sampled suffix array and its inverse, and
the RAM requirement to a negligible amount. Our         (2) the Burrows-Wheeler transform represented us-
inference engine supports counting the occurrences      ing a Huffman-shaped wavelet tree. Below we de-
of a query string and retrieving documents that con-    scribe each component as applied to a single string
tain the query string, both within seconds when         (appropriate for our application; see §3). Figure 2
working on the above corpora.                           shows a toy example illustrating the data structure.
   We apply INFINI - GRAM MINI to analyzing con-
                                                        Suffix array and sampling. The suffix array SA
   1
       https://github.com/simongog/sdsl-lite            of a string T of length n is an array of integers
                                                          from the index.

                                                          Huffman-shaped wavelet trees. Given the BWT
                                                          L and the sampled SA, we can reconstruct the orig-
                                                          inal string T from these data structures so we don’t
                                                          need to store T . However, L still takes as much
                                                          storage as T . To further compress L, a wavelet tree
                                                          is used to represent L by hierarchically partitioning
                                                          the alphabet. In the wavelet tree, each leaf node rep-
Figure 2: The FM-index data structure (§2.1) used in      resents a symbol in the alphabet, and each non-leaf
INFINI - GRAM MINI , shown for a toy string with length   node stores a bitvector marking whether the symbol
n = 7. The suffix array is sampled with a sampling        at each position of L belongs to the left or right sub-
rate a = 3 and only elements corresponding to bolded      tree. A Huffman-shaped wavelet tree (Mäkinen and
suffixes are stored. The BWT can be derived from the      Navarro, 2005) optimizes the hierarchy by group-
SA, and is stored in compressed form as a Huffman-
                                                          ing and coding symbols based on their frequencies
shaped wavelet tree.
                                                          in L. Storing it costs (nH0 + 2σ log n) bits, where
                                                          σ is the alphabet size, and H0 is the zeroth-order
representing the starting positions of all suffixes       entropy2 of L and H0 ≈ 0.26 log2 σ in our exper-
of T in lexicographic order. Formally, SA[i] = j          iments with natural language text; this is smaller
if the suffix T [j...n] ranks ith among all suffixes      than storing L directly, which is n log2 σ bits. Be-
in lexicographical ordering. The suffix array en-         sides compression, the tree can efficiently support
ables quickly locating the positions where a pat-         two operations crucial to LF mapping: rank(c, ℓ)
tern appears in the string. Canonically, storing the      counts the number of occurrences of a character c
whole suffix array would take O(n log n) space,           in L[0...ℓ − 1], and select(c, m) finds the position
or 5n bytes for strings with length up to one tril-       of the mth occurrence of c in L. Both operations
lion (Liu et al., 2024). To reduce storage overhead,      have a time complexity of O(H0 ).
FM-index samples SA at regular intervals, stor-           Inverse suffix array. To allow reconstructing
ing only every ath entry where a is a parameter.          part of the original string T from any given po-
When querying the index, if we need to access             sition, FM-index also stores a sampled version of
an unsampled SA entry, we can recover its value           the inverse suffix array ISA. ISA is the inverse
by referencing the BWT data structure, which is           of the permutation specified in SA and is defined
introduced below.                                         as ISA[j] = i if SA[i] = j. When reconstructing
                                                          T from a given position, ISA is used to identify
Burrows-Wheeler transform (BWT). BWT                      the corresponding rank in the suffix array to start
(Burrows et al., 1994) rearranges the string T (with      reconstruction. In FM-index, we can use a different
termination symbol $) into a reversible permuta-          sampling rate for ISA, storing only every bth entry,
tion L that clusters repeated symbols. The BWT            where b can be larger than a as ISA is used less.
is defined as a string L where L[i] = T [SA[i] − 1]
if SA[i] > 0, or L[i] = $ otherwise. Intuitively, L       2.2      Querying the FM-index
is the concatenation of the symbol preceding each
                                                          There are three basic operations supported by FM-
suffix when the suffixes are sorted in the SA order
                                                          index: find, locate, and reconstruct. Counting a
(see Figure 2).
                                                          pattern can be done with a find operation, and re-
   A key property of BWT is the Last-to-First (LF)        trieving segments containing the pattern from the
mapping, which maps the ith occurrence of a sym-          original string can be done with a combination of
bol c in L to the ith occurrence of it in F , where       all three operations. See App. §A for more details.
F [i] = T [SA[i ]]. The LF mapping is defined as
LF (i) = C[c] + rank(c, i), where C[c] is the num-        3       I NFINI - GRAM MINI
ber of symbols in L lexicographically smaller than
c, and rank(c, i) counts the occurrences of c in          We develop INFINI - GRAM MINI as a scalable and
L[0...i]. This property allows us to traverse the         efficient implementation of FM-index on natural
string in reverse order, which is essential for finding       2
                                                               zeroth-order entropy measures the unigram probability
patterns in and reconstructing (part of) the string T     distribution of individual symbols (i.e. UTF-8 characters).
language data. For indexing, INFINI - GRAM MINI            Partitioning. For large corpora, we partition it
achieves 18× speedup and reduces RAM usage by              into multiple shards and index each shard indepen-
3.2× compared to the previous best existing imple-         dently. Searching across multiple shards produces
mentation in SDSL. This paves way for indexing             the same result as searching a single, larger index,
petabyte-level text corpora with reasonable time           and this allows us to build indexes with limited
and compute. For querying, we extended the im-             RAM per node as well as embarrassingly paral-
plementation in SDSL to work with indexes stored           lelize across multiple nodes.
on disk, allowing us to query large indexes on ma-
                                                           Indexed corpora. We have built the index for
chines with limited RAM.
                                                           the following corpora: the Pile (1.3TB training set,
   In contrast to the original infini-gram which tok-
                                                           1.4GB validation set; Gao et al., 2020), DCLM-
enizes the text, we construct the index directly on
                                                           baseline (17TB; Li et al., 2024), and the Common
raw UTF-8 bytes, which saves tokenization time
                                                           Crawl between January and July 2025 ( “CC-2025-
and allows more flexibility when querying. Tok-
                                                           05” to “CC-2025-30” ; 7 crawls and 65TB total;
enization is a technique to reduce index size, but
                                                           Common Crawl Foundation, 2025).3
since FM-index is already a compressed index, to-
kenization would not be helpful here.                      Indexing time. We use CPU nodes with 128 vC-
                                                           PUs and 2TiB RAM to construct indexes, and under
3.1   Index Construction                                   this constraint, each shard can be as large as 700GB
Optimizing the indexing steps. We use a paral-             which can be indexed in 12–19 hours. Table 1 re-
lelized implementation of each indexing step. For          ports the time for indexing the above corpora. In-
building SA and BWT, we adapt from the parallel            dexing time shows a slightly super-linear increase
implementation in Liu et al. (2024). For building          with respect to shard size, and we report detailed
the wavelet tree, we use the parallel implemen-            stepwise breakdown in App. §B. If we were to in-
tation in Labeit et al. (2017). For sampling SA            dex the full Common Crawl dataset (about 1PB),
and creating a sampled version of ISA, we par-             we would split it into 1,500 shards, which can be
allelized the implementation in SDSL. All these            indexed in 1,200 node-days.
steps were single-threaded in SDSL, and by paral-
                                                           Index size. We choose the sampling rate empiri-
lelizing them we achieve significant speedup.
                                                           cally to balance storage savings and query latency.
   We measured the time and peak RAM usage
                                                           In our implementation, we sample every a = 32
for indexing an 8.7 GB corpus (a single file from
                                                           entries of SA and every b = 64 entries of ISA. This
DCLM-baseline) using implementation of SDSL
                                                           yields indexes with 0.44× the size of the corpora
and INFINI - GRAM MINI. Their implementation re-
                                                           (Table 1). Conceptually, if we set a and b to be very
quired 5,847 seconds and 74,807 MB of peak RAM,
                                                           big, then the SA and ISA would have negligible
whereas INFINI - GRAM MINI completed indexing
                                                           size, and the index can be as small as 0.26× the
in 324 seconds (18× speedup) and peak RAM of
                                                           size of the corpus.
23,742 MB (3.2× reduction).
                                                           3.2    Querying with INFINI - GRAM MINI
Preprocessing. FM-index is designed to work on
a single string. To index a text corpora, which con-       Similar to other exact-match search engines,
sists of a collection of documents, we encode all          INFINI - GRAM MINI supports two types of query:
documents with UTF-8 and concatenate all these             counting the number of occurrences of a string, and
byte arrays with the \xff byte (not used by UTF-8)         retrieving documents that contain a string. At query
to mark document boundaries. We then construct             time, INFINI - GRAM MINI keeps all index files on
FM-index for this big byte string. We work on              disk as memory-mapped files, thus requiring only
UTF-8 bytes rather than characters to keep the al-         minimal CPU and RAM (loading all indexes uses
phabet size small (σ = 256). Along with this text          only ∼ 30 MB if RAM and 1 vCPU in our setting).
index, we also store a text offset file that records the      Querying in INFINI - GRAM MINI is slower and
starting position of each document, which is useful        more complex than in canonical suffix arrays due
for retrieving whole document and metadata. Aside          to the compressive nature of FM-index. The SA is
from the actual text, we also index the metadata           subsampled, and the original text is shuffled and
of the documents to make the metadata searchable              3
                                                               We extract text from the CC crawls with resiliparse, fol-
while storing it in a similar compressed form.             lowing Li et al. (2024).
          Dataset            Original Size     Indexing Time             Index Size    Num. Shards
                                 (TB)         (CPU node-days)              (TB)
          Pile-validation      0.0001345           0.00057          0.000602 (0.45×)        1
          Pile-train             1.308               1.31             0.588 (0.45×)         2
          DCLM-baseline          16.666              12.6             7.523 (0.45×)         25
          CC-2025-05              9.079              11.8             3.972 (0.44×)         15
          CC-2025-08              8.163              10.6             3.574 (0.44×)         15
          CC-2025-13             10.393              13.6             4.563 (0.44×)         17
          CC-2025-18             10.498              13.7             4.664 (0.44×)         17
          CC-2025-21              9.221              12.1             4.302 (0.46×)         15
          CC-2025-26              8.724              11.4             3.835 (0.44×)         15
          CC-2025-30              9.006              11.7             3.952 (0.44×)         15
          Total                  83.059              98.8            36.884 (0.44×)         137

Table 1: Text corpora we indexed with INFINI - GRAM MINI, along with indexing time and index size. The reported
numbers only include actual document content and do not include metadata.


compressed, both requiring additional random disk           stored on Google Cloud Platform (GCP) SSD disks
reads to recover. The latency of queries is largely         with 80,000 IOPS and 1200 MB/s throughput. We
determined by the disk I/O performance.                     measure the average latency over 100 queries for
                                                            each query type and setting. For counting, short
Counting a string. We use FM-index’s find op-
                                                            queries (|Q| ≤ 10) can be handled within 0.4
eration to count the number of occurrences of a
                                                            seconds for all corpora we indexed, while longer
query string. We parallelize this operation across
                                                            queries (|Q| = 1000) are handled in 8 seconds for
all k shards, and report the sum of counts across all
                                                            CC-2025-05 and 25 seconds for DCLM-baseline.
shards. The total number of random disk reads is
                                                            The difference in query latency is caused by the
O(k|Q|H0 ), where |Q| is the length of the query
                                                            different number of shards in the corpora. For doc-
string. If the disk’s I/O throughput is high enough,
                                                            ument retrieval, retrieving a snippet of d = 100
the query latency can be as low as O(|Q|H0 ).
                                                            bytes can be done within 2 seconds for all corpora,
Retrieving documents. We first use the find op-             and retrieving d = 3000 bytes takes up to 4.5 sec-
eration to get the range in the SA that corresponds         onds. See App. §C for benchmarking details.
to the query string. Each element in this range indi-
cates one appearance of the query string in the text        3.3    Web Interface and API Endpoint
corpus. If the index has multiple shards, we will           We host a Hugging Face interface for easy access
have one range for each shard. For each element in          to counting and document retrieval with 83 TB
this range, we can use the locate operation to find         corpora (App. §E). We also release an API endpoint
the position of the match in the original text T , then     that allows programmatically submitting requests.
get the boundaries of the enclosing document with
a binary search in the text offset file, and finally        4     Analyzing and Monitoring Benchmark
use the reconstruct operation to get the document                 Contamination with INFINI - GRAM
text. The number of random disk reads for locate                  MINI
is O(aH0 ), and for reconstruct is O((b + d)H0 )
                                                            In this section, we showcase how INFINI - GRAM
where d is the document length. We parallelize
                                                            MINI can be used to analyze benchmark contam-
reconstruct operation by dividing document text
                                                            ination at scale. I NFINI - GRAM MINI allows us
into up to t = 10 chunks, or chunks of length
                                                            to search in the largest body of text ever in the
100 if d < 1000. Retrieving a single document is
                                                            open-source community with minimal storage over-
parallelized across t threads, and it can be further
                                                            head, enabling large-scale benchmark contamina-
parallelized for retrieving multiple documents.
                                                            tion analysis at low cost. Our work also supports
Query latency. I NFINI - GRAM MINI presents                 analyzing new benchmarks uploaded later.
second-level latency on both types of query. We                By identifying lexical overlap between 24
benchmark the query latency with the index files            widely-used evaluation benchmarks and three ma-
jor corpora, we find non-trivial contamination          of textual context. In total, we analyze 24 bench-
across 12 benchmarks (§4.2). We then retrieve doc-      marks; see App. §F for the full list and citations.
uments accounting for contamination for further            For benchmarks with question-answering format,
analysis (§4.3). Powered by INFINI - GRAM MINI,         we specifically check the question entries for con-
we develop a monitoring system with a bench-            tamination. Questions are usually longer and con-
mark contamination bulletin, where we continu-          tain sufficient contextual information to uniquely
ously monitor new crawls and report benchmarks’         identify the benchmark example, whereas answers
contamination status over time (§4.4). We also          can be short, such as a multiple choice option or
invite the community to contribute by suggesting        a number. For language understanding tasks that
additional benchmarks or uploading new ones for         involve reading a context or paragraph before an-
analysis.                                               swering, we check the context entries for contami-
   I NFINI - GRAM MINI has further potential applica-   nation.
tions where exact-match search is needed, includ-          We evaluate on only the test set of benchmarks.
ing (1) task-specific dataset construction, where       For benchmarks with large test set, we downsam-
creation of those datasets requires retrieving docu-    ple to 1,000 entries for efficiency. For benchmarks
ments containing specific terms or phrases, and (2)     with multiple subtasks, we sample proportionally
data curation, where INFINI - GRAM MINI can assist      from each subtask to maintain representative distri-
in identifying and removing duplicate, low-quality,     bution.
or sensitive text and documents. We leave these
directions for future exploration.                      4.2   Results
                                                        Table 2 reports the percentage of dirty entries in
4.1   Setup
                                                        benchmarks against the Pile (knowledge cutoff in
Contamination detection method. We check                2020), DCLM-baseline (knowledge cutoff in 2022),
contamination of test examples by measuring lex-        and seven CC crawls (knowledge cutoff in January-
ical overlap with the text corpus, following stan-      July 2025). The detailed count of dirty and suspi-
dard practice in the literature (Brown et al., 2020;    cious entries can be found in App. §G.
Llama Team, 2024). Given a text entry, we extract
all 50-character substrings S with a stride of one      Many widely-used benchmarks are highly
word, and determine entry contamination based           contaminated. When checking against DCLM-
on the proportion of substrings that appear at least    baseline, MMLU has 27.70% dirty entries, MMLU-
once in the corpus:                                     Pro has 16.20%, ARC-Challenge has 32.6%, and
                P                                     ARC-Easy has 32.3%. These benchmarks, espe-
                   s∈S I count(s) > 0                   cially MMLU, have been used to evaluate virtually
           η=                            .
                           |S|                          every new LM in recent years. We believe that the
                                                        observed contamination levels are a strong signal
   We classify an entry into three contamination
                                                        that many recently reported results may overesti-
levels using the same thresholds with different nam-
                                                        mate language model abilities on truly new, unseen
ings as Touvron et al. (2023):
                                                        evaluation items.
• Clean, if η < 20%
• Suspicious, if 20% ≤ η < 80%                          Larger and newer corpora show greater contam-
• Dirty, if η ≥ 80%                                     ination. Compared with Pile-train, most bench-
                                                        marks show a higher dirty rate on DCLM-baseline.
Benchmarks. To comprehensively evaluate con-            For example, ARC-Challenge has 17× more and
tamination, we analyze a broad range of bench-          MMLU has 1.1× more dirty entries. However, CC-
marks widely used to evaluate state-of-the-art LMs.     2025-05 has lower contamination rate than DCLM-
We categorize benchmarks into five groups based         baseline on most benchmarks, which is likely be-
on their primary focus: (1) knowledge and reason-       cause DCLM-baseline corpus is larger and is a
ing, i.e., understanding and reasoning over factual     high-quality subset of crawls spanning a decade,
and conceptual knowledge; (2) math, i.e., mathe-        while CC-2025-05 is an unfiltered single crawl.
matical reasoning ability; (3) code generation and
modification abilities; (4) commonsense under-          Contamination level varies by domain. Bench-
standing, i.e., commonsense knowledge and rea-          marks in historically-significant domains, such
soning abilities; and (5) reading comprehension         as commonsense understanding (e.g., ARC,
OpenbookQA) and reading comprehension (e.g.,                 1% on DCLM-baseline, and 3% on CC-2025-05.
SQuAD, CoQA), tend to show higher dirty rates
                                                            We show 7 contamination examples in App. §H
on all corpora, while those in emerging domains
                                                         and their source of contamination. We found that
like math (e.g., GSM8K, MATH-500) and code
                                                         contamination is caused by (1) the benchmark is
(e.g., HumanEval, LiveCodeBench) remain rela-
                                                         sourced from the Internet (e.g., AIME-2024, Fig-
tively clean at this writing.
                                                         ure 6; SWE-bench, Figure 8), (2) there are other
New benchmarks are gradually getting contam-             benchmarks or online quizzes that are sourced from
inated in recent corpora. Newer benchmarks               the benchmark (e.g, GSM8K, Figure 10), (3) blogs
are initially clean on earlier corpora. For example,     and papers cite a benchmark entry as example (e.g,
AIME-2024 is uncontaminated on Pile-train and            GPQA, Figure 9; BigBenchHard, Figure 11; Open-
DCLM-baseline, but show 40.00% dirty rate on the         bookQA, Figure 12), and (4) benchmark entry co-
more recent CC-2025-21; similarly, GPQA shows            incide with online source (e.g., MMLU, Figure 7).
2.70% dirty rate on CC-2025-26. GSM8K is very               Contamination could cause LLM to overperform
clean on Pile-train, DCLM-baseline, and earlier          on evaluation benchmarks by enabling models to re-
CC-2025 crawls, but has 74.2% dirty entries on           trieve memorized answers from training data rather
CC-2025-21.                                              than performing task-specific reasoning. Our find-
                                                         ing shows that a large majority of dirty entries con-
4.3   Analysis                                           tain exact matches of both question and answer,
With INFINI - GRAM MINI, we can retrieve docu-           though even subtler forms of contamination such
ments from the corpus that contain contaminated          as question-only matches and paraphrased answers
examples for further analysis. We use LLM-as-a-          could inflate benchmark scores. As training cor-
judge to categorize all dirty instances in Pile-train,   pora grow, the risk of benchmark contamination
DCLM-baseline, and CC-2025-05 into following             and the potential for rote memorization increase,
scenarios (see App. §I for more details), with ex-       making it more important to decontaminate training
amples shown in Table 3:                                 corpora and construct contamination-free bench-
                                                         marks to avoid overestimating model capabilities.
• Type 1: Question and answer appear as-is in
  corpus. The question and correct answer appear         4.4    Benchmark Contamination Bulletin
  in the corpus in the exact format as in bench-
  mark entry. The model may memorize entirely            Using INFINI - GRAM MINI, we implement a bench-
  from these data instead of performing capabili-        mark contamination monitoring system that tracks
  ties examined in benchmarks. This accounts for         benchmark contamination. In the future, we will
  72.5% of all dirty entries in Pile-train, 82.6% on     keep indexing the latest crawl in Common Crawl
  DCLM-baseline, and 58% on CC-2025-05.                  and update contamination results to track bench-
• Type 2: Question appeared, answer in natural           mark contamination as corpora evolve. The system
  language. The answer is expressed in natural           also allows anyone to add or upload new bench-
  language rather than the exact benchmark for-          marks to be monitored, fostering collaboration in
  mat, and model may directly infer the correct an-      benchmark monitoring. See App. §J for system
  swer from it. This accounts for 4.5% of all dirty      interface.
  entries in Pile-train, 2.1% on DCLM-baseline,
                                                         5     Related Work
  and 6.2% on CC-2025-05.
• Type 3: Question appeared, no corresponding            Exact-match search in large text corpora. Prior
  answer. The question (and maybe the multiple-          work has used different techniques to enable exact-
  choice options) appeared in the corpus, but            match full-text search in large text corpora, includ-
  correct answer is missing. This accounts for           ing suffix arrays, suffix automata, and proprietary
  18.1% of all dirty entries in Pile-train, 10.9% on     search engines. Below we survey the largest-scale
  DCLM-baseline, and 30.2% on CC-2025-05.                implementation of each technique known to us.
• Type 4: False positives. There are documents           Merrill et al. (2024) apply a suffix automaton to
  that superficially match the question but are un-      1.3TB of text. Liu et al. (2024) apply a suffix array
  related to the benchmark example. This happens         to 12TB of text. Elazar et al. (2024) use the pro-
  on entries with a very short question. This ac-        prietary ElasticSearch to index and analyze 35TB
  counts for 3.1% of all dirty entries in Pile-train,    of text. All these methods have significant storage
                    Test     Pile    DCLM          CC          CC        CC        CC         CC         CC
                    Size    train   baseline   2025-05     2025-08   2025-13   2025-18    2025-21    2025-26
                                         Knowledge and Reasoning
 MMLU              1000    13.20      28.40    13.50     9.00    12.10           11.50      11.70       9.20
 MMLU-Pro          1000     5.50      16.20     7.10     5.40     6.00            6.30       7.40       6.90
 BigBenchHard      1000     0.00       0.10     1.40     1.40     3.20            2.30       1.80       1.70
 AGIEval           1000     0.80       3.10     2.70     3.60     3.00            7.00       9.40       4.60
 GPQA               448     0.00       0.00     0.90     2.00     1.30            0.70       0.90       2.70
 HLE                881     0.00       0.30     0.10     0.00     0.10            0.00       0.00       0.00
                                                    Math
 AIME-2024           30     0.00       0.00      10.00       3.30       6.70     40.00      40.00      13.30
 GSM8K             1000     0.00       5.00       5.00       0.80       6.90      0.70      74.20       7.30
 MATH-500           500     0.60       3.20       0.60       7.80       0.80      0.80       0.80       8.20
 MGSM               250     0.00       0.00       5.60       1.60      35.60      0.80      72.80       6.00
                                                    Code
 HumanEval          164     0.00       0.00       0.00       0.60       0.60      0.60       0.00       0.00
 HumanEval+         164     0.00       0.00       0.00       0.60       0.60      0.60       0.00       0.00
 LiveCodeBench      880     0.00       0.00       0.00       0.00       0.00      0.00       0.00       0.00
 SWE-bench          500     0.00       0.00       0.20       0.20       0.00      0.00       0.00       0.00
 MBPP               500     0.00       0.40       1.00       1.40       1.20      1.80       1.00       1.40
                                       Commonsense Understanding
 ARC-Challenge     1000     1.80      34.10  11.90     4.00       3.10            3.80       4.20       4.80
 ARC-Easy          1000     1.30      31.70   5.40     9.50       5.50            5.50       6.10       6.20
 CSQA              1000     0.10       1.00   0.10     0.10       0.20            0.10       0.00       0.10
 HellaSwag         1000     0.00       0.00   0.00     0.00       0.00            0.00       0.00       0.10
 OpenbookQA         500    10.80      15.60  14.60     30.20     13.20           13.40      13.20      12.20
 Social IQa        1000     0.00       0.50   0.20     4.40       0.20            0.30       0.20       0.10
 WinoGrande        1000     0.00       0.00   0.00     0.00       0.00            0.00       0.00       0.00
                                          Reading Comprehension
 CoQA               500     8.00      18.40      7.40    8.80           8.60      7.20       7.60       8.80
 SQuAD             1000     2.80      40.10      2.70   33.00          10.10      1.50       2.00       8.50

Table 2: Dirty rates for benchmarks across the Pile, DCLM-baseline, and Common Crawl from January to July,
2025. Full result is reported in Table 7. For benchmark entries with over 1000 entries, we report dirty rate on
the downsampled subset. Cell background color indicates benchmark cleaniness, with more redness representing
increasing levels of contamination.


multiplier wrt the size of text indexed, ranging from    full-text exact match (Elazar et al., 2024). Sainz
ElasticSearch’s 2× to suffix automata’s 29×. In          et al. (2024) reports data contamination from multi-
contrast, our FM-index-based index has a storage         ple sources through shared efforts. However, index-
multiplier as small as 0.26×, allowing us to index       ing large-scale pretraining corpora and performing
the largest body of text ever in the open-source         exhaustive searches is computationally expensive,
community.                                               and prior studies on open corpora are limited in
                                                         their scale (up to RedPajama-1T and Dolma, 12TB).
Benchmark contamination. Benchmark con-                  To the best of our knowledge, our work conducts
tamination appeared as a critical concern in LLM         contamination analysis on the largest open corpora
evaluations in recent studies. Prior works has quan-     to date.
tified benchmark contamination on open-sourced
models using various matching strategies, includ-        6    Conclusion
ing n-gram or token overlap (Soldaini et al., 2024;
Llama Team, 2024; Soldaini et al., 2024; OLMo            We introduce INFINI - GRAM MINI, an efficient sys-
et al., 2025), longest substring match (Singh et al.,    tem for indexing text with 0.44× their original size,
2024), skipgram match (Touvron et al., 2023), and        enabling efficient counting and searching in mas-
Figure 3: Examples of four contamination types. Violet text is the text overlap between benchmark entry and corpus.
Magenta text is the mapping of answers.


sive text corpus, and we show its scalability to a         its position in original text takes very long time
petabyte-scale corpus. We showcase INFINI - GRAM           thus making this operation impractical. In contrast,
MINI ’s application on benchmark contamination             the original infini-gram supports this operation effi-
analysis at scale.                                         ciently.
                                                              I NFINI - GRAM MINI only supports exact-match
Limitations                                                searches. As a result, our benchmark contami-
                                                           nation analysis is limited to case-sensitive exact
Although INFINI - GRAM MINI shows significant im-          matching, which may fail to detect contamination
provements on text compression rate, document              of instances with minor textual discrepancies.
retrieval latency remains higher than a canoni-
                                                              As the text corpora may contain biased or toxic
cal suffix array. For example, reconstructing a
                                                           content, document retrieval output may contain
3000-character document takes 1.8 seconds on Pile
                                                           content that can be perceived as ethically problem-
and 4.5 seconds on DCLM-baseline, compared to
                                                           atic, and may contain sensitive information. The
millisecond-level retrieval in system like Infini-
                                                           output of INFINI - GRAM MINI does not reflect au-
gram (Liu et al., 2024). This is a trade-off with
                                                           thors’ views.
INFINI - GRAM MINI ’s compression rate: INFINI -
GRAM MINI does not store the original text in a
contiguous block. To retrieve a document, we need          Acknowledgments
to reconstruct it character-by-character, leading to a
large number of reads in random addresses. These           We thank Christina Boucher (University of Florida)
process takes long time since the entire index is          for telling us about the FM-index data structure.
kept on disk at inference time. The high latency           We would like to thank members of H2lab and
can potentially be reduced using techniques like           AllenNLP for sharing their valuable feedback on
disk page prefetching.                                     this project. This work was funded in part by NSF
   Identifying co-occurrences of multiple patterns         IIS-2044660 and IIS-2113530, and by RS-2024-
is inefficient with the FM-index, since mapping            00457882, National AI Research Lab Project.
every match from the suffix array range back to
References                                               Leo Gao, Stella Biderman, Sid Black, Laurence Gold-
                                                           ing, Travis Hoppe, Charles Foster, Jason Phang,
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten         Horace He, Anish Thite, Noa Nabeshima, Shawn
   Bosma, Henryk Michalewski, David Dohan, Ellen           Presser, and Connor Leahy. 2020. The Pile: An
   Jiang, Carrie Cai, Michael Terry, Quoc Le, and 1        800gb dataset of diverse text for language modeling.
   others. 2021. Program synthesis with large language     arXiv preprint arXiv:2101.00027.
   models. arXiv preprint arXiv:2108.07732.
                                                         Simon Gog, Timo Beller, Alistair Moffat, and Matthias
Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis,
                                                           Petri. 2014. From theory to practice: Plug and play
  Wen tau Yih, Sebastian Riedel, and Fabio Petroni.
                                                           with succinct data structures. In 13th International
  2022. Autoregressive search engines: Generat-
                                                           Symposium on Experimental Algorithms, (SEA 2014),
  ing substrings as document identifiers. Preprint,
                                                           pages 326–337.
  arXiv:2204.10628.
                                                         Li-Lu Guo. 2025. Effloc: An efficient locating algo-
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
                                                            rithm for mass-occurrence biological patterns with
  Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
                                                            fm-index. Journal of computational biology : a jour-
  Neelakantan, Pranav Shyam, Girish Sastry, Amanda
                                                            nal of computational molecular cell biology.
  Askell, Sandhini Agarwal, Ariel Herbert-Voss,
  Gretchen Krueger, Tom Henighan, Rewon Child,           Xiaochuang Han and Yulia Tsvetkov. 2022. Orca: In-
  Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,            terpreting prompted language models via locating
  Clemens Winter, and 12 others. 2020.        Lan-         supporting data evidence in the ocean of pretraining
  guage models are few-shot learners. Preprint,            data. ArXiv, abs/2205.12600.
  arXiv:2005.14165.
                                                         Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Michael Burrows, D J Wheeler D I G I T A L, Robert W.      Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
  Taylor, David J. Wheeler, and David Wheeler. 1994.       2021. Measuring massive multitask language under-
  A block-sorting lossless data compression algorithm.     standing. Preprint, arXiv:2009.03300.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
                                                         Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia
 Henrique Ponde de Oliveira Pinto, Jared Kaplan,
                                                           Yan, Tianjun Zhang, Sida Wang, Armando Solar-
 Harri Edwards, Yuri Burda, Nicholas Joseph, Greg
                                                           Lezama, Koushik Sen, and Ion Stoica. 2024. Live-
 Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
                                                           codebench: Holistic and contamination free evalu-
 Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela
                                                           ation of large language models for code. Preprint,
 Mishkin, Brooke Chan, Scott Gray, and 39 others.
                                                           arXiv:2403.07974.
 2021. Evaluating large language models trained on
 code. Preprint, arXiv:2107.03374.                       Carlos E. Jimenez, John Yang, Alexander Wettig,
                                                           Shunyu Yao, Kexin Pei, Ofir Press, and Karthik
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,
                                                           Narasimhan. 2024. Swe-bench: Can language mod-
  Ashish Sabharwal, Carissa Schoenick, and Oyvind
                                                           els resolve real-world github issues? Preprint,
  Tafjord. 2018. Think you have solved question
                                                           arXiv:2310.06770.
  answering? try arc, the ai2 reasoning challenge.
  arXiv:1803.05457v1.                                    Julian Labeit, Julian Shun, and Guy E. Blelloch. 2017.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,             Parallel lightweight wavelet tree, suffix array and fm-
  Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias            index construction. Journal of Discrete Algorithms,
  Plappert, Jerry Tworek, Jacob Hilton, Reiichiro           43:2–17.
  Nakano, Christopher Hesse, and John Schulman.          Katherine Lee, Daphne Ippolito, Andrew Nystrom,
  2021. Training verifiers to solve math word prob-        Chiyuan Zhang, Douglas Eck, Chris Callison-Burch,
  lems. arXiv preprint arXiv:2110.14168.                   and Nicholas Carlini. 2022. Deduplicating training
Common Crawl Foundation. 2025. Common crawl.               data makes language models better. In Proceedings
                                                           of the 60th Annual Meeting of the Association for
Lore Depuydt, Luca Renders, Thomas Abeel, and Jan          Computational Linguistics (Volume 1: Long Papers),
  Fostier. 2023. Pan-genome de bruijn graph using the      pages 8424–8445, Dublin, Ireland. Association for
  bidirectional fm-index. BMC Bioinformatics, 24(1).       Computational Linguistics.

Yanai Elazar, Akshita Bhagia, Ian Magnusson, Abhi-       Heng Li. 2014. Fast construction of fm-index for long
  lasha Ravichander, Dustin Schwenk, Alane Suhr,           sequence reads. Bioinformatics, 30(22):3274–3275.
  Pete Walsh, Dirk Groeneveld, Luca Soldaini, Sameer
  Singh, Hanna Hajishirzi, Noah A. Smith, and Jesse      Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi,
  Dodge. 2024. What’s in my big data? Preprint,             Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha,
  arXiv:2310.20707.                                         Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin,
                                                            Niklas Muennighoff, Reinhard Heckel, Jean Mercat,
P. Ferragina and G. Manzini. 2000. Opportunistic data       Mayee Chen, Suchin Gururangan, Mitchell Worts-
   structures with applications. In Proceedings 41st        man, Alon Albalak, and 40 others. 2024. Datacomp-
   Annual Symposium on Foundations of Computer Sci-         lm: In search of the next generation of training sets
   ence, pages 390–398.                                     for language models. Preprint, arXiv:2406.11794.
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri          et al. 2025. Humanity’s last exam.         Preprint,
  Edwards, Bowen Baker, Teddy Lee, Jan Leike,                arXiv:2501.14249.
  John Schulman, Ilya Sutskever, and Karl Cobbe.
  2023.    Let’s verify step by step.     Preprint,        Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
  arXiv:2305.20050.                                           Percy Liang. 2016. SQuAD: 100,000+ questions for
                                                              machine comprehension of text. In Proceedings of
Jiacheng Liu, Taylor Blanton, Yanai Elazar, Sewon             the 2016 Conference on Empirical Methods in Natu-
   Min, YenSung Chen, Arnavi Chheda-Kothary, Huy              ral Language Processing, pages 2383–2392, Austin,
   Tran, Byron Bischoff, Eric Stuart Marsh, Michael          Texas. Association for Computational Linguistics.
   Schmitz, Cassidy Trier, Aaron Sarnat, Jenna James,
   Jon Borchardt, Bailey Kuehl, Evie Cheng, Karen Far-     Siva Reddy, Danqi Chen, and Christopher D. Manning.
   ley, Sruthi Sreeram, Taira Anderson, and 12 others.        2019. CoQA: A conversational question answering
   2025. Olmotrace: Tracing language model outputs            challenge. Transactions of the Association for Com-
   back to trillions of training tokens.                      putational Linguistics, 7:249–266.

Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin           David Rein, Betty Li Hou, Asa Cooper Stickland,
   Choi, and Hannaneh Hajishirzi. 2024. Infini-gram:         Jackson Petty, Richard Yuanzhe Pang, Julien Di-
   Scaling unbounded n-gram language models to a tril-       rani, Julian Michael, and Samuel R. Bowman. 2023.
   lion tokens. arXiv preprint arXiv:2401.17377.             Gpqa: A graduate-level google-proof q&a bench-
                                                             mark. Preprint, arXiv:2311.12022.
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Ling-
   ming Zhang. 2023. Is your code generated by chat-       Oscar Sainz, Iker García-Ferrero, Alon Jacovi, Jon An-
   GPT really correct? rigorous evaluation of large lan-     der Campos, Yanai Elazar, Eneko Agirre, Yoav Gold-
   guage models for code generation. In Thirty-seventh       berg, Wei-Lin Chen, Jenny Chim, Leshem Choshen,
   Conference on Neural Information Processing Sys-          Luca D’Amico-Wong, Melissa Dell, Run-Ze Fan,
   tems.                                                     Shahriar Golchin, Yucheng Li, Pengfei Liu, Bhavish
                                                             Pahwa, Ameya Prabhu, Suryansh Sharma, and 9 oth-
AI @ Meta Llama Team. 2024. The llama 3 herd of              ers. 2024. Data contamination report from the 2024
  models. Preprint, arXiv:2407.21783.                        conda shared task. Preprint, arXiv:2407.21530.
Ximing Lu, Melanie Sclar, Skyler Hallinan, Niloofar        Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat-
  Mireshghallah, Jiacheng Liu, Seungju Han, Allyson          ula, and Yejin Choi. 2019. Winogrande: An adver-
  Ettinger, Liwei Jiang, Khyathi Chandu, Nouha Dziri,        sarial winograd schema challenge at scale. Preprint,
  and Yejin Choi. 2024. Ai as humanity’s salieri: Quan-      arXiv:1907.10641.
  tifying linguistic creativity of language models via
  systematic attribution of machine text against web       Maarten Sap, Hannah Rashkin, Derek Chen, Ronan
  text. ArXiv, abs/2410.04265.                              LeBras, and Yejin Choi. 2019. Socialiqa: Common-
                                                            sense reasoning about social interactions. Preprint,
Veli Mäkinen and Gonzalo Navarro. 2005. Succinct            arXiv:1904.09728.
  suffix arrays based on run-length encoding. In Com-
  binatorial Pattern Matching, pages 45–56, Berlin,        Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang,
  Heidelberg. Springer Berlin Heidelberg.                     Suraj Srivats, Soroush Vosoughi, Hyung Won Chung,
                                                             Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan
William Merrill, Noah A. Smith, and Yanai Elazar. 2024.
                                                              Das, and Jason Wei. 2022. Language models are
  Evaluating n-gram novelty of language models using
                                                              multilingual chain-of-thought reasoners. Preprint,
  rusty-dawg. ArXiv, abs/2406.13069.
                                                              arXiv:2210.03057.
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish
  Sabharwal. 2018. Can a suit of armor conduct elec-       Aaditya K. Singh, Muhammed Yusuf Kocyigit, Andrew
  tricity? a new dataset for open book question answer-      Poulton, David Esiobu, Maria Lomeli, Gergely Szil-
  ing. In EMNLP.                                             vasy, and Dieuwke Hupkes. 2024. Evaluation data
                                                             contamination in llms: how do we measure it and
Art of Problem Solving. 2024. Aime problems and              (when) does it matter? Preprint, arXiv:2411.03923.
  solutions.
                                                           Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin
Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groen-            Schwenk, David Atkinson, Russell Authur, Ben
  eveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling        Bogin, Khyathi Chandu, Jennifer Dumas, Yanai
  Gu, Shengyi Huang, Matt Jordan, Nathan Lambert,            Elazar, Valentin Hofmann, Ananya Jha, Sachin Ku-
  Dustin Schwenk, Oyvind Tafjord, Taira Anderson,            mar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Mag-
  David Atkinson, Faeze Brahman, Christopher Clark,          nusson, Jacob Morrison, Niklas Muennighoff, and 17
  Pradeep Dasigi, Nouha Dziri, and 21 others. 2025. 2        others. 2024. Dolma: an open corpus of three trillion
  olmo 2 furious. Preprint, arXiv:2501.00656.                tokens for language model pretraining research. In
                                                             Proceedings of the 62nd Annual Meeting of the As-
Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li,             sociation for Computational Linguistics (Volume 1:
  Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang,            Long Papers), pages 15725–15788, Bangkok, Thai-
  Mohamed Shaaban, John Ling, and Sean Shi                   land. Association for Computational Linguistics.
Mirac Suzgun, Nathan Scales, Nathanael Schärli, Se-
  bastian Gehrmann, Yi Tay, Hyung Won Chung,
  Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny
  Zhou, , and Jason Wei. 2022. Challenging big-bench
  tasks and whether chain-of-thought can solve them.
  arXiv preprint arXiv:2210.09261.
Alon Talmor, Jonathan Herzig, Nicholas Lourie, and
  Jonathan Berant. 2019. CommonsenseQA: A ques-
  tion answering challenge targeting commonsense
  knowledge. In Proceedings of the 2019 Conference
  of the North American Chapter of the Association for
  Computational Linguistics: Human Language Tech-
  nologies, Volume 1 (Long and Short Papers), pages
  4149–4158, Minneapolis, Minnesota. Association for
  Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin R. Stone, Peter
  Albert, Amjad Almahairi, Yasmine Babaei, Niko-
  lay Bashlykov, Soumya Batra, Prajjwal Bhargava,
  Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cris-
  tian Cantón Ferrer, Moya Chen, Guillem Cucurull,
  David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin
  Fu, and 49 others. 2023. Llama 2: Open foundation
  and fine-tuned chat models. ArXiv, abs/2307.09288.
Yuanrong Wang, Xueqi Li, Dawei Zang, Guangming
  Tan, and Ninghui Sun. 2018. Accelerating fm-index
  search for genomic data processing. In Proceedings
  of the 47th International Conference on Parallel Pro-
  cessing, ICPP ’18, New York, NY, USA. Association
  for Computing Machinery.
Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni,
  Abhranil Chandra, Shiguang Guo, Weiming Ren,
  Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max
  Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue,
  and Wenhu Chen. 2024. Mmlu-pro: A more robust
  and challenging multi-task language understanding
  benchmark. Preprint, arXiv:2406.01574.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali
  Farhadi, and Yejin Choi. 2019. Hellaswag: Can a
  machine really finish your sentence? In Proceedings
  of the 57th Annual Meeting of the Association for
  Computational Linguistics.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo
 Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu
 Chen, and Nan Duan. 2023. Agieval: A human-
 centric benchmark for evaluating foundation models.
 Preprint, arXiv:2304.06364.
A    Querying the FM-Index                              structing alphabet, wavelet tree, and sampling are
                                                        linear operations. Indexing time also varies across
This section introduces in details three operations     similar-size shards originated from different text
supported by FM-index: find, locate, and recon-         corpora, showing the time also depend on text cor-
struct.                                                 pora properties. For example, over 60% duplicate
Find. Each occurrence of a pattern string Q in          documents in the Pile (Elazar et al., 2024) causes
the haystack string T corresponds to an element in      SA construction to take significantly longer com-
the SA, and all these occurrences live in a consecu-    pared to DCLM-baseline.
tive range in the SA. The find operation computes
                                                        C   Benchmarking Query Latency
this range with backward search: starting with the
full range, it iterates through the symbols in Q in     We benchmark the query latency of INFINI -
reverse order, with each iteration narrowing the        GRAM MINI , and report results in Table 4. For
range using the character table C and the rank oper-    counting, we experiment with query lengths
ation of the BWT’s wavelet tree. The length of the      |Q| ∈ {1, 2, 5, 10, 50, 100, 500, 1000}, and for
final range is the count of Q in T . Figure 4 (left)    each length, randomly sample the query strings
illustrates finding pattern “ana” in string “banana”.   from the corpus. For retrieving documents, we
The time complexity of find is O(|Q|H0 ), where         use queries of length 10 sampled from the cor-
|Q| is the length of the pattern.                       pus, and reconstruct text surrounding the query
                                                        with total length of d ∈ {10, 50, 100, 500, 1000}.
Locate. The locate operation maps any position
                                                        For each corpus and parameter setting, we repeat
in the SA to its corresponding position in T . Since
                                                        with 100 random queries and report the average
the SA is sampled at interval a, for an unsampled
                                                        latency. For benchmarking, we store the indexes
position i, the algorithm applies LF mapping at
                                                        on pd-balanced SSD disks on GCP which has a
most a times to locate the nearest sampled SA en-
                                                        max IOPS of 80,000 and max throughput of 1200
try. Figure 4 (middle) illustrates locating the two
                                                        MB/s, and we experiment on an n2-highcpu-64
occurrences of “ana” in “banana”. The time com-
                                                        node where the max disk I/O performance can be
plexity of locating each position is O(tSA ), where
                                                        achieved.
tSA is the complexity of accessing an SA entry,
which is O(1) for sampled entries and O(aH0 ) for       D   Query Latency Comparison with Prior
unsampled entries.                                          Works
Reconstruct. After getting the position of a pat-       Table 5 reports query latency on Pile-train corpus
tern occurrence in T , we can reconstruct a sub-        using our method compared to prior work by Liu
string of T enclosing that occurrence with some         et al. (2024).
additional context. Starting at the end position of
the desired substring, we apply the LF mapping to       E   I NFINI - GRAM MINI Web Interface
traverse through the BWT and use ISA to recover         We host web interface for easy access to counting
the symbol, and reconstruct symbols in reverse          (Figure 5 left) and document retrieval (Figure 5
order until reaching the start position of the de-      right) with the indexes we have built.
sired substring. Figure 4 (right) shows reconstruct-
ing the second occurrence of “ana” with context         F   Details for Benchmarks in
length of 1. The time complexity of reconstruct             Contamination Analysis
is O(dH0 + tISA ), where d is the length to recon-
struct, and tISA is the complexity of accessing an      Table 6 shows the benchmarks we analyze under
ISA entry, which is O(1) for sampled entries and        each category with citation and source.
O(bH0 ) for unsampled entries.
                                                        G   Detailed Benchmark Contamination
B   Indexing Time                                           Result
                                                        Table 7 reports the number of suspicious and dirty
Table 3 presents indexing time for each shard of
                                                        entries of each benchmark.
text corpora with stepwise breakdown. Suffix ar-
ray construction scales super-linearly depending
on the level of duplication (Lee et al., 2022). Con-
Figure 4: Illustration of operations on FM-index (§2.2, App. §A). Left: find operation computes the SA range
corresponding to all occurrences of the pattern. Middle: locate operation computes the position of pattern occurrence
in the original string for each position in the SA range. Right: reconstruct operation gets a substring of the original
string enclosing the second pattern occurrence with a context length of 1. The occurrence ranking is based on its
order in SA.


                  Step              Pile-val    Pile-train     DCLM-baseline          CC-2025-05
                                   (1.4 GB)     (653 GB)           (667 GB)             (654 GB)
                  SA+BWT                29 s       41710 s                29543 s          55692 s
                  alphabet                4s        2584 s                 2895 s            2580 s
                  wavelet tree          13 s        5773 s                 6257 s            5325 s
                  sample SA               1s        2540 s                 2232 s            2013 s
                  sample ISA              2s        3975 s                 2659 s            2313 s
                  Total                 49 s         15.7 h                12.1 h            18.9 h

Table 3: Index construction time for each shard of the text corpora, with stepwise breakdown. The size of text in
each shard is noted at the top. To get the indexing time of the full corpus on a single node, roughly multiply by
the number of shards; though this can be embarrassingly parallelized across multiple nodes. Metadata size and
metadata indexing time are excluded.




    Figure 5: The web interface of INFINI - GRAM MINI. Left: counting a string. Right: retrieving documents.
                                        Pile-train    DCLM-baseline       CC-2025-05           Time
                                        (n = 1.3T)     (n = 16.7T)         (n = 9.1T)        Complexity
                                          (S = 2)        (S = 25)           (S = 15)
 Counting a query of length |Q|                                                               O(|Q|H0 )
   . . . (|Q| = 1)                          0.004 s        0.005 s            0.032 s
   . . . (|Q| = 2)                          0.015 s        0.017 s            0.094 s
   . . . (|Q| = 5)                          0.061 s        0.207 s            0.206 s
   . . . (|Q| = 10)                         0.106 s        0.402 s            0.350 s
   . . . (|Q| = 20)                         0.182 s        0.868 s            0.638 s
   . . . (|Q| = 50)                         0.393 s        1.743 s            1.063 s
   . . . (|Q| = 100)                        0.696 s        2.857 s            1.642 s
   . . . (|Q| = 200)                        1.281 s        5.699 s            2.753 s
   . . . (|Q| = 500)                        2.763 s        12.46 s            4.626 s
   . . . (|Q| = 1000)                       4.808 s        25.47 s            7.957 s
 Retrieving a text of length d                                                            O((a + b + d)H0 )
   . . . (d = 10)                           0.426 s        0.895 s            1.101 s
   . . . (d = 50)                           0.634 s        1.549 s            1.302 s
   . . . (d = 100)                          0.734 s        1.991 s            1.326 s
   . . . (d = 500)                          0.874 s        2.363 s            1.609 s
   . . . (d = 1000)                          0.94 s        2.385 s            1.705 s
   . . . (d = 2000)                         1.213 s        3.464 s            2.849 s
   . . . (d = 3000)                         1.858 s        4.456 s            3.330 s

Table 4: Inference time latency of INFINI - GRAM MINI. Average latency of each query is reported. Notations: n =
number of bytes in the text corpus, S = number of shards for the index, |Q| = length of query in bytes, d =
length of text (in bytes) to reconstruct from the index, a = sampling rate of SA, b = sampling rate of ISA, H0 =
zeroth-order entropy of the corpus.




                                              |Q| = 10 bytes    |Q| = 20 bytes     |Q| = 100 bytes
           infini-gram (Liu et al., 2024)         13 ms               14 ms             13 ms
           INFINI - GRAM MINI                     106 ms             182 ms             696 ms

 Table 5: Text retrieving latency comparison between infini-gram and INFINI - GRAM MINI on Pile-train corpus.
Benchmark                               Citation     Source
                                              Knowledge and Reasoning
MMLU                      Hendrycks et al., 2021     https://huggingface.co/datasets/cais/mmlu
MMLU-Pro                      Wang et al., 2024      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
BigBenchHard                Suzgun et al., 2022      https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh
AGIEval                      Zhong et al., 2023      https://github.com/ruixiangcui/AGIEval/tree/main/data/v1_1
GPQA                           Rein et al., 2023     https://huggingface.co/datasets/Idavidrein/gpqa
HLE                           Phan et al., 2025      https://huggingface.co/datasets/cais/hle
                                                         Math
AIME-2024              of Problem Solving, 2024      https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
GSM8K                         Cobbe et al., 2021     https://huggingface.co/datasets/openai/gsm8k
MATH-500                   Lightman et al., 2023     https://huggingface.co/datasets/HuggingFaceH4/MATH-500
MGSM                             Shi et al., 2022    https://huggingface.co/datasets/juletxara/mgsm
                                                         Code
HumanEval                       Chen et al., 2021    https://huggingface.co/datasets/openai/openai_humaneval
HumanEval+                        Liu et al., 2023   https://huggingface.co/datasets/evalplus/humanevalplus
LiveCodeBench                    Jain et al., 2024   https://huggingface.co/datasets/livecodebench/code_generation
  (code generation)
SWE-bench                    Jimenez et al., 2024    https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
MBPP                           Austin et al., 2021   https://huggingface.co/datasets/google-research-datasets/mbpp
                                            Commonsense Understanding
ARC-Challenge                  Clark et al., 2018    https://huggingface.co/datasets/allenai/ai2_arc
ARC-Easy                       Clark et al., 2018    https://huggingface.co/datasets/allenai/ai2_arc
CSQA                          Talmor et al., 2019    https://huggingface.co/datasets/tau/commonsense_qa
HellaSwag                     Zellers et al., 2019   https://huggingface.co/datasets/Rowan/hellaswag
Openbook QA                 Mihaylov et al., 2018    https://huggingface.co/datasets/allenai/openbookqa
Social IQa                       Sap et al., 2019    https://huggingface.co/datasets/allenai/social_i_qa
WinoGrande                 Sakaguchi et al., 2019    https://huggingface.co/datasets/allenai/winogrande
                                               Reading Comprehension
CoQA                          Reddy et al., 2019     https://huggingface.co/datasets/stanfordnlp/coqa
SQuAD                      Rajpurkar et al., 2016    https://huggingface.co/datasets/rajpurkar/squad

                      Table 6: Citation and source of each benchmark analyzed in this paper.
Benchmark        Test Set    Pile-train   DCLM-baseline      CC-2025-05    CC-202508     CC-202513     CC-202518     CC-202521     CC-202526
                     Size   Sus   Dirty   Sus       Dirty    Sus   Dirty   Sus   Dirty   Sus   Dirty   Sus   Dirty   Sus   Dirty   Sus   Dirty
                                                         Knowledge and Reasoning
MMLU                1000     57     133   142         277    122    135    139      90   121    122    115    120    117    125     92    124
MMLU-Pro            1000     35      59    77         162     31     81     55      54    60     52     63     58     74     60     69     40
BigBenchHard        1000      7       0    54           1    357     14    296      14   306     32    302     23    396     18    338     17
AGIEval             1000     45       8   155          31    107     27     92      36   115     30    109     70     95     94    104     46
GPQA                 448      1       0     5           0      6      4     10       9     5      6      6      3      5      4     15     12
HLE                  881      3       0     6           0      8      0      3       0     4      1      5      0      3      0      2      0
                                                                    Math
AIME-2024             30      2       0     4           0     13      3      3       1    10      2     13     12     13     12     12      4
GSM8K               1000      1       0     5           4     33     76      8       8    37     69     10      7    210    742     20     73
MATH-500             500      9       3    48          16     26      3     46      39    25      4     27      4     26      4     41     41
MGSM                 250      0       0     2           0      9     14      3       4    37     89      5      2     54    182      6     15
                                                                    Code
HumanEval             164     1       0     1           0     14      0     60      1     53      1     55      1     65      0     77      0
HumanEval+            164     1       0     1           0     14      0     60      1     53      1     55      1     65      0     77      0
LiveCodeBench         880     2       0     2           0    188      0    241      0    221      0    274      0    167      0    168      0
SWE-bench             500     5       0    26           0     29      1     38      1     32      0     34      2     33      0     28      0
MBPP                  500    16       0    64           2     87      5     58      7     89      6     75      9     76      5     84      7
                                                       Commonsense Understanding
ARC-Challenge       1000      6      14     4         341     18    119     29     40     27     31     26     38     28     42     29     48
ARC-Easy            1000     15      15     5         317     31     54     30     95     31     55     37     55     28     61     31     62
CSQA                1000      0       1     1          10      1      1      0      1      0      2      0      1      0      0      0      1
HellaSwag           1000     27       0   341           0     36      0     29      0     34      0     29      0     30      0     28      1
OpenbookQA           500      2      54     3          78      5     73      2    151      2     66      3     67      3     66      2     61
Social IQa          1000      0       0     2           5      2      2      1     44      2      2      2      3      3      2      0      1
WinoGrande          1000      2       0     0           0      0      0      0      0      0      0      0      0      0      0      0      0
                                                            Reading Comprehension
CoQA                 500    189      40   188          92    121     37    113     44    123     43    134     36    123     38    125     44
SQuAD               1000     16      28     9         401     11     27     12    330     15    101     13     15     12     20     11     85

        Table 7: Detailed benchmark contamination results. For test sets with more than 1000 entries, we randomly downsample to 1000.
H    Contamination Examples
Figure 6 to Figure 12 shows example dirty entries
in seven benchmarks and its contamination source
retrieved from one of three corpus using INFINI -
GRAM MINI. We also present the original webpage
that is responsible for the contamination.

I   Dirty Entry Categorization using
    LLM-as-a-judge
We use gpt-4o-mini to categorize all dirty entries
into one of four categories described in §4.3. For
each dirty entry, we extract the 50-character sub-
string that has least occurrence in the corpus. We
then retrieve text snippet with context length of
400 characters (850 characters in total). We prompt
gpt-4o-mini by providing both the dirty entry and
the text snippet and let the model decide which cat-
egory the dirty entry belongs. We use the following
prompt:
   You are an expert in evaluating benchmark
contamination. Given an entry and detected
overlap in the corpus, categorize how the en-
try is contaminated in corpus into one of four
categories:
1. You can find exact match of question stem
and the correct answer (correct choice if mul-
tiple choices, or answer matching exactly the
answer field) in corpus.
2. You can find exact match of question stem,
and the correct answer appears, though not in
exact match.
3. You can find exact match of question stem,
but you cannot find the correct answer in any
form.
4. False positive
Output only the category number.

J   Interface of the Benchmark
    Contamination Monitoring System
Figure 13 shows the interface of the system. It
consists of two tabs: (1) Benchmark Contamination
Bulletin, and (2) submission page for community to
contribute. Benchmarks analyzed in this paper are
reported in “core” table, and submitted benchmarks
will be added to “community” table.

K    License of Corpora Used in the Paper
Pile and DCLM are licensed under the MIT Li-
cense. Common Crawl is licensed under its cus-
tomized Limited License. We followed the listed
intended use.
(a) Left upper: An entry in the AIME-2024 benchmark. Right: A document contaminating this entry, retrieved from CC-2025-
05 by INFINI - GRAM MINI. This example belongs to Category 1, where the correct answer can be found in the document.




                               (b) The original webpage responsible for the contamination.

Figure 6: AIME dirty entry example in CC-2025-05. The contamination source is the official AOPS website, where
AIME exams are published.
(a) Left upper: An entry in the MMLU benchmark. Right: A document contaminating this entry, retrieved from CC-2025-05
by INFINI - GRAM MINI. This example belongs to Category 1, where the correct answer presents (though choices are not in the
exact same order).




                                (b) The original webpage responsible for the contamination.

Figure 7: MMLU dirty entry example in CC-2025-05. The contamination source is a website containing multiple-
choice question in related fields.
(a) Left upper: An entry in the SWE-bench benchmark. Right: A document contaminating this entry, retrieved from CC-2025-
05 by INFINI - GRAM MINI. This example is Category 3, where only question appears but not the answer.




                               (b) The original webpage responsible for the contamination.

Figure 8: SWE-bench dirty entry example in CC-2025-05. The contamination source is a website recording pull
requests for software developing.
(a) Left bottom: An entry in the GPQA benchmark. Right: A document contaminating this entry, retrieved from CC-2025-05
by INFINI - GRAM MINI. This example belongs to Category 3, where only the question appears, not the answers.




                              (b) The original webpage responsible for the contamination.

Figure 9: GPQA dirty entry example in CC-2025-05. The contamination source is a blog post citing a test set
example.
(a) Left bottom: An entry in the GSM8K benchmark. Right: A document contaminating this entry, retrieved from CC-2025-05
by INFINI - GRAM MINI. This example is Category 3 because the correct answer is not present.




                               (b) The original webpage responsible for the contamination.

Figure 10: GSM8K dirty entry example in CC-2025-05. The contamination source is a HuggingFace dataset
sourcing from GSM8K examples, and is the major source for GSM8K dirty entries. This new dataset contains
erroneous “steps” field and final answer to examine LLM’s ability to identify errors, so the correct answers do not
appear.
(a) Left bottom: An entry in the BigBenchHard benchmark. Right: A document contaminating this entry, retrieved from
CC-2025-05 by INFINI - GRAM MINI. This example is Category 3 because the correct answer is not present.




                              (b) The original webpage responsible for the contamination.

Figure 11: BigBenchHard dirty entry example in CC-2025-05. The contamination source is a HuggingFace commit
history that list it as few-shot example.
(a) Left bottom: An entry in the OpenbookQA benchmark. Right: A document contaminating this entry, retrieved from
Pile-train by INFINI - GRAM MINI. This example is Category 2 where the correct answer appears in natural language.




                             (b) The original webpage responsible for the contamination.

Figure 12: OpenbookQA dirty entry example in Pile-train. The contamination source is a paper citing an example
from OpenbookQA benchmark.
Figure 13: Upper: A screenshot of our online Benchmark Contamination Bulletin. Lower: We invite the community
to add new benchmarks or upload new ones for contamination analysis, which will be added to the bulletin.
