# paper10 vggt

> **Original PDF Size:** 3.1M
> **Text Size:** 132K
> **Conversion Date:** 2026-01-22 07:34:31

---

                                                                    VGGT: Visual Geometry Grounded Transformer

                                                   Jianyuan Wang1,2                Minghao Chen1,2                Nikita Karaev1,2               Andrea Vedaldi1,2
                                                                              Christian Rupprecht1                      David Novotny2
                                                              1                                                                                 2
                                                                  Visual Geometry Group, University of Oxford                                       Meta AI
arXiv:2503.11651v1 [cs.CV] 14 Mar 2025

                                               …




                                         Figure 1. VGGT is a large feed-forward transformer with minimal 3D-inductive biases trained on a trove of 3D-annotated data. It accepts
                                         up to hundreds of images and predicts cameras, point maps, depth maps, and point tracks for all images at once in less than a second, which
                                         often outperforms optimization-based alternatives without further processing.

                                                                    Abstract                                     1. Introduction
                                                                                                                 We consider the problem of estimating the 3D attributes
                                         We present VGGT, a feed-forward neural network that di-                 of a scene, captured in a set of images, utilizing a feed-
                                         rectly infers all key 3D attributes of a scene, including cam-          forward neural network. Traditionally, 3D reconstruction
                                         era parameters, point maps, depth maps, and 3D point                    has been approached with visual-geometry methods, uti-
                                         tracks, from one, a few, or hundreds of its views. This                 lizing iterative optimization techniques like Bundle Adjust-
                                         approach is a step forward in 3D computer vision, where                 ment (BA) [45]. Machine learning has often played an im-
                                         models have typically been constrained to and special-                  portant complementary role, addressing tasks that cannot
                                         ized for single tasks. It is also simple and efficient, re-             be solved by geometry alone, such as feature matching and
                                         constructing images in under one second, and still out-                 monocular depth prediction. The integration has become
                                         performing alternatives that require post-processing with               increasingly tight, and now state-of-the-art Structure-from-
                                         visual geometry optimization techniques. The network                    Motion (SfM) methods like VGGSfM [125] combine ma-
                                         achieves state-of-the-art results in multiple 3D tasks, in-             chine learning and visual geometry end-to-end via differ-
                                         cluding camera parameter estimation, multi-view depth es-               entiable BA. Even so, visual geometry still plays a major
                                         timation, dense point cloud reconstruction, and 3D point                role in 3D reconstruction, which increases complexity and
                                         tracking. We also show that using pretrained VGGT as a                  computational cost.
                                         feature backbone significantly enhances downstream tasks,                   As networks become ever more powerful, we ask if,
                                         such as non-rigid point tracking and feed-forward novel                 finally, 3D tasks can be solved directly by a neural net-
                                         view synthesis. Code and models are publicly available at               work, eschewing geometry post-processing almost entirely.
                                         https://github.com/facebookresearch/vggt.                               Recent contributions like DUSt3R [129] and its evolution


                                                                                                             1
MASt3R [62] have shown promising results in this direc-                VGGT achieves state-of-the-art results across the board,
tion, but these networks can only process two images at                even when compared to methods that specialize in a sub-
once and rely on post-processing to reconstruct more im-               set of 3D tasks, often improving quality substantially.
ages, fusing pairwise reconstructions.                                      We make our code and models publicly available at
    In this paper, we take a further step towards removing the         https://github.com/facebookresearch/vggt. We believe that
need to optimize 3D geometry in post-processing. We do                 this will facilitate further research in this direction and bene-
so by introducing Visual Geometry Grounded Transformer                 fit the computer vision community by providing a new foun-
(VGGT), a feed-forward neural network that performs 3D                 dation for fast, reliable, and versatile 3D reconstruction.
reconstruction from one, a few, or even hundreds of input
views of a scene. VGGT predicts a full set of 3D attributes,           2. Related Work
including camera parameters, depth maps, point maps, and
                                                                       Structure from Motion is a classic computer vision prob-
3D point tracks. It does so in a single forward pass, in sec-
                                                                       lem [45, 77, 80] that involves estimating camera parameters
onds. Remarkably, it often outperforms optimization-based
                                                                       and reconstructing sparse point clouds from a set of images
alternatives even without further processing. This is a sub-
                                                                       of a static scene captured from different viewpoints. The
stantial departure from DUSt3R, MASt3R, or VGGSfM,
                                                                       traditional SfM pipeline [2, 36, 70, 94, 103, 134] consists
which still require costly iterative post-optimization to ob-
                                                                       of multiple stages, including image matching, triangulation,
tain usable results.
                                                                       and bundle adjustment. COLMAP [94] is the most popu-
    We also show that it is unnecessary to design a special
                                                                       lar framework based on the traditional pipeline. In recent
network for 3D reconstruction. Instead, VGGT is based
                                                                       years, deep learning has improved many components of the
on a fairly standard large transformer [119], with no par-
                                                                       SfM pipeline, with keypoint detection [21, 31, 116, 149]
ticular 3D or other inductive biases (except for alternating
                                                                       and image matching [11, 67, 92, 99] being two primary ar-
between frame-wise and global attention), but trained on a
                                                                       eas of focus. Recent methods [5, 102, 109, 112, 113, 118,
large number of publicly available datasets with 3D annota-
                                                                       122, 125, 131, 160] explored end-to-end differentiable SfM,
tions. VGGT is thus built in the same mold as large models
                                                                       where VGGSfM [125] started to outperform traditional al-
for natural language processing and computer vision, such
                                                                       gorithms on challenging phototourism scenarios.
as GPTs [1, 29, 148], CLIP [86], DINO [10, 78], and Stable
Diffusion [34]. These have emerged as versatile backbones              Multi-view Stereo aims to densely reconstruct the geome-
that can be fine-tuned to solve new, specific tasks. Simi-             try of a scene from multiple overlapping images, typically
larly, we show that the features computed by VGGT can                  assuming known camera parameters, which are often esti-
significantly enhance downstream tasks like point tracking             mated with SfM. MVS methods can be divided into three
in dynamic videos, and novel view synthesis.                           categories: traditional handcrafted [38, 39, 96, 130], global
    There are several recent examples of large 3D neural net-          optimization [37, 74, 133, 147], and learning-based meth-
works, including DepthAnything [142], MoGe [128], and                  ods [42, 72, 84, 145, 157]. As in SfM, learning-based
LRM [49]. However, these models only focus on a sin-                   MVS approaches have recently seen a lot of progress. Here,
gle 3D task, such as monocular depth estimation or novel               DUSt3R [129] and MASt3R [62] directly estimate aligned
view synthesis. In contrast, VGGT uses a shared backbone               dense point clouds from a pair of views, similar to MVS
to predict all 3D quantities of interest together. We demon-           but without requiring camera parameters. Some concurrent
strate that learning to predict these interrelated 3D attributes       works [111, 127, 141, 156] explore replacing DUSt3R’s
enhances overall accuracy despite potential redundancies.              test-time optimization with neural networks, though these
At the same time, we show that, during inference, we can               attempts achieve only suboptimal or comparable perfor-
derive the point maps from separately predicted depth and              mance to DUSt3R. Instead, VGGT outperforms DUSt3R
camera parameters, obtaining better accuracy compared to               and MASt3R by a large margin.
directly using the dedicated point map head.                           Tracking-Any-Point was first introduced in Particle
    To summarize, we make the following contributions: (1)             Video [91] and revived by PIPs [44] during the deep learn-
We introduce VGGT, a large feed-forward transformer that,              ing era, aiming to track points of interest across video se-
given one, a few, or even hundreds of images of a scene,               quences including dynamic motions. Given a video and
can predict all its key 3D attributes, including camera intrin-        some 2D query points, the task is to predict 2D correspon-
sics and extrinsics, point maps, depth maps, and 3D point              dences of these points in all other frames. TAP-Vid [23]
tracks, in seconds. (2) We demonstrate that VGGT’s pre-                proposed three benchmarks for this task and a simple base-
dictions are directly usable, being highly competitive and             line method later improved in TAPIR [24]. CoTracker [55,
usually better than those of state-of-the-art methods that use         56] utilized correlations between different points to track
slow post-processing optimization techniques. (3) We also              through occlusions, while DOT [60] enabled dense track-
show that, when further combined with BA post-processing,              ing through occlusions. Recently, TAPTR [63] proposed

                                                                   2
                                   Add            Global        Frame                           Camera Head            Cameras
                                camera token     Attention     Attention
     Input


                                                                                                                       Depth maps



               DINO        Concat


                                                                                                   DPT                 Point maps


                                                                                                                       Tracks


                                                        × 𝐿 times


Figure 2. Architecture Overview. Our model first patchifies the input images into tokens by DINO, and appends camera tokens for camera
prediction. It then alternates between frame-wise and global self attention layers. A camera head makes the final prediction for camera
extrinsics and intrinsics, and a DPT [87] head for any dense output.


an end-to-end transformer for this task, and LocoTrack [13]                 We denote the domain of the image Ii with I(Ii ) =
extended commonly used pointwise features to nearby re-                 {1, . . . , H} × {1, . . . , W }, i.e., the set of pixel locations.
gions. All of these methods are specialized point trackers.             The depth map Di associates each pixel location y ∈ I(Ii )
Here, we demonstrate that VGGT’s features yield state-of-               with its corresponding depth value Di (y) ∈ R+ , as ob-
the-art tracking performance when coupled with existing                 served from the i-th camera. Likewise, the point map Pi
point trackers.                                                         associates each pixel with its corresponding 3D scene point
                                                                        Pi (y) ∈ R3 . Importantly, like in DUSt3R [129], the point
3. Method                                                               maps are viewpoint invariant, meaning that the 3D points
                                                                        Pi (y) are defined in the coordinate system of the first cam-
We introduce VGGT, a large transformer that ingests a set               era g1 , which we take as the world reference frame.
of images as input and produces a variety of 3D quantities
                                                                            Finally, for keypoint tracking, we follow track-any-
as output. We start by introducing the problem in Sec. 3.1,
                                                                        point methods such as [25, 57]. Namely, given a fixed query
followed by our architecture in Sec. 3.2 and its prediction
                                                                        image point yq in the query image Iq , the network outputs
heads in Sec. 3.3, and finally the training setup in Sec. 3.4.
                                                                        a track T ⋆ (yq ) = (yi )N  i=1 formed by the corresponding 2D
3.1. Problem definition and notation                                    points yi ∈ R2 in all images Ii .
                                                                            Note that the transformer f above does not output the
The input is a sequence (Ii )N
                             i=1 of N RGB images Ii ∈
                                                                        tracks directly but instead features Ti ∈ RC×H×W , which
R3×H×W , observing the same 3D scene. VGGT’s trans-
                                                                        are used for tracking. The tracking is delegated to a sep-
former is a function that maps this sequence to a corre-
                                                                        arate module, described in Sec. 3.3, which implements a
sponding set of 3D annotations, one per frame:
                                                                        function T ((yj )M             N                N   M
                                                                                          j=1 , (Ti )i=1 ) = ((ŷj,i )i=1 )j=1 . It ingests
            f (Ii )N
                                            N                          the query point yq and the dense tracking features Ti output
                   i=1 = (gi , Di , Pi , Ti )i=1 .   (1)
                                                                        by the transformer f and then computes the track. The two
The transformer thus maps each image Ii to its camera pa-               networks f and T are trained jointly end-to-end.
rameters gi ∈ R9 (intrinsics and extrinsics), its depth map             Order of Predictions. The order of the images in the input
Di ∈ RH×W , its point map Pi ∈ R3×H×W , and a grid                      sequence is arbitrary, except that the first image is chosen as
Ti ∈ RC×H×W of C-dimensional features for point track-                  the reference frame. The network architecture is designed
ing. We explain next how these are defined.                             to be permutation equivariant for all but the first frame.
   For the camera parameters gi , we use the parametriza-
tion from [125] and set g = [q, t, f ] which is the concatena-          Over-complete Predictions. Notably, not all quantities
tion of the rotation quaternion q ∈ R4 , the translation vec-           predicted by VGGT are independent. For example, as
tor t ∈ R3 , and the field of view f ∈ R2 . We assume that              shown by DUSt3R [129], the camera parameters g can
the camera’s principal point is at the image center, which is           be inferred from the invariant point map P , for instance,
common in SfM frameworks [95, 125].                                     by solving the Perspective-n-Point (PnP) problem [35, 61].

                                                                    3
           Input                                         Ours                                                DUSt3R
                                                                           < 0.1s                                                   < 0.1s




        Single view

                                                                           < 0.1s                                                   < 0.1s




         Two views


                                                                          < 0.6 s                                                   > 200s
                 …


                 …


          32 views

Figure 3. Qualitative comparison of our predicted 3D points to DUSt3R on in-the-wild images. As shown in the top row, our method
successfully predicts the geometric structure of an oil painting, while DUSt3R predicts a slightly distorted plane. In the second row, our
method correctly recovers a 3D scene from two images with no overlap, while DUSt3R fails. The third row provides a challenging example
with repeated textures, while our prediction is still high-quality. We do not include examples with more than 32 frames, as DUSt3R runs
out of memory beyond this limit.


Furthermore, the depth maps can be deduced from the point               (AA), making the transformer focus within each frame and
map and the camera parameters. However, as we show in                   globally in an alternate fashion. Specifically, frame-wise
Sec. 4.5, tasking VGGT with explicitly predicting all afore-            self-attention attends to the tokens tIk within each frame
mentioned quantities during training brings substantial per-            separately, and global self-attention attends to the tokens
formance gains, even when these are related by closed-form              tI across all frames jointly. This strikes a balance between
relationships. Meanwhile, during inference, it is observed              integrating information across different images and normal-
that combining independently estimated depth maps and                   izing the activations for the tokens within each image. By
camera parameters produces more accurate 3D points com-                 default, we employ L = 24 layers of global and frame-wise
pared to directly employing a specialized point map branch.             attention. In Sec. 4, we demonstrate that our AA architec-
                                                                        ture brings significant performance gains. Note that our ar-
3.2. Feature Backbone                                                   chitecture does not employ any cross-attention layers, only
Following recent works in 3D deep learning [53, 129, 132],              self-attention ones.
we design a simple architecture with minimal 3D induc-
tive biases, letting the model learn from ample quantities              3.3. Prediction heads
of 3D-annotated data. In particular, we implement the
                                                                        Here, we describe how f predicts the camera parameters,
model f as a large transformer [119]. To this end, each
                                                                        depth maps, point maps, and point tracks. First, for each
input image I is initially patchified into a set of K tokens1
                                                                        input image Ii , we augment the corresponding image to-
tI ∈ RK×C through DINO [78]. The combined set of im-                                                                                   ′
                                                                        kens tIi with an additional camera token tgi ∈ R1×C and
age tokens from all frames, i.e., tI = ∪N        I
                                           i=1 {ti }, is subse-                                                     ′

quently processed through the main network structure, al-               four register tokens [19] tR    i ∈R
                                                                                                                4×C
                                                                                                                      . The concatenation of
                                                                          I g R N
ternating frame-wise and global self-attention layers.                  (ti , ti , ti j)i=1 is then passed to the AA transformer, yield-
                                                                        ing output tokens (t̂Ii , t̂gi , t̂R  N
                                                                                                           i )i=1 . Here, the camera token
Alternating-Attention. We slightly adjust the standard                  and register tokens of the first frame (tg1 := t̄g , tR          R
                                                                                                                                  1 := t̄ )
transformer design by introducing Alternating-Attention                                                                      g R
                                                                        are set to a different set of learnable tokens t , t than those
                                                                                                         g            R
   1 The number of tokens depends on the image resolution.              of all other frames (tgi := t , tR
                                                                                                         i := t , i ∈ [2, . . . , N ]),


                                                                    4
Figure 4. Additional Visualizations of Point Map Estimation. Camera frustums illustrate the estimated camera poses. Explore our
interactive demo for better visualization quality.


which are also learnable. This allows the model to distin-               sics and extrinsics.
guish the first frame from the rest, and to represent the 3D
                                                                         Dense Predictions. The output image tokens t̂Ii are used
predictions in the coordinate frame of the first camera. Note
                                                                         to predict the dense outputs, i.e., the depth maps Di , point
that the refined camera and register tokens now become
                                                                         maps Pi , and tracking features Ti . More specifically, t̂Ii are
frame-specific—–this is because our AA transformer con-                                                                   ′′
                                                                         first converted to dense feature maps Fi ∈ RC ×H×W with
tains frame-wise self-attention layers that allow the trans-
                                                                         a DPT layer [87]. Each Fi is then mapped with a 3 × 3 con-
former to match the camera and register tokens with the cor-
                                                                         volutional layer to the corresponding depth and point maps
responding tokens from the same image. Following com-
                                                                         Di and Pi . Additionally, the DPT head also outputs dense
mon practice, the output register tokens t̂R i are discarded
                                                                         features Ti ∈ RC×H×W , which serve as input to the track-
while t̂Ii , t̂gi are used for prediction.
                                                                         ing head. We also predict the aleatoric uncertainty [58, 76]
                                                                                   H×W                 H×W
Coordinate Frame. As noted above, we predict cameras,                    ΣDi ∈ R+         and ΣP
                                                                                               i ∈ R+         for each depth and point
point maps, and depth maps in the coordinate frame of the                map, respectively. As described in Sec. 3.4, the uncertainty
first camera g1 . As such, the camera extrinsics output for              maps are used in the loss and, after training, are proportional
the first camera are set to the identity, i.e., the first rotation       to the model’s confidence in the predictions.
quaternion is q1 = [0, 0, 0, 1] and the first translation vector         Tracking. In order to implement the tracking module T ,
is t1 = [0, 0, 0]. Recall that the special camera and register           we use the CoTracker2 architecture [57], which takes the
                g          R
tokens tg1 := t , tR1 := t allow the transformer to identify
                                                                         dense tracking features Ti as input. More specifically, given
the first camera.                                                        a query point yj in a query image Iq (during training, we al-
                                                                         ways set q = 1, but any other image can be potentially used
Camera Predictions. The camera parameters (ĝi )N     i=1 are            as a query), the tracking head T predicts the set of 2D points
predicted from the output camera tokens (t̂gi )N
                                               i=1 using four            T ((yj )M          N                N    M
                                                                                 j=1 , (Ti )i=1 ) = ((ŷj,i )i=1 )j=1 in all images Ii that
additional self-attention layers followed by a linear layer.             correspond to the same 3D point as y. To do so, the fea-
This forms the camera head that predicts the camera intrin-              ture map Tq of the query image is first bilinearly sampled at


                                                                     5
the query point yj to obtain its feature. This feature is then         point map P to the origin and use this scale to normalize
correlated with all other feature maps Ti , i ̸= q to obtain a         the camera translations t, the point map P , and the depth
set of correlation maps. These maps are then processed by              map D. Importantly, unlike [129], we do not apply such
self-attention layers to predict the final 2D points ŷi , which       normalization to the predictions output by the transformer;
are all in correspondence with yj . Note that, similar to VG-          instead, we force it to learn the normalization we choose
GSfM [125], our tracker does not assume any temporal or-               from the training data.
dering of the input frames and, hence, can be applied to any
                                                                       Implementation Details. By default, we employ L = 24
set of input images, not just videos.
                                                                       layers of global and frame-wise attention, respectively. The
3.4. Training                                                          model consists of approximately 1.2 billion parameters in
                                                                       total. We train the model by optimizing the training loss (2)
Training Losses. We train the VGGT model f end-to-end                  with the AdamW optimizer for 160K iterations. We use a
using a multi-task loss:                                               cosine learning rate scheduler with a peak learning rate of
                                                                       0.0002 and a warmup of 8K iterations. For every batch,
          L = Lcamera + Ldepth + Lpmap + λLtrack .          (2)        we randomly sample 2–24 frames from a random training
                                                                       scene. The input frames, depth maps, and point maps are re-
We found that the camera (Lcamera ), depth (Ldepth ), and
                                                                       sized to a maximum dimension of 518 pixels. The aspect ra-
point-map (Lpmap ) losses have similar ranges and do not
                                                                       tio is randomized between 0.33 and 1.0. We also randomly
need to be weighted against each other. The tracking loss
                                                                       apply color jittering, Gaussian blur, and grayscale augmen-
Ltrack is down-weighted with a factor of λ = 0.05. We de-
                                                                       tation to the frames. The training runs on 64 A100 GPUs
scribe each loss term in turn.
                                                                       over nine days. We employ gradient norm clipping with a
            PN loss Lcamera supervises the cameras ĝ:
    The camera
                                                                       threshold of 1.0 to ensure training stability. We leverage
Lcamera = i=1 ∥ĝi − gi ∥ϵ , comparing the predicted cam-
                                                                       bfloat16 precision and gradient checkpointing to improve
eras ĝi with the ground truth gi using the Huber loss | · |ϵ .
                                                                       GPU memory and computational efficiency.
    The depth loss Ldepth follows DUSt3R [129] and im-
plements the aleatoric-uncertainty loss [59, 75] weighing              Training Data. The model was trained using a large and
the discrepancy between the predicted depth D̂i and the                diverse collection of datasets, including: Co3Dv2 [88],
ground-truth depth Di with the predicted uncertainty map               BlendMVS [146], DL3DV [69], MegaDepth [64],
Σ̂D
  i . Differently from DUSt3R, we also apply a gradient-               Kubric [41], WildRGB [135], ScanNet [18], Hyper-
based term, which is widely used in monocular                          Sim [89], Mapillary [71], Habitat [107], Replica [104],
                                                   PN depth es-
timation. Hence, the depth loss is Ldepth = i=1 ∥ΣD         i ⊙        MVS-Synth [50], PointOdyssey [159], Virtual KITTI [7],
                     D                               D
(D̂i − Di )∥ + ∥Σi ⊙ (∇D̂i − ∇Di )∥ − α log Σi , where ⊙               Aria Synthetic Environments [82], Aria Digital Twin [82],
is the channel-broadcast element-wise product. The point               and a synthetic dataset of artist-created assets similar
map loss is defined analogously       but with the point-map un-       to Objaverse [20]. These datasets span various domains,
                            PN
certainty ΣP i : Lpmap =
                                      P                     P          including indoor and outdoor environments, and encompass
                                i=1 ∥Σi ⊙ (P̂i − Pi )∥ + ∥Σi ⊙
(∇P̂i − ∇Pi )∥ − α log Σi .   P                                        synthetic and real-world scenarios. The 3D annotations
                                                                       for these datasets are derived from multiple sources,
PM      PN the tracking loss is given by Ltrack =
    Finally,
                                                                       such as direct sensor capture, synthetic engines, or SfM
   j=1     i=1 ∥yj,i − ŷj,i ∥. Here, the outer sum runs over
all ground-truth query points yj in the query image Iq , yj,i          techniques [95]. The combination of our datasets is broadly
is yj ’s ground-truth correspondence in image Ii , and ŷj,i           comparable to those of MASt3R [30] in size and diversity.
is the corresponding prediction obtained by the application
T ((yj )M           N                                                  4. Experiments
         j=1 , (Ti )i=1 ) of the tracking module. Additionally,
following CoTracker2 [57], we apply a visibility loss (bi-             This section compares our method to state-of-the-art ap-
nary cross-entropy) to estimate whether a point is visible in          proaches across multiple tasks to show its effectiveness.
a given frame.
Ground Truth Coordinate Normalization. If we scale a
                                                                       4.1. Camera Pose Estimation
scene or change its global reference frame, the images of the          We first evaluate our method on the CO3Dv2 [88] and
scene are not affected at all, meaning that any such variant           RealEstate10K [161] datasets for camera pose estimation,
is a legitimate result of 3D reconstruction. We remove this            as shown in Tab. 1. Following [124], we randomly select 10
ambiguity by normalizing the data, thus making a canoni-               images per scene and evaluate them using the standard met-
cal choice and task the transformer to output this particular          ric AUC@30, which combines RRA and RTA. RRA (Rela-
variant. We follow [129] and, first, express all quantities in         tive Rotation Accuracy) and RTA (Relative Translation Ac-
the coordinate frame of the first camera g1 . Then, we com-            curacy) calculate the relative angular errors in rotation and
pute the average Euclidean distance of all 3D points in the            translation, respectively, for each image pair. These angu-


                                                                   6
                           Re10K (unseen)         CO3Dv2                         Method            AUC@5 ↑      AUC@10 ↑      AUC@20 ↑
          Methods                                              Time
                            AUC@30 ↑             AUC@30 ↑
                                                                                 SuperGlue [92]       16.2         33.8          51.8
  Colmap+SPSG [92]              45.2               25.3       ∼ 15s              LoFTR [105]          22.1         40.8          57.6
     PixSfM [66]                49.4               30.1       > 20s              DKM [32]             29.4         50.7          68.3
    PoseDiff [124]              48.0               66.5       ∼ 7s               CasMTR [9]           27.1         47.0          64.4
    DUSt3R [129]                67.7               76.7       ∼ 7s               Roma [33]            31.8         53.4          70.9
    MASt3R [62]                 76.4               81.8       ∼ 9s               Ours                 33.9         55.2          73.4
  VGGSfM v2 [125]               78.9               83.4       ∼ 10s
 MV-DUSt3R [111] ‡              71.3               69.5       ∼ 0.6s        Table 4. Two-View matching comparison on ScanNet-1500 [18,
  CUT3R [127] ‡                 75.3               82.8       ∼ 0.6s        92]. Although our tracking head is not specialized for the two-
  FLARE [156] ‡                 78.8               83.3       ∼ 0.5s        view setting, it outperforms the state-of-the-art two-view matching
   Fast3R [141] ‡               72.7               82.5       ∼ 0.2s        method Roma. Measured in AUC (higher is better).
 Ours (Feed-Forward)            85.3               88.2       ∼ 0.2s
   Ours (with BA)               93.5               91.8       ∼ 1.8s
                                                                            ods across all metrics on both datasets, including those that
Table 1. Camera Pose Estimation on RealEstate10K [161] and                  employ computationally expensive post-optimization steps,
CO3Dv2 [88] with 10 random frames. All metrics the higher the               such as Global Alignment for DUSt3R/MASt3R and Bun-
better. None of the methods were trained on the Re10K dataset.
                                                                            dle Adjustment for VGGSfM, typically requiring more than
Runtime were measured using one H100 GPU. Methods marked
with ‡ represent concurrent work.
                                                                            10 seconds. In contrast, VGGT achieves superior perfor-
                                                                            mance while only operating in a feed-forward manner, re-
                                                                            quiring just 0.2 seconds on the same hardware. Compared
 Known GT
                       Method            Acc.↓     Comp.↓    Overall↓       to concurrent works [111, 127, 141, 156] (indicated by ‡ ),
  camera
                                                                            our method demonstrates significant performance advan-
     ✓              Gipuma [40]          0.283      0.873     0.578
                                                                            tages, with speed similar to the fastest variant Fast3R [141].
     ✓             MVSNet [144]          0.396      0.527     0.462
     ✓              CIDER [139]          0.417      0.437     0.427         Furthermore, our model’s performance advantage is even
     ✓          PatchmatchNet [121]      0.427      0.377     0.417         more pronounced on the RealEstate10K dataset, which
     ✓              MASt3R [62]          0.403      0.344     0.374         none of the methods presented in Tab. 1 were trained on.
     ✓           GeoMVSNet [157]         0.331      0.259     0.295
                                                                            This validates the superior generalization of VGGT.
      ✗              DUSt3R [129]        2.677      0.805     1.741            Our results also show that VGGT can be improved even
      ✗                 Ours             0.389      0.374     0.382
                                                                            further by combining it with optimization methods from vi-
                                                                            sual geometry optimization like BA. Specifically, refining
Table 2. Dense MVS Estimation on the DTU [51] Dataset.
Methods operating with known ground-truth camera are in the top             the predicted camera poses and tracks with BA further im-
part of the table, while the bottom part contains the methods that          proves accuracy. Note that our method directly predicts
do not know the ground-truth camera.                                        close-to-accurate point/depth maps, which can serve as a
                                                                            good initialization for BA. This eliminates the need for tri-
           Methods          Acc.↓      Comp.↓     Overall↓   Time
                                                                            angulation and iterative refinement in BA as done by [125],
                                                                            making our approach significantly faster (only around 2 sec-
         DUSt3R             1.167      0.842       1.005     ∼ 7s
        MASt3R              0.968      0.684       0.826     ∼ 9s
                                                                            onds even with BA). Hence, while the feed-forward mode of
      Ours (Point)          0.901      0.518       0.709     ∼ 0.2s         VGGT outperforms all previous alternatives (whether they
   Ours (Depth + Cam)       0.873      0.482       0.677     ∼ 0.2s         are feed-forward or not), there is still room for improvement
                                                                            since post-optimization still brings benefits.
Table 3. Point Map Estimation on ETH3D [97]. DUSt3R and
MASt3R use global alignment while ours is feed-forward and,                 4.2. Multi-view Depth Estimation
hence, much faster. The row Ours (Point) indicates the results
using the point map head directly, while Ours (Depth + Cam) de-             Following MASt3R [62], we further evaluate our multi-
notes constructing point clouds from the depth map head com-                view depth estimation results on the DTU [51] dataset. We
bined with the camera head.                                                 report the standard DTU metrics, including Accuracy (the
                                                                            smallest Euclidean distance from the prediction to ground
                                                                            truth), Completeness (the smallest Euclidean distance from
lar errors are then thresholded to determine the accuracy                   the ground truth to prediction), and their average Overall
scores. AUC is the area under the accuracy-threshold curve                  (i.e., Chamfer distance). In Tab. 2, DUSt3R and our VGGT
of the minimum values between RRA and RTA across vary-                      are the only two methods operating without the knowl-
ing thresholds. The (learnable) methods in Tab. 1 have been                 edge of ground truth cameras. MASt3R derives depth maps
trained on Co3Dv2 and not on RealEstate10K. Our feed-                       by triangulating matches using the ground truth cameras.
forward model consistently outperforms competing meth-                      Meanwhile, deep multi-view stereo methods like GeoMVS-


                                                                        7
   Ours
CoTracker
 +Ours




Figure 5. Visualization of Rigid and Dynamic Point Tracking. Top: VGGT’s tracking module T outputs keypoint tracks for an
unordered set of input images depicting a static scene. Bottom: We finetune the backbone of VGGT to enhance a dynamic point tracker
CoTracker [56], which processes sequential inputs.


Net use ground truth cameras to construct cost volumes.                       ETH3D Dataset           Acc.↓   Comp.↓    Overall↓
   Our method substantially outperforms DUSt3R, reduc-                        Cross-Attention         1.287    0.835      1.061
ing the Overall score from 1.741 to 0.382. More impor-                   Global Self-Attention Only   1.032    0.621      0.827
tantly, it achieves results comparable to methods that know                Alternating-Attention      0.901    0.518      0.709
ground-truth cameras at test time. The significant perfor-
mance gains can likely be attributed to our model’s multi-          Table 5. Ablation Study for Transformer Backbone on ETH3D.
                                                                    We compare our alternating-attention architecture against two
image training scheme that teaches it to reason about multi-
                                                                    variants: one using only global self-attention and another employ-
view triangulation natively, instead of relying on ad hoc           ing cross-attention.
alignment procedures, such as in DUSt3R, which only av-
erages multiple pairwise camera triangulations.
                                                                    well, excelling on challenging out-of-domain examples,
4.3. Point Map Estimation                                           such as oil paintings, non-overlapping frames, and scenes
                                                                    with repeating or homogeneous textures like deserts.
We also compare the accuracy of our predicted point cloud
to DUSt3R and MASt3R on the ETH3D [97] dataset. For                 4.4. Image Matching
each scene, we randomly sample 10 frames. The pre-                  Two-view image matching is a widely-explored topic [68,
dicted point cloud is aligned to the ground truth using the         93, 105] in computer vision. It represents a specific case of
Umeyama [117] algorithm. The results are reported after             rigid point tracking, which is restricted to only two views,
filtering out invalid points using the official masks. We re-       and hence a suitable evaluation benchmark to measure our
port Accuracy, Completeness, and Overall (Chamfer dis-              tracking accuracy, even though our model is not specialized
tance) for point map estimation. As shown in Tab. 3, al-            for this task. We follow the standard protocol [33, 93] on
though DUSt3R and MASt3R conduct expensive optimiza-                the ScanNet dataset [18] and report the results in Tab. 4.
tion (global alignment–—around 10 seconds per scene),               For each image pair, we extract the matches and use them
our method still outperforms them significantly in a simple         to estimate an essential matrix, which is then decomposed
feed-forward regime at only 0.2 seconds per reconstruction.         to a relative camera pose. The final metric is the relative
    Meanwhile, compared to directly using our estimated             pose accuracy, measured by AUC. For evaluation, we use
point maps, we found that the predictions from our depth            ALIKED [158] to detect keypoints, treating them as query
and camera heads (i.e., unprojecting the predicted depth            points yq . These are then passed to our tracking branch T
maps to 3D using the predicted camera parameters) yield             to find correspondences in the second frame. We adopt the
higher accuracy. We attribute this to the benefits of de-           evaluation hyperparameters (e.g., the number of matches,
composing a complex task (point map estimation) into sim-           RANSAC thresholds) from Roma [33]. Despite not being
pler subproblems (depth map and camera prediction), even            explicitly trained for two-view matching, Tab. 4 shows that
though camera, depth maps, and point maps are jointly su-           VGGT achieves the highest accuracy among all baselines.
pervised during training.
    We present a qualitative comparison with DUSt3R on in-
                                                                    4.5. Ablation Studies
the-wild scenes in Fig. 3 and further examples in Fig. 4.           Feature Backbone. We first validate the effectiveness of
VGGT outputs high-quality predictions and generalizes               our proposed Alternating-Attention design by comparing it


                                                                8
      w. Lcamera   w. Ldepth   w. Ltrack   Acc.↓   Comp.↓   Overall↓
                                                                           Input Images
          ✗           ✓           ✓        1.042   0.627     0.834
          ✓           ✗           ✓        0.920   0.534     0.727
          ✓           ✓           ✗        0.976   0.603     0.790
          ✓           ✓           ✓        0.901   0.518     0.709
                                                                           Ground Truth


Table 6. Ablation Study for Multi-task Learning, which shows
that simultaneous training with camera, depth and track estimation
                                                                            Prediction
yields the highest accuracy in point map estimation on ETH3D.

                                                                           Figure 6. Qualitative Examples of Novel View Synthesis. The
against two alternative attention architectures: (a) global                top row shows the input images, the middle row displays the
self-attention only, and (b) cross-attention. To ensure a                  ground truth images from target viewpoints, and the bottom row
fair comparison, all model variants maintain an identical                  presents our synthesized images.
number of parameters, using a total of 2L attention lay-
ers. For the cross-attention variant, each frame indepen-                           Method    Known Input Cam   Size   PSNR ↑   SSIM ↑   LPIPS ↓
dently attends to tokens from all other frames, maximiz-                        LGM [110]           ✓           256    21.44    0.832     0.122
ing cross-frame information fusion although significantly                      GS-LRM [154]         ✓           256    29.59    0.944     0.051
                                                                                LVSM [53]           ✓           256    31.71    0.957     0.027
increasing the runtime, particularly as the number of input
                                                                                 Ours-NVS∗          ✗           224    30.41    0.949     0.033
frames grows. The hyperparameters such as the hidden di-
mension and the number of heads are kept the same. Point
map estimation accuracy is chosen as the evaluation metric                 Table 7. Quantitative comparisons for view synthesis on
                                                                           GSO [28] dataset. Finetuning VGGT for feed-forward novel view
for our ablation study, as it reflects the model’s joint under-
                                                                           synthesis, it demonstrates competitive performance even without
standing of scene geometry and camera parameters. Results                  knowing camera extrinsic and intrinsic parameters for the input
in Tab. 5 demonstrate that our Alternating-Attention archi-                images. Note that ∗ indicates using a small training set (only 20%).
tecture outperforms both baseline variants by a clear mar-
gin. Additionally, our other preliminary exploratory exper-
iments consistently showed that architectures using cross-
attention generally underperform compared to those exclu-                  closely, e.g., using 4 input views and adopting Plücker rays
sively employing self-attention.                                           to represent target viewpoints. We make a simple modifi-
                                                                           cation to VGGT. As before, the input images are converted
Multi-task Learning. We also verify the benefit of train-                  into tokens by DINO. Then, for the target views, we use
ing a single network to simultaneously learn multiple 3D                   a convolutional layer to encode their Plücker ray images
quantities, even though these outputs may potentially over-                into tokens. These tokens, representing both the input im-
lap (e.g., depth maps and camera parameters together can                   ages and the target views, are concatenated and processed
produce point maps). As shown in Tab. 6, there is a no-                    by the AA transformer. Subsequently, a DPT head is used
ticeable decrease in the accuracy of point map estimation                  to regress the RGB colors for the target views. It is impor-
when training without camera, depth, or track estimation.                  tant to note that we do not input the Plücker rays for the
Notably, incorporating camera parameter estimation clearly                 source images. Hence, the model is not given the camera
enhances point map accuracy, whereas depth estimation                      parameters for these input frames.
contributes only marginal improvements.                                       LVSM was trained on the Objaverse dataset [20]. We
                                                                           use a similar internal dataset of approximately 20% the size
4.6. Finetuning for Downstream Tasks
                                                                           of Objaverse. Further details on training and evaluation can
We now show that the VGGT pre-trained feature extractor                    be found in [53]. As shown in Tab. 7, despite not requir-
can be reused in downstream tasks. We show this for feed-                  ing the input camera parameters and using less training data
forward novel view synthesis and dynamic point tracking.                   than LVSM, our model achieves competitive results on the
Feed-forward Novel View Synthesis is progressing                           GSO dataset [28]. We expect that better results would be
rapidly [8, 43, 49, 53, 108, 126, 140, 155]. Most exist-                   obtained using a larger training dataset. Qualitative exam-
ing methods take images with known camera parameters as                    ples are shown in Fig. 6.
input and predict the target image corresponding to a new                  Dynamic Point Tracking has emerged as a highly com-
camera viewpoint. Instead of relying on an explicit 3D rep-                petitive task in recent years [25, 44, 57, 136], and it serves
resentation, we follow LVSM [53] and modify VGGT to di-                    as another downstream application for our learned features.
rectly output the target image. However, we do not assume                  Following standard practices, we report these point-tracking
known camera parameters for the input frames.                              metrics: Occlusion Accuracy (OA), which comprises the bi-
                                                                                                                      vis
   We follow the training and evaluation protocol of LVSM                  nary accuracy of occlusion predictions; δavg   , comprising the


                                                                       9
                                                                         Input Frames    1      2      4      8      10     20      50     100     200
                      Kinetics       RGB-S          DAVIS                  Time (s)     0.04   0.05   0.07   0.11   0.14   0.31    1.04    3.12    8.75
  Method
                       vis OA AJ δ vis OA AJ δ vis OA                     Mem. (GB)     1.88   2.07   2.45   3.23   3.63   5.58   11.41   21.15   40.63
                   AJ δavg        avg         avg

  TAPTR [63]      49.0 64.4 85.2 60.8 76.2 87.0 63.0 76.1 91.1          Table 9. Runtime and peak GPU memory usage across differ-
  LocoTrack [13] 52.9 66.8 85.3 69.7 83.2 89.5 62.9 75.3 87.2           ent numbers of input frames. Runtime is measured in seconds,
  BootsTAPIR [26] 54.6 68.4 86.5 70.8 83.0 89.9 61.4 73.6 88.7
                                                                        and GPU memory usage is reported in gigabytes.
  CoTracker [56]   49.6 64.3 83.3 67.4 78.9 85.2 61.8 76.1 88.3
  CoTracker + Ours 57.2 69.0 88.9 72.1 84.0 91.6 64.7 77.5 91.4
                                                                        Runtime and Memory. As shown in Tab. 9, we evalu-
Table 8. Dynamic Point Tracking Results on the TAP-Vid
                                                                        ate inference runtime and peak GPU memory usage of the
benchmarks. Although our model was not designed for dynamic
                                                                        feature backbone when processing varying numbers of in-
scenes, simply fine-tuning CoTracker with our pretrained weights
significantly enhances performance, demonstrating the robustness        put frames. Measurements are conducted using a single
and effectiveness of our learned features.                              NVIDIA H100 GPU with flash attention v3 [98]. Images
                                                                        have a resolution of 336 × 518.
                                                                           We focus on the cost associated with the feature back-
mean proportion of visible points accurately tracked within             bone since users may select different branch combinations
a certain pixel threshold; and Average Jaccard (AJ), mea-               depending on their specific requirements and available re-
suring tracking and occlusion prediction accuracy together.             sources. The camera head is lightweight, typically account-
    We adapt the state-of-the-art CoTracker2 model [57] by              ing for approximately 5% of the runtime and about 2% of
substituting its backbone with our pretrained feature back-             the GPU memory used by the feature backbone. A DPT
bone. This is necessary because VGGT is trained on un-                  head uses an average of 0.03 seconds and 0.2 GB GPU
ordered image collections instead of sequential videos. Our             memory per frame.
backbone predicts the tracking features Ti , which replace                 When GPU memory is sufficient, multiple frames can be
the outputs of the feature extractor and later enter the rest of        processed efficiently in a single forward pass. At the same
the CoTracker2 architecture, that finally predicts the tracks.          time, in our model, inter-frame relationships are handled
We finetune the entire modified tracker on Kubric [41]. As              only within the feature backbone, and the DPT heads make
illustrated in Tab. 8, the integration of pretrained VGGT sig-          independent predictions per frame. Therefore, users con-
nificantly enhances CoTracker’s performance on the TAP-                 strained by GPU resources may perform predictions frame
Vid benchmark [23]. For instance, VGGT’s tracking fea-                  by frame. We leave this trade-off to the user’s discretion.
                     vis                                                   We recognize that a naive implementation of global self-
tures improve the δavg   metric from 78.9 to 84.0 on the TAP-
Vid RGB-S dataset. Despite the TAP-Vid benchmark’s in-                  attention can be highly memory-intensive with a large num-
clusion of videos featuring rapid dynamic motions from var-             ber of tokens. Savings or accelerations can be achieved by
ious data sources, our model’s strong performance demon-                employing techniques used in large language model (LLM)
strates the generalization capability of its features, even in          deployments. For instance, Fast3R [141] employs Ten-
scenarios for which it was not explicitly designed.                     sor Parallelism to accelerate inference with multiple GPUs,
                                                                        which can be directly applied to our model.
5. Discussions                                                          Patchifying. As discussed in Sec. 3.2, we have explored
                                                                        the method of patchifying images into tokens by utilizing
Limitations. While our method exhibits strong generaliza-
                                                                        either a 14 × 14 convolutional layer or a pretrained DI-
tion to diverse in-the-wild scenes, several limitations re-
                                                                        NOv2 model. Empirical results indicate that the DINOv2
main. First, the current model does not support fisheye
                                                                        model provides better performance; moreover, it ensures
or panoramic images. Additionally, reconstruction perfor-
                                                                        much more stable training, particularly in the initial stages.
mance drops under conditions involving extreme input ro-
                                                                        The DINOv2 model is also less sensitive to variations in
tations. Moreover, although our model handles scenes with
                                                                        hyperparameters such as learning rate or momentum. Con-
minor non-rigid motions, it fails in scenarios involving sub-
                                                                        sequently, we have chosen DINOv2 as the default method
stantial non-rigid deformation.
                                                                        for patchifying in our model.
    However, an important advantage of our approach is its
flexibility and ease of adaptation. Addressing these limi-              Differentiable BA. We also explored the idea of using
tations can be straightforwardly achieved by fine-tuning the            differentiable bundle adjustment as in VGGSfM [125].
model on targeted datasets with minimal architectural modi-             In small-scale preliminary experiments, differentiable BA
fications. This adaptability clearly distinguishes our method           demonstrated promising performance. However, a bottle-
from existing approaches, which typically require extensive             neck is its computational cost during training. Enabling
re-engineering during test-time optimization to accommo-                differentiable BA in PyTorch using Theseus [85] typically
date such specialized scenarios.                                        makes each training step roughly 4 times slower, which


                                                                   10
is expensive for large-scale training. While customizing a             Appendix
framework to expedite training could be a potential solu-
tion, it falls outside the scope of this work. Thus, we opted          In the Appendix, we provide the following:
not to include differentiable BA in this work, but we recog-           • formal definitions of key terms in Appendix A.
nize it as a promising direction for large-scale unsupervised          • comprehensive implementation details, including archi-
training, as it can serve as an effective supervision signal in          tecture and training hyperparameters in Appendix B.
scenarios lacking explicit 3D annotations.                             • additional experiments and discussions in Appendix C.
                                                                       • qualitative examples of single-view reconstruction in Ap-
Single-view Reconstruction. Unlike systems like DUSt3R                   pendix D.
and MASt3R that have to duplicate an image to create a                 • an expanded review of related works in Appendix E.
pair, our model architecture inherently supports the input of
a single image. In this case, global attention simply transi-          A. Formal Definitions
tions to frame-wise attention. Although our model was not
explicitly trained for single-view reconstruction, it demon-           In this section, we provide additional formal definitions that
strates surprisingly good results. Some examples can be                further ground the method section.
found in Fig. 3 and Fig. 7. We strongly encourage trying                   The camera extrinsics are defined in relation to the world
our demo for better visualization.                                     reference frame, which we take to be the coordinate system
                                                                       of the first camera. We thus introduce two functions. The
Normalizing Prediction. As discussed in Sec. 3.4, our ap-
                                                                       first function γ(g, p) = p′ applies the rigid transformation
proach normalizes the ground truth using the average Eu-
                                                                       encoded by g to a point p in the world reference frame to
clidean distance of the 3D points. While some methods,
                                                                       obtain the corresponding point p′ in the camera reference
such as DUSt3R, also apply such normalization to network
                                                                       frame. The second function π(g, p) = y further applies
predictions, our findings suggest that it is neither necessary
                                                                       perspective projection, mapping the 3D point p to a 2D im-
for convergence nor advantageous for final model perfor-
                                                                       age point y. We also denote the depth of the point as ob-
mance. Furthermore, it tends to introduce additional insta-
                                                                       served from the camera g by π D (g, p) = d ∈ R+ .
bility during the training phase.
                                                                           We model the scene as a collection of regular surfaces
6. Conclusions                                                         Si ⊂ R3 . We make this a function of the i-th input image
                                                                       as the scene can change over time [151]. The depth at pixel
We present Visual Geometry Grounded Transformer                        location y ∈ I(Ii ) is defined as the minimum depth of any
(VGGT), a feed-forward neural network that can directly                3D point p in the scene that projects to y, i.e., Di (y) =
estimate all key 3D scene properties for hundreds of in-               min{π D (gi , p) : p ∈ Si ∧ π(gi , p) = y}. The point at
put views. It achieves state-of-the-art results in multiple            pixel location y is then given by Pi (y) = γ(g, p), where
3D tasks, including camera parameter estimation, multi-                p ∈ Si is the 3D point that minimizes the expression above,
view depth estimation, dense point cloud reconstruction,               i.e., p ∈ Si ∧ π(gi , p) = y ∧ π D (gi , p) = Di (y).
and 3D point tracking. Our simple, neural-first approach
departs from traditional visual geometry-based methods,                B. Implementation Details
which rely on optimization and post-processing to obtain
accurate and task-specific results. The simplicity and effi-           Architecture. As mentioned in the main paper, VGGT
ciency of our approach make it well-suited for real-time ap-           consists of 24 attention blocks, each block equipped with
plications, which is another benefit over optimization-based           one frame-wise self-attention layer and one global self-
approaches.                                                            attention layer. Following the ViT-L model used in DI-
                                                                       NOv2 [78], each attention layer is configured with a feature
                                                                       dimension of 1024 and employs 16 heads. We use the of-
                                                                       ficial implementation of the attention layer from PyTorch,
                                                                       i.e., torch.nn.MultiheadAttention, with flash attention en-
                                                                       abled. To stabilize training, we also use QKNorm [48] and
                                                                       LayerScale [115] for each attention layer. The value of Lay-
                                                                       erScale is initialized with 0.01. For image tokenization,
                                                                       we use DINOv2 [78] and add positional embedding. As
                                                                       in [143], we feed the tokens from the 4-th, 11-th, 17-th, and
                                                                       23-rd block into DPT [87] for upsampling.
                                                                       Training. To form a training batch, we first choose a ran-
                                                                       dom training dataset (each dataset has a different yet ap-
                                                                       proximately similar weight, as in [129]), and from the


                                                                  11
dataset, we then sample a random scene (uniformly). Dur-                      Method            Test-time Opt.   AUC@3◦   AUC@5◦   AUC@10◦   Runtime
                                                                      COLMAP (SIFT+NN) [94]           ✓           23.58    32.66    44.79     >10s
ing the training phase, we select between 2 and 24 frames             PixSfM (SIFT + NN) [66]         ✓           25.54    34.80    46.73     >20s
per scene while maintaining the constant total of 48 frames             PixSfM (LoFTR) [66]
                                                                       PixSfM (SP + SG) [66]
                                                                                                      ✓
                                                                                                      ✓
                                                                                                                  44.06
                                                                                                                  45.19
                                                                                                                           56.16
                                                                                                                           57.22
                                                                                                                                    69.61
                                                                                                                                    70.47
                                                                                                                                              >20s
                                                                                                                                              >20s
within each batch. For training, we use the respective train-           DFSfM (LoFTR) [47]            ✓           46.55    58.74    72.19     >10s

ing sets of each dataset. We exclude training sequences con-              DUSt3R [129]
                                                                           MASt3R [62]
                                                                                                      ✓
                                                                                                      ✓
                                                                                                                  13.46
                                                                                                                  30.25
                                                                                                                           21.24
                                                                                                                           46.79
                                                                                                                                    35.62
                                                                                                                                    57.42
                                                                                                                                              ∼ 7s
                                                                                                                                              ∼ 9s
taining fewer than 24 frames. RGB frames, depth maps,                     VGGSfM [125]                ✓           45.23    58.89    73.92     ∼ 6s
                                                                         VGGSfMv2 [125]               ✓           59.32    67.78    76.82     ∼ 10s
and point maps are first isotropically resized, so the longer
                                                                           VGGT (ours)                ✗           39.23    52.74    71.26     0.2s
size has 518 pixels. Then, we crop the shorter dimen-                    VGGT + BA (ours)             ✓           66.37    75.16    84.91     1.8s

sion (around the principal point) to a size between 168 and
518 pixels while remaining a multiple of the 14-pixel patch          Table 10. Camera Pose Estimation on IMC [54]. Our method
size. It is worth mentioning that we apply aggressive color          achieves state-of-the-art performance on the challenging pho-
augmentation independently across each frame within the              totropism data, outperforming VGGSfMv2 [125] which ranked
                                                                     first on the latest CVPR’24 IMC Challenge in camera pose (ro-
same scene, enhancing the model’s robustness to varying
                                                                     tation and translation) estimation.
lighting conditions. We build ground truth tracks follow-
ing [33, 105, 125], which unprojects depth maps to 3D,
reprojects points to target frames, and retains correspon-           methods, our VGGT’s feed-forward performance is on par
dences where reprojected depths match target depth maps.             with the state-of-the-art VGGSfMv2 with AUC@10 of
Frames with low similarity to the query frame are excluded           71.26 versus 76.82, while being significantly faster (0.2
during batch sampling. In rare cases with no valid corre-            vs. 10 seconds per scene). Remarkably, VGGT outper-
spondences, the tracking loss is omitted.                            forms both MASt3R [62] and DUSt3R [129] significantly
                                                                     across all accuracy thresholds while being much faster. This
C. Additional Experiments                                            is because MASt3R’s and DUSt3R’s feed-forward predic-
Camera Pose Estimation on IMC We also evaluate using                 tions can only process pairs of frames and, hence, require a
the Image Matching Challenge (IMC) [54], a camera pose               costly global alignment step. Additionally, with bundle ad-
estimation benchmark focusing on phototourism data. Until            justment, VGGT + BA further improves drastically, achiev-
recently, the benchmark was dominated by classical incre-            ing state-of-the-art performance on IMC, raising AUC@10
mental SfM methods [94].                                             from 71.26 to 84.91, and raising AUC@3 from 39.23 to
                                                                     66.37. Note that our model directly predicts 3D points,
Baselines. We evaluate two flavors of our model: VGGT                which can serve as the initialization for BA. This elimi-
and VGGT + BA. VGGT directly outputs camera pose es-                 nates the need for triangulation and iterative refinement of
timates, while VGGT + BA refines the estimates using an              BA as in [125]. As a result, VGGT + BA is much faster
additional Bundle Adjustment stage. We compare to the                than [125].
classical incremental SfM methods such as [66, 94] and
to recently-proposed deep methods. Specifically, recently
                                                                     D. Qualitative Examples
VGGSfM [125] provided the first end-to-end trained deep
method that outperformed incremental SfM on the chal-                We further present qualitative examples of single-view re-
lenging phototourism datasets.                                       construction in Fig. 7.
   Besides VGGSfM, we additionally compare to recently
popularized DUSt3R [129] and MASt3R [62]. It is im-                  E. Related Work
portant to note that DUSt3R and MASt3R utilized a sub-
stantial portion of the MegaDepth dataset for training, only         In this section, we discuss additional related works.
excluding scenes 0015 and 0022. The MegaDepth scenes                 Vision Transformers. The Transformer architecture was
employed in their training have some overlap with the                initially proposed for language processing tasks [6, 22,
IMC benchmark, although the images are not identical; the            120]. It was later introduced to the computer vision com-
same scenes are present in both datasets. For instance, the          munity by ViT [27], sparking widespread adoption. Vision
MegaDepth scene 0024 corresponds to the British Museum,              Transformers and their variants have since become domi-
while the British Museum is also a scene in the IMC bench-           nant in the design of architectures for various computer vi-
mark. For an apples-to-apples comparison, we adopt the               sion tasks [4, 12, 83, 137], thanks to their simplicity, high
same training split as DUSt3R and MASt3R. In the main                capacity, flexibility, and ability to capture long-range de-
paper, to ensure a fair comparison on ScanNet-1500, we ex-           pendencies.
clude the corresponding ScanNet scenes from our training.
                                                                         DeiT [114] demonstrated that Vision Transformers can
Results. Table 10 contains the results of our evaluation. Al-        be effectively trained on datasets like ImageNet using
though phototourism data is the traditional focus of SfM             strong data augmentation strategies. DINO [10] revealed


                                                                12
Figure 7. Single-view Reconstruction by Point Map Estimation. Unlike DUSt3R, which requires duplicating an image into a pair, our
model can predict the point map from a single input image. It demonstrates strong generalization to unseen real-world images.


intriguing properties of features learned by Vision Trans-            References
formers in a self-supervised manner. CaiT [115] introduced
                                                                       [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
layer scaling to address the challenges of training deeper
                                                                           Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Vision Transformers, effectively mitigating gradient-related               Almeida, Janko Altenschmidt, Sam Altman, Shyamal
issues. Further, techniques such as QKNorm [48, 150] have                  Anadkat, et al. Gpt-4 technical report. arXiv preprint
been proposed to stabilize the training process. Addition-                 arXiv:2303.08774, 2023. 2
ally, [138] also explores the dynamics between frame-wise              [2] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian
and global attention modules in object tracking, though us-                Simon, Brian Curless, Steven M Seitz, and Richard
ing cross-attention.                                                       Szeliski. Building rome in a day. Communications of the
                                                                           ACM, 54(10):105–112, 2011. 2, 13
Camera Pose Estimation. Estimating camera poses from                   [3] Mica Arie-Nachimson, Shahar Z Kovalsky, Ira
multi-view images is a crucial problem in 3D computer vi-                  Kemelmacher-Shlizerman, Amit Singer, and Ronen
sion. Over the last decades, Structure from Motion (SfM)                   Basri. Global motion estimation from point matches. In
has emerged as the dominant approach [46], whether in-                     2012 Second international conference on 3D imaging,
cremental [2, 36, 94, 103, 134] or global [3, 14–17, 52,                   modeling, processing, visualization & transmission, pages
73, 79, 81, 90, 106]. Recently, a set of methods treat                     81–88. IEEE, 2012. 13
camera pose estimation as a regression problem [65, 100,               [4] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen
109, 112, 113, 118, 122, 123, 131, 152, 153, 160], which                   Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video
show promising results under the sparse-view setting. Ace-                 vision transformer. In Proceedings of the IEEE/CVF inter-
Zero [5] further proposes to regress 3D scene coordinates                  national conference on computer vision, pages 6836–6846,
                                                                           2021. 12
and FlowMap [101] focuses on depth maps, as intermedi-
                                                                       [5] Eric Brachmann, Jamie Wynn, Shuai Chen, Tommaso Cav-
ates for camera prediction. Instead, VGGSfM [125] simpli-
                                                                           allari, Áron Monszpart, Daniyar Turmukhambetov, and
fies the classical SfM pipeline to a differentiable framework,             Victor Adrian Prisacariu. Scene coordinate reconstruction:
demonstrating exceptional performance, particularly with                   Posing of image collections via incremental learning of a
phototourism datasets. At the same time, DUSt3R [62, 129]                  relocalizer. In ECCV, 2024. 2, 13
introduces an approach to learn pixel-aligned point map,               [6] Tom B Brown. Language models are few-shot learners.
and hence camera poses can be recovered by simple align-                   arXiv preprint arXiv:2005.14165, 2020. 12
ment. This paradigm shift has garnered considerable inter-             [7] Yohann Cabon, Naila Murray, and Martin Humenberger.
est as the point map, an over-parameterized representation,                Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 6
offers seamless integration with various downstream appli-             [8] Ang Cao, Justin Johnson, Andrea Vedaldi, and David
cations, such as 3D Gaussian splatting.                                    Novotny. Lightplane: Highly-scalable components for neu-
                                                                           ral 3Dfields. In Proceedings of the International Confer-
                                                                           ence on 3D Vision (3DV), 2025. 9
                                                                       [9] Chenjie Cao and Yanwei Fu. Improving transformer-based
                                                                           image matching by cascaded capturing spatially informa-
                                                                           tive keypoints. In Proceedings of the IEEE/CVF Inter-
                                                                           national Conference on Computer Vision (ICCV), pages
                                                                           12129–12139, 2023. 7
                                                                      [10] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou,
                                                                           Julien Mairal, Piotr Bojanowski, and Armand Joulin.


                                                                 13
     Emerging properties in self-supervised vision transformers.              Zisserman, and Yi Yang. Tap-vid: A benchmark for track-
     In Proc. ICCV, 2021. 2, 12                                               ing any point in a video. arXiv, 2022. 2, 10
[11] Hongkai Chen, Zixin Luo, Jiahui Zhang, Lei Zhou, Xuyang             [24] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush
     Bai, Zeyu Hu, Chiew-Lan Tai, and Long Quan. Learning                     Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman.
     to match features with seeded graph matching network. In                 TAPIR: Tracking any point with per-frame initialization
     Proceedings of the IEEE/CVF International Conference on                  and temporal refinement. arXiv, 2306.08637, 2023. 2
     Computer Vision, pages 6301–6310, 2021. 2                           [25] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush
[12] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexan-                   Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman.
     der Kirillov, and Rohit Girdhar. Masked-attention mask                   TAPIR: tracking any point with per-frame initialization and
     transformer for universal image segmentation. In Proceed-                temporal refinement. In Proc. CVPR, 2023. 3, 9
     ings of the IEEE/CVF conference on computer vision and              [26] Carl Doersch, Yi Yang, Dilara Gokay, Pauline Luc, Skanda
     pattern recognition, pages 1290–1299, 2022. 12                           Koppula, Ankush Gupta, Joseph Heyward, Ross Goroshin,
[13] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seun-                    João Carreira, and Andrew Zisserman. Bootstap: Boot-
     gryong Kim, and Joon-Young Lee. Local all-pair corre-                    strapped training for tracking-any-point. arXiv preprint
     spondence for point tracking. Proc. ECCV, 2024. 3, 10                    arXiv:2402.00847, 2024. 10
[14] David J Crandall, Andrew Owens, Noah Snavely, and                   [27] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
     Daniel P Huttenlocher. Sfm with mrfs: Discrete-continuous                Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
     optimization for large-scale structure from motion. IEEE                 Mostafa Dehghani, Matthias Minderer, Georg Heigold,
     transactions on pattern analysis and machine intelligence,               Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An im-
     35(12):2841–2853, 2012. 13                                               age is worth 16×16 words: Transformers for image recog-
[15] Hainan Cui, Xiang Gao, Shuhan Shen, and Zhanyi Hu.                       nition at scale. In Proc. ICLR, 2021. 12
     Hsfm: Hybrid structure-from-motion. In Proceedings of               [28] Laura Downs, Anthony Francis, Nate Koenig, Brandon
     the IEEE conference on computer vision and pattern recog-                Kinman, Ryan Hickman, Krista Reymann, Thomas B
     nition, pages 1212–1221, 2017.                                           McHugh, and Vincent Vanhoucke. Google scanned objects:
[16] Zhaopeng Cui and Ping Tan. Global structure-from-motion                  A high-quality dataset of 3d scanned household items. In
     by similarity averaging. In Proceedings of the IEEE Inter-               2022 International Conference on Robotics and Automa-
     national Conference on Computer Vision, pages 864–872,                   tion (ICRA), pages 2553–2560. IEEE, 2022. 9
     2015.                                                               [29] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
                                                                              hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
[17] Zhaopeng Cui, Nianjuan Jiang, Chengzhou Tang, and Ping
                                                                              Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh
     Tan. Linear global translation estimation with feature
                                                                              Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra,
     tracks. arXiv preprint arXiv:1503.01832, 2015. 13
                                                                              Archie Sravankumar, Artem Korenev, Arthur Hinsvark,
[18] Angela Dai, Angel X Chang, Manolis Savva, Maciej Hal-
                                                                              Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen
     ber, Thomas Funkhouser, and Matthias Nießner. Scannet:
                                                                              Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron,
     Richly-annotated 3d reconstructions of indoor scenes. In
                                                                              Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya
     Proceedings of the IEEE conference on computer vision
                                                                              Nayak, Chloe Bi, Chris Marra, Chris McConnell, Chris-
     and pattern recognition, pages 5828–5839, 2017. 6, 7, 8
                                                                              tian Keller, Christophe Touret, Chunyang Wu, Corinne
[19] Timothée Darcet, Maxime Oquab, Julien Mairal, and Pi-                   Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien
     otr Bojanowski. Vision transformers need registers. arXiv                Allonsius, Daniel Song, Danielle Pintz, Danny Livshits,
     preprint arXiv:2309.16588, 2023. 4                                       David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego
[20] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,                 Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor
     Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana                      Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Di-
     Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse:                  nan, Eric Michael Smith, Filip Radenovic, Frank Zhang,
     A universe of annotated 3d objects. In Proceedings of                    Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Ander-
     the IEEE/CVF Conference on Computer Vision and Pattern                   son, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem
     Recognition, pages 13142–13153, 2023. 6, 9                               Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo
[21] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-                      Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M.
     novich. Superpoint: Self-supervised interest point detection             Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon
     and description. In Proceedings of the IEEE conference on                Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar,
     computer vision and pattern recognition workshops, pages                 Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny
     224–236, 2018. 2                                                         Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang,
[22] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina                   Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak,
     Toutanova. Bert: Pre-training of deep bidirectional trans-               Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe,
     formers for language understanding. In North American                    Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani,
     Chapter of the Association for Computational Linguistics,                Kate Plawiak, Ke Li, Kenneth Heafield, and Kevin Stone.
     2019. 12                                                                 The Llama 3 herd of models. arXiv, 2407.21783, 2024. 2
[23] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adrià Re-             [30] Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel,
     casens, Lucas Smaira, Yusuf Aytar, João Carreira, Andrew                Vincent Leroy, Yohann Cabon, and Jerome Revaud.


                                                                    14
     MASt3R-SfM: a fully-integrated solution for unconstrained           [42] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai,
     structure-from-motion. arXiv, 2409.19152, 2024. 6                        Feitong Tan, and Ping Tan. Cascade cost volume for high-
[31] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Polle-                  resolution multi-view stereo and stereo matching. In Pro-
     feys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-               ceedings of the IEEE/CVF conference on computer vision
     net: A trainable cnn for joint description and detection of              and pattern recognition, pages 2495–2504, 2020. 2
     local features. In Proceedings of the ieee/cvf conference on        [43] Junlin Han, Jianyuan Wang, Andrea Vedaldi, Philip Torr,
     computer vision and pattern recognition, pages 8092–8101,                and Filippos Kokkinos. Flex3d: Feed-forward 3d genera-
     2019. 2                                                                  tion with flexible reconstruction model and input view cu-
[32] Johan Edstedt, Ioannis Athanasiadis, Mårten Wadenbäck,                 ration. arXiv preprint arXiv:2410.00890, 2024. 9
     and Michael Felsberg. DKM: Dense kernelized feature                 [44] Adam W Harley, Zhaoyuan Fang, and Katerina Fragki-
     matching for geometry estimation. In IEEE Conference on                  adaki. Particle video revisited: Tracking through occlusions
     Computer Vision and Pattern Recognition, 2023. 7                         using point trajectories. In Proc. ECCV, 2022. 2, 9
[33] Johan Edstedt, Qiyu Sun, Georg Bökman, Mårten                     [45] Richard Hartley and Andrew Zisserman. Multiple View Ge-
     Wadenbäck, and Michael Felsberg. Roma: Robust dense                     ometry in Computer Vision. Cambridge University Press,
     feature matching. In Proceedings of the IEEE/CVF Confer-                 2000. 1, 2
     ence on Computer Vision and Pattern Recognition, pages              [46] Richard Hartley and Andrew Zisserman. Multiple View Ge-
     19790–19800, 2024. 7, 8, 12                                              ometry in Computer Vision. Cambridge University Press,
[34] Patrick Esser, Robin Rombach, and Björn Ommer. Taming                   ISBN: 0521540518, 2004. 13
     transformers for high-resolution image synthesis. In Proc.          [47] Xingyi He, Jiaming Sun, Yifan Wang, Sida Peng, Qixing
     CVPR, 2021. 2                                                            Huang, Hujun Bao, and Xiaowei Zhou. Detector-free struc-
                                                                              ture from motion. In arxiv, 2023. 12
[35] Martin A Fischler and Robert C Bolles. Random sample
                                                                         [48] Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and
     consensus: a paradigm for model fitting with applications
                                                                              Yuxuan Chen. Query-key normalization for transformers.
     to image analysis and automated cartography. Communica-
                                                                              arXiv preprint arXiv:2010.04245, 2020. 11, 13
     tions of the ACM, 24(6):381–395, 1981. 3
                                                                         [49] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
[36] Jan-Michael Frahm, Pierre Fite-Georgel, David Gallup,
                                                                              Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and
     Tim Johnson, Rahul Raguram, Changchang Wu, Yi-Hung
                                                                              Hao Tan. LRM: Large reconstruction model for single im-
     Jen, Enrique Dunn, Brian Clipp, Svetlana Lazebnik, et al.
                                                                              age to 3D. In Proc. ICLR, 2024. 2, 9
     Building rome on a cloudless day. In Computer Vision–
                                                                         [50] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra
     ECCV 2010: 11th European Conference on Computer Vi-
                                                                              Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view
     sion, Heraklion, Crete, Greece, September 5-11, 2010, Pro-
                                                                              stereopsis. In IEEE Conference on Computer Vision and
     ceedings, Part IV 11, pages 368–381. Springer, 2010. 2, 13
                                                                              Pattern Recognition (CVPR), 2018. 6
[37] Qiancheng Fu, Qingshan Xu, Yew Soon Ong, and Wen-                   [51] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engil Tola,
     bing Tao. Geo-neus: Geometry-consistent neural implicit                  and Henrik Aanæs. Large scale multi-view stereopsis eval-
     surfaces learning for multi-view reconstruction. Advances                uation. In 2014 IEEE Conference on Computer Vision and
     in Neural Information Processing Systems, 35:3403–3416,                  Pattern Recognition, pages 406–413. IEEE, 2014. 7
     2022. 2                                                             [52] Nianjuan Jiang, Zhaopeng Cui, and Ping Tan. A global
[38] Yasutaka Furukawa, Carlos Hernández, et al. Multi-view                  linear method for camera pose registration. In Proceedings
     stereo: A tutorial. Foundations and Trends® in Computer                  of the IEEE international conference on computer vision,
     Graphics and Vision, 9(1-2):1–148, 2015. 2                               pages 481–488, 2013. 13
[39] Silvano Galliani, Katrin Lasinger, and Konrad Schindler.            [53] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi,
     Massively parallel multiview stereopsis by surface normal                Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang
     diffusion. In Proceedings of the IEEE international confer-              Xu. LVSM: a large view synthesis model with minimal 3D
     ence on computer vision, pages 873–881, 2015. 2                          inductive bias. arXiv, 2410.17242, 2024. 4, 9
[40] Silvano Galliani, Katrin Lasinger, and Konrad Schindler.            [54] Yuhe Jin, Dmytro Mishkin, Anastasiia Mishchuk, Jiri
     Massively parallel multiview stereopsis by surface normal                Matas, Pascal Fua, Kwang Moo Yi, and Eduard Trulls. Im-
     diffusion. In ICCV, 2015. 7                                              age matching across wide baselines: From paper to prac-
[41] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch,               tice. International Journal of Computer Vision, 129(2):
     Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapra-                 517–547, 2021. 12
     gasam, Florian Golemo, Charles Herrmann, Thomas Kipf,               [55] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia
     Abhijit Kundu, Dmitry Lagun, Issam Laradji, Hsueh-                       Neverova, Andrea Vedaldi, and Christian Rupprecht. Co-
     Ti (Derek) Liu, Henning Meyer, Yishu Miao, Derek                         tracker3: Simpler and better point tracking by pseudo-
     Nowrouzezahrai, Cengiz Oztireli, Etienne Pot, Noha Rad-                  labelling real videos. arXiv preprint arXiv:2410.11831,
     wan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi,                    2024. 2
     Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun,             [56] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia
     Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi,                        Neverova, Andrea Vedaldi, and Christian Rupprecht. Co-
     Fangcheng Zhong, and Andrea Tagliasacchi. Kubric: a                      tracker: It is better to track together. Proc. ECCV, 2024. 2,
     scalable dataset generator. In Proc. CVPR, 2022. 6, 10                   8, 10


                                                                    15
[57] Nikita Karaev, Ignacio Rocco, Ben Graham, Natalia                          European Conference on Computer Vision (ECCV), 2020.
     Neverova, Andrea Vedaldi, and Christian Rupprecht. Co-                     6
     Tracker: It is better to track together. In Proceedings of the        [72] Zeyu Ma, Zachary Teed, and Jia Deng. Multiview stereo
     European Conference on Computer Vision (ECCV), 2024.                       with cascaded epipolar raft. In European Conference on
     3, 5, 6, 9, 10                                                             Computer Vision, pages 734–750. Springer, 2022. 2
[58] Alex Kendall and Roberto Cipolla. Modelling uncertainty               [73] Pierre Moulon, Pascal Monasse, and Renaud Marlet.
     in deep learning for camera relocalization. In Proc. ICRA.                 Global fusion of relative motions for robust, accurate and
     IEEE, 2016. 5                                                              scalable structure from motion. In Proceedings of the IEEE
[59] Alex Kendall and Yarin Gal. What uncertainties do we                       international conference on computer vision, pages 3248–
     need in Bayesian deep learning for computer vision? Proc.                  3255, 2013. 13
     NeurIPS, 2017. 6                                                      [74] Michael Niemeyer, Lars Mescheder, Michael Oechsle,
[60] Guillaume Le Moing, Jean Ponce, and Cordelia Schmid.                       and Andreas Geiger. Differentiable volumetric rendering:
     Dense optical tracking: Connecting the dots. In CVPR,                      Learning implicit 3d representations without 3d supervi-
     2024. 2                                                                    sion. In Proceedings of the IEEE/CVF conference on com-
[61] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua.                   puter vision and pattern recognition, pages 3504–3515,
     Ep n p: An accurate o (n) solution to the p n p problem. In-               2020. 2
     ternational journal of computer vision, 81:155–166, 2009.             [75] David Novotný, Diane Larlus, and Andrea Vedaldi. Learn-
     3                                                                          ing 3D object categories by looking around them. In Pro-
[62] Vincent Leroy, Yohann Cabon, and Jérôme Revaud.                          ceedings of the International Conference on Computer Vi-
     Grounding image matching in 3d with mast3r. arXiv                          sion (ICCV), 2017. 6
     preprint arXiv:2406.09756, 2024. 2, 7, 12, 13                         [76] David Novotný, Diane Larlus, and Andrea Vedaldi. Captur-
[63] Hongyang Li, Hao Zhang, Shilong Liu, Zhaoyang Zeng,                        ing the geometry of object categories from video supervi-
     Tianhe Ren, Feng Li, and Lei Zhang. Taptr: Tracking                        sion. IEEE Transactions on Pattern Analysis and Machine
     any point with transformers as detection. arXiv preprint                   Intelligence, 2018. 5
     arXiv:2403.13042, 2024. 2, 10                                         [77] John Oliensis. A critique of structure-from-motion algo-
[64] Zhengqi Li and Noah Snavely. Megadepth: Learning                           rithms. Computer Vision and Image Understanding, 80(2):
     single-view depth prediction from internet photos. In Pro-                 172–214, 2000. 2
     ceedings of the IEEE conference on computer vision and                [78] Maxime Oquab, Timothée Darcet, Théo Moutakanni,
     pattern recognition, pages 2041–2050, 2018. 6                              Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fer-
[65] Amy Lin, Jason Y Zhang, Deva Ramanan, and Shubham                          nandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-
     Tulsiani. Relpose++: Recovering 6d poses from sparse-                      Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba,
     view observations. arXiv preprint arXiv:2305.04926, 2023.                  Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan
     13                                                                         Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve,
[66] Philipp Lindenberger, Paul-Edouard Sarlin, Viktor Larsson,                 Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Ar-
     and Marc Pollefeys. Pixel-perfect structure-from-motion                    mand Joulin, and Piotr Bojanowski. DINOv2: Learning
     with featuremetric refinement. arXiv.cs, abs/2108.08291,                   robust visual features without supervision. Transactions on
     2021. 7, 12                                                                Machine Learning Research, 2024. 2, 4, 11
[67] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc                   [79] Onur Ozyesil and Amit Singer. Robust camera location es-
     Pollefeys. Lightglue: Local feature matching at light speed.               timation by convex programming. In Proceedings of the
     arXiv preprint arXiv:2306.13643, 2023. 2                                   IEEE Conference on Computer Vision and Pattern Recog-
[68] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc                        nition, pages 2674–2683, 2015. 13
     Pollefeys. LightGlue: local feature matching at light speed.          [80] Onur Özyeşil, Vladislav Voroninski, Ronen Basri, and
     In Proc. ICCV, 2023. 8                                                     Amit Singer. A survey of structure from motion*. Acta
[69] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,                    Numerica, 26:305–364, 2017. 2
     Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,                   [81] Linfei Pan, Daniel Barath, Marc Pollefeys, and Jo-
     et al. Dl3dv-10k: A large-scale scene dataset for deep                     hannes Lutz Schönberger. Global Structure-from-Motion
     learning-based 3d vision. In Proceedings of the IEEE/CVF                   Revisited. In European Conference on Computer Vision
     Conference on Computer Vision and Pattern Recognition,                     (ECCV), 2024. 13
     pages 22160–22169, 2024. 6                                            [82] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Pe-
[70] Shaohui Liu, Yidan Gao, Tianyi Zhang, Rémi Pautrat, Jo-                   ters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard
     hannes L Schönberger, Viktor Larsson, and Marc Pollefeys.                 Newcombe, and Yuheng (Carl) Ren. Aria digital twin: A
     Robust incremental structure-from-motion with hybrid fea-                  new benchmark dataset for egocentric 3d machine percep-
     tures. In European Conference on Computer Vision, pages                    tion. In Proceedings of the IEEE/CVF International Con-
     249–269. Springer, 2025. 2                                                 ference on Computer Vision (ICCV), pages 20133–20143,
[71] Manuel Lopez-Antequera, Pau Gargallo, Markus Hofinger,                     2023. 6
     Samuel Rota BulÃ², Yubin Kuang, and Peter Kontschieder.              [83] William Peebles and Saining Xie. Scalable diffusion mod-
     Mapillary planet-scale depth dataset. In Proceedings of the                els with transformers. In Proceedings of the IEEE/CVF In-


                                                                      16
     ternational Conference on Computer Vision, pages 4195–                      unstructured multi-view stereo. In Computer Vision–ECCV
     4205, 2023. 12                                                              2016: 14th European Conference, Amsterdam, The Nether-
[84] Rui Peng, Rongjie Wang, Zhenyu Wang, Yawen Lai, and                         lands, October 11-14, 2016, Proceedings, Part III 14, pages
     Ronggang Wang. Rethinking depth estimation for multi-                       501–518. Springer, 2016. 2
     view stereo: A unified representation. In Proceedings of               [97] Thomas Schops, Johannes L Schonberger, Silvano Galliani,
     the IEEE/CVF conference on computer vision and pattern                      Torsten Sattler, Konrad Schindler, Marc Pollefeys, and An-
     recognition, pages 8645–8654, 2022. 2                                       dreas Geiger. A multi-view stereo benchmark with high-
[85] Luis Pineda, Taosha Fan, Maurizio Monge, Shobha                             resolution images and multi-camera videos. In Proceed-
     Venkataraman, Paloma Sodhi, Ricky TQ Chen, Joseph Or-                       ings of the IEEE conference on computer vision and pattern
     tiz, Daniel DeTone, Austin Wang, Stuart Anderson, et al.                    recognition, pages 3260–3269, 2017. 7, 8
     Theseus: A library for differentiable nonlinear optimiza-              [98] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
     tion. Advances in Neural Information Processing Systems,                    Pradeep Ramani, and Tri Dao. Flashattention-3: Fast
     35:3801–3818, 2022. 10                                                      and accurate attention with asynchrony and low-precision.
[86] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya                          Advances in Neural Information Processing Systems, 37:
     Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,                       68658–68685, 2024. 10
     Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen                    [99] Yan Shi, Jun-Xiong Cai, Yoli Shavit, Tai-Jiang Mu, Wensen
     Krueger, and Ilya Sutskever. Learning transferable visual                   Feng, and Kai Zhang. Clustergnn: Cluster-based coarse-
     models from natural language supervision. In Proc. ICML,                    to-fine graph neural network for efficient feature matching.
     pages 8748–8763, 2021. 2                                                    In Proceedings of the IEEE/CVF Conference on Computer
[87] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-                   Vision and Pattern Recognition, pages 12517–12526, 2022.
     sion transformers for dense prediction. In Proceedings of                   2
     the IEEE/CVF international conference on computer vi-                 [100] Samarth Sinha, Jason Y Zhang, Andrea Tagliasacchi, Igor
     sion, pages 12179–12188, 2021. 3, 5, 11                                     Gilitschenski, and David B Lindell. Sparsepose: Sparse-
[88] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler,                      view camera pose regression and refinement. In Proceed-
     Luca Sbordone, Patrick Labatut, and David Novotny. Com-                     ings of the IEEE/CVF Conference on Computer Vision and
     mon Objects in 3D: Large-scale learning and evaluation of                   Pattern Recognition, pages 21349–21359, 2023. 13
     real-life 3D category reconstruction. In Proc. ICCV, 2021.
                                                                           [101] Cameron Smith, David Charatan, Ayush Tewari, and Vin-
     6, 7
                                                                                 cent Sitzmann. Flowmap: High-quality camera poses, in-
[89] Mike Roberts, Jason Ramapuram, Anurag Ranjan, At-
                                                                                 trinsics, and depth via gradient descent. arXiv preprint
     ulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ
                                                                                 arXiv:2404.15259, 2024. 13
     Webb, and Joshua M. Susskind. Hypersim: A photorealis-
                                                                           [102] Cameron Smith, David Charatan, Ayush Tewari, and Vin-
     tic synthetic dataset for holistic indoor scene understanding.
                                                                                 cent Sitzmann. FlowMap: high-quality camera poses, in-
     In International Conference on Computer Vision (ICCV)
                                                                                 trinsics, and depth via gradient descent. arXiv, 2404.15259,
     2021, 2021. 6
                                                                                 2024. 2
[90] Rother. Linear multiview reconstruction of points, lines,
     planes and cameras using a reference plane. In Proceedings            [103] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo
     Ninth IEEE International Conference on Computer Vision,                     tourism: exploring photo collections in 3d. In ACM sig-
     pages 1210–1217. IEEE, 2003. 13                                             graph 2006 papers, pages 835–846. 2006. 2, 13
[91] Peter Sand and Seth Teller. Particle video: Long-range mo-            [104] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen,
     tion estimation using point trajectories. IJCV, 80, 2008. 2                 Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-
[92] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,                     Artal, Carl Ren, Shobhit Verma, et al. The replica
     and Andrew Rabinovich. Superglue: Learning feature                          dataset: A digital replica of indoor spaces. arXiv preprint
     matching with graph neural networks. In Proceedings of                      arXiv:1906.05797, 2019. 6
     the IEEE/CVF conference on computer vision and pattern                [105] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and
     recognition, pages 4938–4947, 2020. 2, 7                                    Xiaowei Zhou. Loftr: Detector-free local feature matching
[93] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,                     with transformers. In Proceedings of the IEEE/CVF con-
     and Andrew Rabinovich. SuperGlue: learning feature                          ference on computer vision and pattern recognition, pages
     matching with graph neural networks. In Proc. CVPR,                         8922–8931, 2021. 7, 8, 12
     2020. 8                                                               [106] Chris Sweeney, Torsten Sattler, Tobias Hollerer, Matthew
[94] Johannes Lutz Schönberger and Jan-Michael Frahm.                           Turk, and Marc Pollefeys. Optimizing the viewing graph
     Structure-from-motion revisited. In Conference on Com-                      for structure-from-motion. In Proceedings of the IEEE in-
     puter Vision and Pattern Recognition (CVPR), 2016. 2, 12,                   ternational conference on computer vision, pages 801–809,
     13                                                                          2015. 13
[95] Johannes Lutz Schönberger and Jan-Michael Frahm.                     [107] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wi-
     Structure-from-motion revisited. In Proc. CVPR, 2016. 3,                    jmans, Yili Zhao, John Turner, Noah Maestre, Mustafa
     6                                                                           Mukadam, Devendra Chaplot, Oleksandr Maksymets,
[96] Johannes L Schönberger, Enliang Zheng, Jan-Michael                         Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur,
     Frahm, and Marc Pollefeys. Pixelwise view selection for                     Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt


                                                                      17
      Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and                  and Illia Polosukhin. Attention is all you need. Advances
      Dhruv Batra. Habitat 2.0: Training home assistants to re-                 in neural information processing systems, 30, 2017. 12
      arrange their habitat. In Advances in Neural Information            [121] Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Pablo
      Processing Systems (NeurIPS), 2021. 6                                     Speciale, and Marc Pollefeys. Patchmatchnet: Learned
[108] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea                   multi-view patchmatch stereo. In CVPR, pages 14194–
      Vedaldi. Splatter image: Ultra-fast single-view 3d recon-                 14203, 2021. 7
      struction. In Proceedings of the IEEE/CVF conference                [122] Jianyuan Wang, Yiran Zhong, Yuchao Dai, Stan Birchfield,
      on computer vision and pattern recognition, pages 10208–                  Kaihao Zhang, Nikolai Smolyanskiy, and Hongdong Li.
      10217, 2024. 9                                                            Deep two-view structure-from-motion revisited. In Pro-
[109] Chengzhou Tang and Ping Tan. Ba-net: Dense bundle ad-                     ceedings of the IEEE/CVF conference on Computer Vision
      justment network. arXiv preprint arXiv:1806.04807, 2018.                  and Pattern Recognition, pages 8953–8962, 2021. 2, 13
      2, 13
                                                                          [123] Jianyuan Wang, Christian Rupprecht, and David Novotny.
[110] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei
                                                                                Posediffusion: Solving pose estimation via diffusion-aided
      Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-
                                                                                bundle adjustment. In Proceedings of the IEEE/CVF In-
      view gaussian model for high-resolution 3d content cre-
                                                                                ternational Conference on Computer Vision, pages 9773–
      ation. In European Conference on Computer Vision, pages
                                                                                9783, 2023. 13
      1–18. Springer, 2024. 9
[111] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu,                  [124] Jianyuan Wang, Christian Rupprecht, and David Novotny.
      Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan.                       PoseDiffusion: solving pose estimation via diffusion-aided
      Mv-dust3r+: Single-stage scene reconstruction from sparse                 bundle adjustment. In Proc. ICCV, 2023. 6, 7
      views in 2 seconds. arXiv preprint arXiv:2412.06974,                [125] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and
      2024. 2, 7                                                                David Novotny. VGGSfM: visual geometry grounded deep
[112] Zachary Teed and Jia Deng. Deepv2d: Video to depth                        structure from motion. In Proc. CVPR, 2024. 1, 2, 3, 6, 7,
      with differentiable structure from motion. arXiv preprint                 10, 12, 13
      arXiv:1812.04605, 2018. 2, 13                                       [126] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan,
[113] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam                   Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai
      for monocular, stereo, and rgb-d cameras. Advances in                     Zhang. PF-LRM: pose-free large reconstruction model for
      neural information processing systems, 34:16558–16569,                    joint pose and shape prediction. arXiv.cs, abs/2311.12024,
      2021. 2, 13                                                               2023. 9
[114] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco              [127] Qianqian Wang, Yifei Zhang, Aleksander Holynski,
      Massa, Alexandre Sablayrolles, and Hervé Jégou. Training                Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d
      data-efficient image transformers & distillation through at-              perception model with persistent state, 2025. 2, 7
      tention. In International conference on machine learning,           [128] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xi-
      pages 10347–10357. PMLR, 2021. 12                                         ang, Yu Deng, Xin Tong, and Jiaolong Yang. MoGe: un-
[115] Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles,                      locking accurate monocular geometry estimation for open-
      Gabriel Synnaeve, and Hervé Jégou. Going deeper with                    domain images with optimal training supervision. arXiv,
      image transformers. In Proceedings of the IEEE/CVF in-                    2410.19115, 2024. 2
      ternational conference on computer vision, pages 32–42,
                                                                          [129] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
      2021. 11, 13
                                                                                Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D
[116] Michał Tyszkiewicz, Pascal Fua, and Eduard Trulls. Disk:
                                                                                vision made easy. In Proc. CVPR, 2024. 1, 2, 3, 4, 6, 7, 11,
      Learning local features with policy gradient. Advances in
                                                                                12, 13
      Neural Information Processing Systems, 33:14254–14265,
      2020. 2                                                             [130] Yuesong Wang, Zhaojie Zeng, Tao Guan, Wei Yang, Zhuo
[117] Shinji Umeyama. Least-squares estimation of transforma-                   Chen, Wenkai Liu, Luoyuan Xu, and Yawei Luo. Adap-
      tion parameters between two point patterns. IEEE Trans.                   tive patch deformation for textureless-resilient multi-view
      Pattern Anal. Mach. Intell., 13(4), 1991. 8                               stereo. In Proceedings of the IEEE/CVF Conference on
[118] Benjamin Ummenhofer, Huizhong Zhou, Jonas Uhrig,                          Computer Vision and Pattern Recognition (CVPR), pages
      Nikolaus Mayer, Eddy Ilg, Alexey Dosovitskiy, and                         1621–1630, 2023. 2
      Thomas Brox. Demon: Depth and motion network for                    [131] Xingkui Wei, Yinda Zhang, Zhuwen Li, Yanwei Fu, and
      learning monocular stereo. In Proceedings of the IEEE con-                Xiangyang Xue. Deepsfm: Structure from motion via deep
      ference on computer vision and pattern recognition, pages                 bundle adjustment. In Computer Vision–ECCV 2020: 16th
      5038–5047, 2017. 2, 13                                                    European Conference, Glasgow, UK, August 23–28, 2020,
[119] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob                          Proceedings, Part I 16, pages 230–247. Springer, 2020. 2,
      Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser,                    13
      and Illia Polosukhin. Attention is all you need. In Proc.           [132] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan,
      NeurIPS, 2017. 2, 4                                                       Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zex-
[120] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob                          iang Xu. MeshLRM: large reconstruction model for high-
      Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser,                     quality mesh. arXiv, 2404.12385, 2024. 4


                                                                     18
[133] Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen                        computer vision and pattern recognition, pages 1790–1799,
      Lu, and Jie Zhou. Nerfingmvs: Guided optimization of neu-                  2020. 6
      ral radiance fields for indoor multi-view stereo. In Proceed-        [147] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan
      ings of the IEEE/CVF International Conference on Com-                      Atzmon, Basri Ronen, and Yaron Lipman. Multiview neu-
      puter Vision (ICCV), pages 5610–5619, 2021. 2                              ral surface reconstruction by disentangling geometry and
[134] Changchang Wu. Towards linear-time incremental struc-                      appearance. Advances in Neural Information Processing
      ture from motion. In 2013 International Conference on 3D                   Systems, 33:2492–2502, 2020. 2
      Vision-3DV 2013, pages 127–134. IEEE, 2013. 2, 13                    [148] Gokul Yenduri, Ramalingam M, Chemmalar Selvi G.,
[135] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang.                        Supriya Y, Gautam Srivastava, Praveen Kumar Reddy
      Rgbd objects in the wild: Scaling real-world 3d object                     Maddikunta, Deepti Raj G, Rutvij H. Jhaveri, Prabadevi
      learning from rgb-d videos, 2024. 6                                        B, Weizheng Wang, Athanasios V. Vasilakos, and
[136] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue,                        Thippa Reddy Gadekallu. Generative pre-trained trans-
      Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker:                   former: A comprehensive review on enabling technologies,
      Tracking any 2d pixels in 3d space. In Proceedings of                      potential applications, emerging challenges, and future di-
      the IEEE/CVF Conference on Computer Vision and Pattern                     rections. arXiv.cs, abs/2305.10435, 2023. 2
      Recognition, pages 20406–20417, 2024. 9                              [149] Kwang Moo Yi, Eduard Trulls, Vincent Lepetit, and Pascal
[137] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar,                       Fua. LIFT: Learned Invariant Feature Transform. In Proc.
      Jose M Alvarez, and Ping Luo. Segformer: Simple and                        ECCV, 2016. 2
      efficient design for semantic segmentation with transform-           [150] Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan
      ers. Advances in neural information processing systems,                    Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and
      34:12077–12090, 2021. 12                                                   Joshua M Susskind. Stabilizing transformer training by pre-
[138] Fei Xie, Chunyu Wang, Guangting Wang, Yue Cao,                             venting attention entropy collapse. In International Confer-
      Wankou Yang, and Wenjun Zeng. Correlation-aware deep                       ence on Machine Learning, pages 40770–40803. PMLR,
      tracking. In Proceedings of the IEEE/CVF conference on                     2023. 13
      computer vision and pattern recognition, pages 8751–8760,            [151] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jam-
      2022. 13                                                                   pani, Trevor Darrell, Forrester Cole, Deqing Sun, and
[139] Qingshan Xu and Wenbing Tao. Learning inverse depth re-                    Ming-Hsuan Yang. MonST3R: a simple approach for
      gression for multi-view stereo with correlation cost volume.               estimating geometry in the presence of motion. arXiv,
      In AAAI, 2020. 7                                                           2410.03825, 2024. 11
[140] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen,                    [152] Jason Y Zhang, Deva Ramanan, and Shubham Tulsiani.
      Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wet-                        Relpose: Predicting probabilistic relative rotation for sin-
      zstein. GRM: Large gaussian reconstruction model for effi-                 gle objects in the wild. In ECCV, pages 592–611. Springer,
      cient 3D reconstruction and generation. arXiv, 2403.14621,                 2022. 13
      2024. 9                                                              [153] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan
[141] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael                         Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as
      Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier,                    rays: Pose estimation via ray diffusion. In International
      and Matt Feiszli. Fast3r: Towards 3d reconstruction                        Conference on Learning Representations (ICLR), 2024. 13
      of 1000+ images in one forward pass. arXiv preprint                  [154] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan
      arXiv:2501.13928, 2025. 2, 7, 10                                           Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large
[142] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Ji-                     reconstruction model for 3d gaussian splatting. In European
      ashi Feng, and Hengshuang Zhao. Depth anything: Un-                        Conference on Computer Vision, pages 1–19. Springer,
      leashing the power of large-scale unlabeled data. In Proc.                 2024. 9
      CVPR, 2024. 2                                                        [155] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan
[143] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiao-                     Zhao, Kalyan Sunkavalli, and Zexiang Xu. GS-LRM: large
      gang Xu, Jiashi Feng, and Hengshuang Zhao. Depth any-                      reconstruction model for 3D Gaussian splatting. arXiv,
      thing v2. arXiv:2406.09414, 2024. 11                                       2404.19702, 2024. 9
[144] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long                   [156] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue,
      Quan. Mvsnet: Depth inference for unstructured multi-                      Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gor-
      view stereo. In ECCV, 2018. 7                                              don Wetzstein. Flare: Feed-forward geometry, appear-
[145] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long                         ance and camera estimation from uncalibrated sparse views,
      Quan. Mvsnet: Depth inference for unstructured multi-                      2025. 2, 7
      view stereo. In Proceedings of the European conference               [157] Zhe Zhang, Rui Peng, Yuxi Hu, and Ronggang Wang. Ge-
      on computer vision (ECCV), pages 767–783, 2018. 2                          omvsnet: Learning multi-view stereo with geometry per-
[146] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan                       ception. In CVPR, 2023. 2, 7
      Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs:                 [158] Xiaoming Zhao, Xingming Wu, Weihai Chen, Peter CY
      A large-scale dataset for generalized multi-view stereo net-               Chen, Qingsong Xu, and Zhengguo Li. Aliked: A lighter
      works. In Proceedings of the IEEE/CVF conference on                        keypoint and descriptor extraction network via deformable


                                                                      19
      transformation. IEEE Transactions on Instrumentation and
      Measurement, 72:1–16, 2023. 8
[159] Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wet-
      zstein, and Leonidas J. Guibas. Pointodyssey: A large-scale
      synthetic dataset for long-term point tracking. In ICCV,
      2023. 6
[160] Tinghui Zhou, Matthew Brown, Noah Snavely, and
      David G Lowe. Unsupervised learning of depth and ego-
      motion from video. In Proceedings of the IEEE conference
      on computer vision and pattern recognition, pages 1851–
      1858, 2017. 2, 13
[161] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
      and Noah Snavely.        Stereo magnification: Learning
      view synthesis using multiplane images. arXiv preprint
      arXiv:1805.09817, 2018. 6, 7




                                                                    20
