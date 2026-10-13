# SatFIN: Deep Optical-Flow Frame Interpolation for Enhancing the Temporal Resolution of Geostationary Satellite Imagery

Riddhesh Morankar, Abhinav Nigade, Virajas Khapli, Atharva Pednekar, Preet Jain
Mentor: Dr. Jayshree Aher

**Abstract**—Geostationary satellites image the Earth every 10–30 minutes, a cadence too coarse to follow rapidly evolving phenomena such as convective initiation, overshooting tops, wildfire smoke plumes, and flash-flood–producing storms. This paper presents SatFIN, a Satellite Frame Interpolation Network that synthesizes physically plausible intermediate frames between consecutive geostationary observations using deep bidirectional optical flow. SatFIN adapts the real-time intermediate flow estimation (RIFE) paradigm to brightness-temperature imagery: a three-scale IFNet predicts time-conditioned flows and a fusion mask, the input frames are backward-warped to an arbitrary target time t, and a compact U-Net adds a residual correction. The 1.14 M-parameter network is trained on ~1-minute GOES-19 Advanced Baseline Imager (ABI) mesoscale sequences, where temporal sparsity is simulated by sampling frame triplets with 2–10-minute gaps, so that real intermediate frames serve as ground truth. A convective-region-weighted Charbonnier loss emphasizes cold cloud tops (brightness temperature below 235 K) and pixels with rapid temporal change. We describe the full open pipeline: anonymous ingestion from AWS, radiance- and count-to-brightness-temperature conversion for GOES and Himawari, training on a single laptop GPU (~35 min), tiled full-disk inference, a near-real-time Himawari-9 service, and an interactive dashboard. On a held-out GOES-19 day and sector, SatFIN outperforms linear blending and the classical Farnebäck and DIS optical-flow baselines at every gap (2, 4, and 10 min) and on every metric, reaching 47.57 dB PSNR and 0.424 K mean absolute error overall versus 44.08 dB and 0.574 K for DIS. On Himawari-9 data from a different satellite, which it never saw in training, SatFIN is the best method at 5-minute gaps (45.67 dB vs. 43.82 dB for DIS), but at 10-minute gaps DIS is slightly more accurate (40.73 vs. 39.68 dB), which identifies large displacements as the main remaining weakness.

**Index Terms**—Frame interpolation, optical flow, deep learning, geostationary satellites, GOES-R ABI, Himawari AHI, temporal super-resolution, nowcasting, convective storms.

## I. Introduction

Geostationary (GEO) meteorological satellites are the backbone of real-time weather monitoring because they observe the same hemisphere continuously. Operational full-disk imaging, however, is typically refreshed only every 10 minutes (GOES-R ABI, Himawari AHI) or every 30 minutes (INSAT-3D/3DR/3DS) [1]–[3]. Many hazardous processes evolve faster than this: a convective cell can go from initiation to a mature, lightning-producing storm in well under an hour, and the cloud-top cooling rates used as convective-initiation predictors are defined over time intervals of minutes [4], [5]. Rapid-scan sectors can image at 30–150-s cadence, but only over small, pre-selected regions, and increasing the cadence of full-disk scans requires new hardware.

Video frame interpolation (VFI) offers a software alternative: given two consecutive frames I0 and I1, synthesize the frame It at an intermediate time t ∈ (0, 1). Flow-based VFI methods such as Super SloMo [6] and RIFE [7] estimate motion between the input frames, warp them toward time t, and fuse the results. These models are trained on natural video, where motion is mostly rigid and appearance changes little between frames. Satellite imagery breaks both assumptions: clouds deform, grow, and dissipate; brightness temperature (BT) is a physical quantity whose absolute value matters; and the most important regions (deep convective cores) are exactly those whose appearance changes fastest.

This paper presents SatFIN, a VFI framework designed specifically for GEO imagery. Its contributions are:

- A training strategy based on simulated temporal sparsity, which draws (I0, It, I1) triplets with configurable gaps from ~1-minute GOES mesoscale sequences, so that real observed frames serve as ground truth for 2–10-minute interpolation.
- A compact (1.14 M parameters) RIFE-style architecture adapted to brightness temperature, with time-conditioned bidirectional flow, backward warping, in-network standardization of low-contrast BT fields, and residual U-Net refinement.
- A convective-region-weighted loss that up-weights cold cloud tops and pixels with large temporal BT change, the regions most relevant for nowcasting.
- An end-to-end, open pipeline covering data ingestion for GOES and Himawari, preprocessing, training, evaluation against classical baselines with physically meaningful metrics (MAE in kelvin, overall and over cold cloud), tiled full-disk inference, a near-real-time Himawari-9 service, and an interactive dashboard.
- An evaluation on a held-out GOES-19 day and a cross-satellite test on Himawari-9.

The rest of the paper is organized as follows. Section II reviews related work. Section III describes the data and preprocessing. Section IV presents the SatFIN model and loss function. Section V gives the experimental setup, Section VI the results, and Section VII the deployment pipeline. Section VIII discusses limitations and Section IX concludes.

## II. Related Work

### A. Video Frame Interpolation

Early VFI methods combined classical optical flow, such as Farnebäck's polynomial-expansion method [8] or Dense Inverse Search (DIS) [9], with image warping. Super SloMo [6] introduced end-to-end training of bidirectional flows and visibility maps that can produce frames at arbitrary t. Learned flow estimators such as PWC-Net [10] and RAFT [11] improved motion accuracy but add significant cost. RIFE [7] showed that an intermediate flow network (IFNet) can directly estimate flows from time t to both inputs in a coarse-to-fine manner, reaching real-time speed with competitive quality. SatFIN builds on this design because the computational budget of full-disk and real-time inference makes efficiency essential.

### B. Optical Flow in Satellite Meteorology

Atmospheric motion vectors derived from feature tracking in successive GEO images are a long-standing operational product, and cloud tracking underlies satellite-based convective-initiation nowcasting [4], [5]. These methods estimate motion for diagnostic purposes rather than synthesizing new imagery.

### C. Temporal Interpolation of Satellite Imagery

Vandal and Nemani [12] showed that deep optical-flow interpolation can generate intermediate GOES-16 ABI frames that outperform linear interpolation, which confirms that learned VFI transfers to GEO data. SatFIN extends this line of work with a compact real-time architecture, arbitrary-t inference, a convection-focused objective, a cross-satellite evaluation, and a working near-real-time service.

## III. Data and Preprocessing

### A. Data Sources

Table I summarizes the sensors used. For training, validation, and testing we use GOES-19 (GOES-East) ABI Level-1b radiances (product ABI-L1b-RadM) from the two mesoscale sectors, which are distributed through the NOAA Open Data Dissemination program on Amazon S3 (bucket noaa-goes19) and can be read anonymously. Each mesoscale sector covers roughly 1000 km × 1000 km at ~60-s cadence, and the 2-km IR bands give 500 × 500-pixel frames. Band 13 (10.3 µm, "clean" longwave IR) is used because it is available day and night and responds directly to cloud-top temperature.

Himawari-9 AHI data (bucket noaa-himawari9, Himawari Standard Data format) are used only for testing and deployment, never for training. The AHI Target sector is a 500 × 500-pixel area imaged every 2.5 min, so it also provides real intermediate frames; the full disk (FLDK) is imaged every 10 min in 10 segments.

**TABLE I — Satellite Sensors Used in This Work**

| Satellite / sensor | Role | Cadence | IR res. |
|---|---|---|---|
| GOES-19 / ABI (East), mesoscale M1/M2 | Training, validation, test | ~1 min | 2 km |
| Himawari-9 / AHI, Target | Cross-satellite test, live service | 2.5 min | 2 km |
| Himawari-9 / AHI, full disk | Tiled full-disk inference | 10 min | 2 km |
| INSAT-3DS / Imager | Planned deployment (Section VII-D) | 30 min | 4 km |

### B. Conversion to Brightness Temperature

ABI L1b files store spectral radiance L. We convert it to brightness temperature using the inverse Planck function with the band-specific coefficients distributed in each file (planck_fk1, planck_fk2, planck_bc1, planck_bc2) [13]:

    TB = [ fk2 / ln( fk1 / L + 1 ) − bc1 ] / bc2    (1)

Himawari files store raw counts; these are converted with the calibration block of each file (count → radiance → effective temperature → corrected BT).

### C. Normalization and Quality Control

BT is normalized to [0, 1] with fixed physical bounds rather than per-image statistics, so that a given normalized value always corresponds to the same temperature across scenes and sensors:

    x = clip( (TB − Tmin) / (Tmax − Tmin), 0, 1 )    (2)

with Tmin = 180 K and Tmax = 330 K. Invalid pixels (GOES data-quality flag ≥ 2, Himawari error or off-disk counts) are filled with the frame mean. A new sequence is started whenever the gap between scans exceeds 1.5 × the median cadence or the sector moves. The Himawari Target window jitters by about 5 pixels between scans; every frame is therefore cropped to the common fixed-grid area (460 × 460 pixels), so that the jitter does not appear as spurious motion.

### D. Simulated Temporal Sparsity

The key idea of the training strategy is to create low-cadence inputs from high-cadence sequences. From a 1-minute sequence we pick a gap Δ ∈ {2, 4, 10} minutes, take I0 and I1 Δ minutes apart, and choose one of the Δ − 1 real intermediate frames as the target It. Because acquisition times are not exactly regular, t is computed from the actual scan timestamps, t = (τk − τ0) / (τ1 − τ0), and is stored with each sample and fed to the network.

### E. Splits and Augmentation

Adjacent satellite frames are highly correlated, so random frame-level splits would leak information between training and test sets. We therefore split by date (Table II): training, validation, and test sets come from disjoint days, and the test day also uses a different mesoscale sector. During training we apply random 256 × 256 crops, horizontal and vertical flips, 90° rotations, and time reversal (swapping I0 and I1 and replacing t with 1 − t).

**TABLE II — Data Splits (GOES-19, Band 13)**

| Split | Date (2025) | Sector | Frames | Triplets |
|---|---|---|---|---|
| Train | Jun 1, 2, 3, 18–21 UTC | M1 | 3 × 180 | 6708 |
| Validation | Jun 5, 18–20 UTC | M1 (35.5°N, 101.0°W) | 120 | 1456 |
| Test | Jun 10, 18–20 UTC | M2 (32.6°N, 102.0°W) | 120 | 1456 |
| Cross-satellite test | Jun 1, 03–05 UTC | Himawari-9 Target | — | 146 evaluated |

## IV. Proposed Method: SatFIN

### A. Overview

Fig. 1 shows the complete pipeline and Fig. 2 the network. Given (I0, I1, t), SatFIN (i) estimates flows from the target time to both inputs together with a fusion mask, (ii) backward-warps the inputs to time t, (iii) blends them, and (iv) adds a learned residual correction. All four stages are trained jointly.

*Fig. 1. End-to-end SatFIN pipeline. (a) Training on ~1-minute GOES-19 mesoscale data with simulated temporal sparsity. (b) Inference on lower-cadence data: tiled full-disk processing, the near-real-time Himawari-9 service, and the dashboard.*

Real BT scenes occupy only a thin slice of the normalized range (typical standard deviation ≈ 0.03). Inside the network, both inputs are therefore standardized with their joint mean and standard deviation; the transformation is undone on the output. Without this step, training made almost no progress. Inputs are padded by edge replication to a multiple of 16 pixels.

### B. Intermediate Flow Estimation (IFNet)

IFNet is a three-block coarse-to-fine estimator whose blocks operate at 1/16, 1/8, and 1/4 of the input resolution (downsampling factors 4, 2, 1 followed by two stride-2 convolutions), with widths 96, 64, and 48. The first block receives I0, I1, and the time map t (broadcast as a constant channel). Each later block also receives the frames warped with the current flow, the current mask, and the current flow, and it predicts a residual update:

    (Ft→0, Ft→1, M) = IFNet(I0, I1, t)    (3)

where Ft→0 and Ft→1 are dense 2-D flow fields and M ∈ [0, 1] (sigmoid) is a per-pixel fusion mask that can down-weight occluded or newly formed cloud in one of the inputs.

### C. Backward Warping

Each input is warped to time t with differentiable bilinear sampling [14] (PyTorch grid_sample):

    Ĩ0(p) = I0( p + Ft→0(p) ),   Ĩ1(p) = I1( p + Ft→1(p) )    (4)

and the warped frames are blended with the mask:

    Īt = M ⊙ Ĩ0 + (1 − M) ⊙ Ĩ1    (5)

### D. Refinement Network

Warping alone cannot represent appearance change, such as a cloud top cooling as it rises or an anvil spreading. A compact three-level U-Net [15], R (widths 32, 64, 128, with skip connections), receives the warped frames, the blend, the flows, the mask, and context features c (16 maps per input frame from a small two-layer network, warped to time t with the same flows). It predicts a bounded residual:

    Ît = clip( Īt + tanh R(Ĩ0, Ĩ1, Īt, F, M, c), 0, 1 )    (6)

*Fig. 2. SatFIN architecture. A three-scale IFNet estimates time-conditioned flows and a fusion mask; the inputs are backward-warped and blended, and a refinement U-Net adds a residual correction to produce the intermediate frame Ît.*

The complete model has 1,136,064 parameters: 57 % in IFNet and 43 % in the refinement network.

### E. Loss Function

The reconstruction term uses the Charbonnier penalty ρ(x) = (x² + ε²)^½ with ε = 10⁻³ [16], a differentiable, outlier-robust approximation of the L1 loss. To focus learning on hazardous weather, each pixel is weighted by

    w(p) = 1 + α·1[ min(TB0, TBt, TB1)(p) < 235 K ] + β·1[ |TB1(p) − TB0(p)| > τ ]    (7)

where 1[·] is the indicator function. The first term marks cold cloud tops typical of deep convection in any of the three frames and the second marks pixels that change by more than τ = 5 K across the gap. The convective reconstruction loss is the normalized weighted Charbonnier error:

    Lconv = Σp w(p) ρ( Ît(p) − It(p) ) / Σp w(p)    (8)

A first-order flow-smoothness term Lsmooth (mean absolute flow gradient) regularizes the flows, and an edge term Ledge (L1 distance between the image gradients of Ît and It) preserves sharp cloud edges. The total objective is

    L = Lconv + λs Lsmooth + λe Ledge    (9)

### F. Multispectral Extension

For C-channel input (e.g., ABI bands 8, 10, 13, and 14), the input and output layers are widened to C channels, and a single flow field is shared across channels. Sharing the flow enforces a consistent motion estimate across bands. The model supports this mode and it is covered by unit tests, but the data pipeline currently prepares a single band, so all experiments in this paper use Band 13 only.

## V. Experimental Setup

### A. Implementation Details

SatFIN is implemented in PyTorch [17]. Table III lists the configuration, which is stored in a YAML file so that every experiment is reproducible. Training ran on a single NVIDIA RTX 4060 Laptop GPU (8 GB) at about 10.6 steps/s, taking about 35 minutes for 20,000 steps. The model was validated every 1000 steps on 256 fixed validation triplets, and the checkpoint with the best validation PSNR (step 12,000: 44.65 dB vs. 38.25 dB for linear blending) is used for all results. Before the full run, the pipeline was verified by overfitting 20 fixed triplets.

**TABLE III — Training Configuration**

| Parameter | Value |
|---|---|
| Input / band | 1 ch., ABI Band 13 (10.3 µm) |
| Normalization bounds | 180–330 K |
| Patch size / batch size | 256 × 256 / 8 |
| Training gaps Δ | 2, 4, 10 min |
| Optimizer | AdamW [18], weight decay 10⁻⁴ |
| Learning rate | 3×10⁻⁴ → 0 (cosine), 20,000 steps |
| Precision / grad. clip | Mixed (AMP) / 1.0 |
| Convective weights α, β | 4.0, 2.0 (235 K, τ = 5 K) |
| Loss weights λs, λe | 0.01, 0.1 |
| IFNet widths / scales; U-Net width | 96, 64, 48 / 4, 2, 1; 32 |

### B. Baselines

SatFIN is compared with (i) linear blending, Ît = (1 − t)I0 + t I1, the de facto operational approach, and (ii) classical optical-flow interpolation using the OpenCV implementations of Farnebäck [8] and DIS [9], which estimate forward and backward flow between I0 and I1, approximate the intermediate flows Ft→0 and Ft→1 as in Super SloMo [6], and warp and blend the inputs, with no learning.

### C. Evaluation Metrics

Image quality is measured with the peak signal-to-noise ratio (PSNR) on normalized BT and with the structural similarity index (SSIM) [19]:

    PSNR = 10 log10( 1 / MSE )    (10)

Because PSNR and SSIM do not have physical units, we also report the mean absolute error in kelvin, over all pixels and over cold-cloud pixels only (TB < 235 K in the ground truth):

    MAEK = (Tmax − Tmin) · (1/N) Σp | Ît(p) − It(p) |    (11)

### D. Protocol

For each gap Δ ∈ {2, 4, 10} minutes, 100 evenly spaced test triplets are drawn from the held-out GOES-19 test day and scored on full 500 × 500 frames. The predicted frame is compared against the real 1-minute GOES observation at the same time. For the cross-satellite test, the same checkpoint is applied without any fine-tuning to the Himawari-9 Target sector with gaps of 2 and 4 scans (5 and 10 min). Runtime is the mean time per interpolated frame; SatFIN runs on the GPU and the OpenCV baselines on the CPU.

## VI. Results and Discussion

### A. Quantitative Results on GOES-19

Table IV reports accuracy and runtime for each method and gap on the held-out GOES-19 test day, as measured by the evaluation script (python -m satfin.evaluate). Fig. 3 plots PSNR and MAE against the gap. Every method degrades as the gap grows, because the scene changes more between I0 and I1: linear blending falls from 49.20 dB at 2 min to 35.07 dB at 10 min, and its MAE rises more than fivefold.

**TABLE IV — Interpolation Accuracy and Runtime Versus Temporal Gap (GOES-19 Test Day, Full 500 × 500 Frames)**

| Gap Δ | Method | PSNR (dB)↑ | SSIM↑ | MAE (K)↓ | Cold MAE (K)↓ | ms/frame |
|---|---|---|---|---|---|---|
| 2 min | Linear | 49.20 | 0.9980 | 0.261 | 0.213 | 0.4 |
| | Farnebäck [8] | 46.93 | 0.9966 | 0.383 | 0.257 | 99.2 (CPU) |
| | DIS [9] | 47.27 | 0.9969 | 0.366 | 0.252 | 21.9 (CPU) |
| | **SatFIN** | **53.05** | **0.9992** | **0.200** | **0.138** | 51.7 (GPU) |
| 4 min | Linear | 42.07 | 0.9896 | 0.575 | 0.452 | 0.5 |
| | Farnebäck [8] | 44.63 | 0.9939 | 0.513 | 0.332 | 99.2 (CPU) |
| | DIS [9] | 44.93 | 0.9946 | 0.491 | 0.313 | 22.3 (CPU) |
| | **SatFIN** | **48.37** | **0.9975** | **0.331** | **0.217** | 74.8 (GPU) |
| 10 min | Linear | 35.07 | 0.9435 | 1.431 | 1.084 | 0.6 |
| | Farnebäck [8] | 39.85 | 0.9800 | 0.901 | 0.574 | 99.1 (CPU) |
| | DIS [9] | 40.04 | 0.9817 | 0.865 | 0.525 | 22.7 (CPU) |
| | **SatFIN** | **41.28** | **0.9852** | **0.740** | **0.454** | 75.4 (GPU) |
| All | Linear | 42.12 | 0.9771 | 0.755 | 0.583 | 0.5 |
| | Farnebäck [8] | 43.80 | 0.9902 | 0.599 | 0.388 | 99.2 |
| | DIS [9] | 44.08 | 0.9911 | 0.574 | 0.363 | 22.3 |
| | **SatFIN** | **47.57** | **0.9940** | **0.424** | **0.270** | 67.3 |

*Best value per gap in bold. 100 evenly spaced triplets per gap. Cold MAE: pixels with ground-truth BT < 235 K. The ranking is the same on 256 × 256 center crops (SatFIN 52.51 / 47.99 / 40.73 dB vs. DIS 46.19 / 44.54 / 39.87 dB at 2 / 4 / 10 min).*

*Fig. 3. Accuracy versus temporal gap on the held-out GOES-19 test day. (a) PSNR; (b) MAE in kelvin. SatFIN is the most accurate method at every gap.*

### B. SatFIN Versus the Baselines

SatFIN is the most accurate method at every gap and on every metric. Compared with DIS, the stronger classical baseline, it gains 5.78, 3.44, and 1.24 dB at 2, 4, and 10 min, and it reduces the MAE at 10 min from 0.865 K to 0.740 K (−14 %). Compared with linear blending at 10 min, the MAE is roughly halved (1.431 K → 0.740 K). The gains are largest on cold cloud, the region emphasized by the loss: at 10 min the cold-cloud MAE is 0.454 K versus 0.525 K for DIS and 1.084 K for linear blending.

At 2 min, both classical flow methods are worse than linear blending (47.27 dB for DIS vs. 49.20 dB). At such short gaps the true motion is tiny, and small errors in the explicitly estimated flow outweigh its benefit. SatFIN does not show this failure, because its flow and mask are learned jointly with the reconstruction.

The advantage over DIS shrinks as the gap grows, from 5.8 dB to 1.2 dB. Large displacements are therefore the hardest case for the current model, as the cross-satellite test confirms (Section VI-C).

An earlier checkpoint trained briefly on a CPU (one training day, 128 × 128 crops, ~1.5k steps) was only marginally better than linear blending and clearly worse than DIS at 4 and 10 min. The model architecture is unchanged between the two runs; the gain comes from the full GPU schedule, larger crops, and three times more training data.

### C. Cross-Satellite Generalization (Himawari-9)

Table V applies the GOES-trained checkpoint, without fine-tuning, to the Himawari-9 Target sector, which differs in satellite, calibration, and region.

**TABLE V — Cross-Satellite Test on Himawari-9 Target Sector (Full Frames, No Fine-Tuning)**

| Gap Δ | Method | PSNR (dB)↑ | SSIM↑ | MAE (K)↓ | Cold MAE (K)↓ |
|---|---|---|---|---|---|
| 5 min | Linear | 31.46 | 0.8765 | 2.216 | 1.743 |
| | Farnebäck [8] | 43.77 | 0.9909 | 0.626 | 0.463 |
| | DIS [9] | 43.82 | 0.9913 | 0.621 | 0.454 |
| | **SatFIN** | **45.67** | **0.9943** | **0.483** | **0.311** |
| 10 min | Linear | 28.16 | 0.7733 | 3.464 | 2.961 |
| | Farnebäck [8] | 40.58 | 0.9801 | 0.903 | 0.635 |
| | DIS [9] | **40.73** | **0.9816** | **0.879** | 0.604 |
| | SatFIN | 39.68 | 0.9789 | 0.904 | **0.580** |

At 5 min, SatFIN is the best method on all metrics. At 10 min, DIS is more accurate in PSNR, SSIM, and MAE (by 1.05 dB), while SatFIN still has the lowest cold-cloud MAE. This scene contains much faster motion in pixel terms than the GOES training data: linear blending reaches only 28.16 dB at 10 min here versus 35.07 dB on GOES. The displacements exceed those seen in training, which is the main current weakness of the model.

### D. Runtime

SatFIN needs 50–75 ms per 500 × 500 interpolated frame on the laptop GPU, and a Himawari full disk (5500 × 5500 pixels) is processed in about 10 s per pair of scans with tiling. This is far below the interval between consecutive satellite images, and it allows the near-real-time service described in Section VII-C.

### E. Discussion and Next Steps

The results show that a small learned model, trained for 35 minutes on 9 hours of data, can outperform classical optical flow on unseen data from the same sensor. The priorities are: (i) more training days, seasons, and regions, and a mix of GOES and Himawari data; (ii) larger training gaps and possibly a coarser fourth IFNet level, to learn large displacements; (iii) evaluation at 20–30-minute gaps, which are relevant for full-disk and INSAT-3DS imagery; and (iv) an ablation of the convective-weighted loss. Absolute-error maps in kelvin (Fig. 4) show where each method fails, such as blurring of rapidly growing cores or ghosting at anvil edges.

*Fig. 4. Example from the GOES-19 test day at a 10-minute gap: ground truth and each method (top), absolute error in kelvin (bottom).*

## VII. Deployment

### A. Arbitrary-Cadence Inference

Because SatFIN accepts any t, a sequence at any cadence is densified in a single pass by inserting k frames at t = j / (k + 1), j = 1, …, k, between each pair of observations (e.g., k = 1 for 10 → 5 min, k = 3 for 30 → 7.5 min). All t values for a pair are processed in one batch, and the single-pass mode avoids accumulating errors from predicted frames.

### B. Tiled Full-Disk Inference

Full-disk scenes exceed GPU memory, so inference runs on 1024 × 1024 tiles that overlap by 128 pixels. Each tile's prediction is weighted by a window ωk that ramps linearly to its edges, and the overlapping predictions are averaged:

    Î(p) = Σk ωk(p) Îk(p) / Σk ωk(p)    (12)

On three Himawari-9 full-disk scans (5500 × 5500 pixels) this takes 31 s in total, with no visible tile seams. Outputs are written as NetCDF files containing BT in kelvin, the synthesized timestamps, the source metadata including the projection parameters, and a per-frame flag that marks every generated frame as interpolated, so that it cannot be mistaken for an observation. PNG frames and GIF/MP4 animations are also produced.

### C. Near-Real-Time Himawari-9 Service

A live service polls the Himawari-9 Target sector on AWS every 30 s. Each new scan is downloaded, calibrated, aligned to the fixed grid, and interpolated against the previous scan with k = 4, turning the 2.5-min cadence into 30-s frames. Processing takes about 1 s per scan on the GPU; the end-to-end latency is dominated by the 4–6 min delivery delay of the data. Gaps longer than 8 min (missed scans) are not interpolated across, and outputs are kept in a rolling 6-hour window.

### D. Toward INSAT-3DS

The motivating application is 30-minute INSAT-3DS imagery (30 → 15 → 7.5 min), obtained in HDF5 format from MOSDAC after registration. Because the model operates on normalized BT with fixed physical bounds, applying it to INSAT-3DS requires only a loader that converts the imager counts to BT with the product's calibration tables. This loader has not yet been implemented, and INSAT-3DS results are therefore not reported here. The Himawari-9 test in Section VI-C is the closest available proxy for cross-sensor transfer; the 4-km resolution and 30-min gaps of INSAT-3DS are both outside the training distribution.

### E. Interactive Dashboard

A Streamlit dashboard lets the user choose a checkpoint and an interpolation factor, load a processed sequence (with 1-minute truth) or upload GOES, Himawari, or NumPy files, play the original and interpolated animations side by side, inspect individual frames with a slider and difference maps, compare per-clip metrics for SatFIN, DIS, and linear blending, browse the test-set results and error maps, and watch the live Himawari-9 feed. Results can be downloaded as NetCDF, MP4, or GIF.

## VIII. Limitations and Future Work

Interpolated frames are estimates, not observations. The model can only reproduce evolution that is implicit in the two bounding frames: a storm that initiates and decays entirely within a gap cannot be recovered, and claims about earlier warning times must be verified with event-based studies before operational use. The training data covers one IR band and 9 hours over the southern U.S. Great Plains in June 2025; other seasons, regions, day–night mixes, and bands have not been tested. Large motions remain hard, as the Himawari-9 10-minute result shows. Invalid and off-disk pixels are filled with the frame mean before interpolation, so these areas are not physical in full-disk output, and the output NetCDF carries projection parameters but no per-pixel latitude/longitude or GeoTIFF export. The domain gap to 4-km, 30-minute INSAT-3DS data may reduce accuracy, so fine-tuning on Himawari data or on INSAT-3DS rapid-scan data, where available, is advisable. Future work includes an INSAT-3DS loader, multiband preprocessing, geolocated output, physics-informed constraints on the predicted motion, uncertainty estimates for each synthesized pixel, and verification of downstream products such as convective-initiation detection and nowcasts.

## IX. Conclusion

This paper presented SatFIN, a deep optical-flow frame-interpolation framework for increasing the temporal resolution of geostationary satellite imagery without additional orbital hardware. By training on real 1-minute GOES-19 mesoscale observations with simulated temporal sparsity, using a convection-focused loss, and evaluating with physically meaningful metrics against classical baselines, SatFIN provides a reproducible path from public data to denser image sequences. On a held-out GOES-19 day it outperforms linear blending and classical optical flow at every gap from 2 to 10 minutes, with its largest gains on cold cloud tops, and it runs fast enough to power a near-real-time 30-second Himawari-9 product. On an unseen satellite it remains best at 5-minute gaps, while classical flow is still slightly ahead at 10 minutes. More diverse training data with larger displacements, and an INSAT-3DS loader, are the next steps toward operational 30-minute imagery.

## Acknowledgment

The authors thank NOAA for open access to GOES-R and Himawari data through the NOAA Open Data Dissemination program, and ISRO/SAC for INSAT-3DS data provided through MOSDAC.

## References

[1] T. J. Schmit, P. Griffith, M. M. Gunshor, J. M. Daniels, S. J. Goodman, and W. J. Lebair, "A closer look at the ABI on the GOES-R series," Bull. Amer. Meteor. Soc., vol. 98, no. 4, pp. 681–698, 2017.
[2] K. Bessho et al., "An introduction to Himawari-8/9—Japan's new-generation geostationary meteorological satellites," J. Meteor. Soc. Japan, vol. 94, no. 2, pp. 151–183, 2016.
[3] Meteorological and Oceanographic Satellite Data Archival Centre (MOSDAC), Space Applications Centre, ISRO. [Online]. Available: https://www.mosdac.gov.in
[4] K. M. Roberts and S. A. Rutledge, "Nowcasting storm initiation and growth using GOES-8 and -9 data," Weather Forecast., vol. 18, no. 4, pp. 562–584, 2003.
[5] J. R. Mecikalski and K. M. Bedka, "Forecasting convective initiation by monitoring the evolution of moving cumulus in daytime GOES imagery," Mon. Weather Rev., vol. 134, no. 1, pp. 49–78, 2006.
[6] H. Jiang, D. Sun, V. Jampani, M.-H. Yang, E. Learned-Miller, and J. Kautz, "Super SloMo: High quality estimation of multiple intermediate frames for video interpolation," in Proc. IEEE/CVF CVPR, 2018, pp. 9000–9008.
[7] Z. Huang, T. Zhang, W. Heng, B. Shi, and S. Zhou, "Real-time intermediate flow estimation for video frame interpolation," in Proc. ECCV, 2022, pp. 624–642.
[8] G. Farnebäck, "Two-frame motion estimation based on polynomial expansion," in Proc. Scandinavian Conf. Image Anal. (SCIA), 2003, pp. 363–370.
[9] T. Kroeger, R. Timofte, D. Dai, and L. Van Gool, "Fast optical flow using dense inverse search," in Proc. ECCV, 2016, pp. 471–488.
[10] D. Sun, X. Yang, M.-Y. Liu, and J. Kautz, "PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume," in Proc. IEEE/CVF CVPR, 2018, pp. 8934–8943.
[11] Z. Teed and J. Deng, "RAFT: Recurrent all-pairs field transforms for optical flow," in Proc. ECCV, 2020, pp. 402–419.
[12] T. J. Vandal and R. R. Nemani, "Temporal interpolation of geostationary satellite imagery with optical flow," IEEE Trans. Neural Netw. Learn. Syst., 2021.
[13] NOAA/NASA, "GOES-R Series Product Definition and Users' Guide (PUG), Volume 3: Level 1b Products," GOES-R Program Office.
[14] M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu, "Spatial transformer networks," in Proc. NeurIPS, 2015, pp. 2017–2025.
[15] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Proc. MICCAI, 2015, pp. 234–241.
[16] P. Charbonnier, L. Blanc-Féraud, G. Aubert, and M. Barlaud, "Two deterministic half-quadratic regularization algorithms for computed imaging," in Proc. IEEE ICIP, 1994, pp. 168–172.
[17] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Proc. NeurIPS, 2019, pp. 8024–8035.
[18] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in Proc. ICLR, 2019.
[19] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.
