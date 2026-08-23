# Near-Field Deep Learning based Hybrid Beamforming — Stage 1 (Communication-Only)

This stage trains a neural network to replace the conventional methods used for analog (RF) precoder in a
hybrid beamforming system for a near-field, multi-user downlink. No sensing
target or ISAC objective is included yet — this is the communication-only
foundation the later ISAC stages build on.

## System Setup

- **M = 128** transmit antennas (uniform linear array), **N_RF = 4** RF chains
- **K = 4** single-antenna downlink users
- **100 GHz** carrier
- half-wavelength element spacing
- Near-field, LoS-dominant per-user channel model (each user has a single,
  fixed location $(\theta_k, r_k)$ relative to the array), users placed
  between 5 m and 80 m from the array , The range covers both near and far field region. With the near field boundary R=25m. so the spherical-wavefront channel model is
  used throughout rather than the far-field plane-wave approximation
- Hybrid transmit signal: 
  $$
\mathbf{x} = \mathbf{F}_{\mathrm{RF}}\mathbf{F}_{\mathrm{BB}}\mathbf{s}
$$
constrained to constant-modulus (phase-shifter)
entries and total transmit power constrained to $P_t$

## Neural Network Architecture

`MultiuserAnalogPrecoderNet` learns **only the analog precoder** $\mathbf{F}_{\rm RF}$.
It takes the near-field channel matrix $\mathbf{H}\in\mathbb{C}^{M\times K}$
(stacked user channels) as input and outputs an $M\times N_{\rm RF}$
constant-modulus matrix.

- Built from complex-valued layers throughout (`ComplexLinear`,
  `ComplexBatchNorm`, `ComplexTanh`) rather than splitting into real/imaginary
  channels, so the network operates natively on complex baseband quantities
- MLP backbone with hidden dimensions **[512, 256, 128]**
- A final **`CMNormalization`** layer projects the raw network output onto
  the constant-modulus feasible set, $\big|[\mathbf{F}_{\rm RF}]_{m,n}\big| = 1/\sqrt{M}$,
  so every output is hardware-feasible by construction — the network never
  needs to *learn* the modulus constraint, only the phases

## Loss Function: Concentrated MSE via KKT Elimination

The network is trained to minimize sum communication MSE, but it does not output the digital precoder $\mathbf{F}_{\mathrm{BB}}$ directly. Instead:

1. For a fixed $\mathbf{F}_{\mathrm{RF}}$, the per-user MSE with an MMSE receive combiner has a known closed form:

   $$
   \mathrm{MSE}_k =
   1 -
   \frac{
   \left|\mathbf{h}_k^{H}\mathbf{F}_{\mathrm{RF}}\mathbf{F}_{\mathrm{BB},k}\right|^2
   }{
   \sum_j
   \left|\mathbf{h}_k^{H}\mathbf{F}_{\mathrm{RF}}\mathbf{F}_{\mathrm{BB},j}\right|^2
   + \sigma_k^2
   }
   $$

2. Minimizing $\sum_k \mathrm{MSE}_k$ subject to the transmit power constraint over $\mathbf{F}_{\mathrm{BB}}$ (with $\mathbf{F}_{\mathrm{RF}}$ fixed) has a **closed-form KKT stationary point** — a regularized zero-forcing-type solution in the $N_{\mathrm{RF}}$-dimensional effective channel domain, computed via `torch.linalg.solve` so it remains differentiable.

3. This closed-form $\mathbf{F}_{\mathrm{BB}}(\mathbf{F}_{\mathrm{RF}})$ is substituted back into the MSE objective, eliminating $\mathbf{F}_{\mathrm{BB}}$ as a free variable — hence **concentrated MSE**. The network only ever has to learn $\mathbf{F}_{\mathrm{RF}}$; the digital stage is always the exact optimum for whatever $\mathbf{F}_{\mathrm{RF}}$ the network currently proposes.

This is a meaningful reduction in what the network has to learn. Instead of searching a much larger joint $(\mathbf{F}_{\mathrm{RF}}, \mathbf{F}_{\mathrm{BB}})$ space, gradient descent only has to solve the constant-modulus analog design problem, with the digital stage handled exactly by classical optimization at every training step and at inference time.

## Training Configuration

20,000 sampled channel realizations (70/15/15 train/val/test split), batch
size 1024, Adam-style optimizer at LR $7.5\times10^{-4}$ with a
plateau-triggered LR scheduler and early stopping (both patience 50 epochs),
trained for up to 500 epochs.

## Results

### Training Curves

![Training loss vs epoch](Outputs/loss_vs_epoch_L1.png)

Concentrated MSE loss falls from ~3.15 to ~0.80 over training, with smooth,
monotonic convergence (aside from a couple of small transient bumps around
epoch 230–250, consistent with a scheduler LR-reduction step) and a clear
plateau after ~400 epochs.

![Sum rate vs epoch](Outputs/sum_rate_vs_epoch_L1.png)

Training and validation sum rate rise together from ~2.3 bps/Hz to ~10
bps/Hz. Training stays slightly above validation for most of training — a
mild, expected generalization gap — and the two curves converge closely by
epoch ~450–500, indicating no severe overfitting.

### Baseline Comparison

Averaged over the test set:

| Configuration | Avg Sum Rate (bps/Hz) | Avg MSE |
|---|---:|---:|
| Random $\mathbf{F}_{\rm RF}$ + ZF $\mathbf{F}_{\rm BB}$ | 0.1466 | 3.9019 |
| Random $\mathbf{F}_{\rm RF}$ + Random $\mathbf{F}_{\rm BB}$ | 0.3304 | 3.9675 |
| Matched Filter $\mathbf{F}_{\rm RF}$ + ZF $\mathbf{F}_{\rm BB}$ | 9.0242 | 3.9497 |
| Matched Filter $\mathbf{F}_{\rm RF}$ + Random $\mathbf{F}_{\rm BB}$ | 1.4009 | 3.8805 |
| Matched Filter $\mathbf{F}_{\rm RF}$ + KKT $\mathbf{F}_{\rm BB}$ | 10.0851 | 0.8190 |
| **Neural Network $\mathbf{F}_{\rm RF}$ + KKT $\mathbf{F}_{\rm BB}$** | **9.9044** | **0.8469** |
| Full Digital $\mathbf{F}_{\rm RF}$ + ZF $\mathbf{F}_{\rm BB}$ | 10.0851 | 0.8190 |
| SOMP $\mathbf{F}_{\rm RF}$ + TH-HMP $\mathbf{F}_{\rm BB}$ | 8.2772 | 1.3294 |

The learned network reaches **9.9044 bps/Hz**, within ~1.8% of the best
achieved sum rate (10.0851 bps/Hz, shared by the matched-filter and fully
digital baselines) — despite replacing a per-channel matched-filter
computation with a single trained forward pass. It also clearly outperforms
the classical SOMP + TH-HMP iterative baseline. Note the Matched-Filter+KKT
and Full-Digital+ZF rows report identical figures to four decimal places;
worth double-checking whether that's a genuine near-field equivalence at
$N_{\rm RF}=K$ or a shared code path in the evaluation script before this
goes further into the write-up.

### Beam Pattern

![Example beam pattern](Outputs/beam_pattern_ex3.png)

Example test case — user angles $[29.59 79.12 49.43 79.43]$ at
ranges $[40.18, 50.51, 78.06, 38.54]$ m, sum rate 8.0726 bps/Hz, MSE
1.0259 (general and concentrated MSE agree exactly, confirming the KKT
closed form is being evaluated consistently). The far-field angular cut of
the transmit beampattern shows clear, well-aligned peaks at all four true
user angles (main-lobe peak at 50.72°, matching the nearest user to within
0.07°), with sidelobes generally 15–30 dB below the peaks.

## Inference Time: Neural Network vs. SOMP (Iterative Baseline)

The inference time averaged over 500 realization for Nueral Network is 10.5867 ms/sample, and using the conventional SOMP technique : 704.1061 ms/sample.
Here sample refers to one multi-user MIMO channel matrix H.
