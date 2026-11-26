## ISAC Extension: Joint Communication and Target Illumination

The analog-precoder network is extended so that the same constant-modulus precoder serves K downlink users and illuminates T targets. No dedicated sensing beams or extra RF chains are used. Only the communication symbols are transmitted, and the analog precoder shapes where that energy goes.

### Method

**Sensing model.** Targets are one-way receivers with unit-norm near-field steering vectors `B = [b(θ_t, r_t)] ∈ C^{M×T}` (`target_model.py`). The power at target *t* is `P_t = ||b_tᴴ F_RF F_BB||²`, which is a transmit-side quantity and does not include array gain, round-trip loss or RCS. `P_t` is expressed as a fraction of the transmit power (P_tx = 1 W).

**Digital precoder.** `F_BB` stays the communication-optimal KKT closed form (`kkt_precoder.py`), so the digital stage needs no extra training.

**Loss** (`loss.py::isac_loss`):

```
L = λ · E_c(F_RF) / K  +  (1 − λ) · L_sen / P_tx
E_c(F_RF) = tr[(I + P_tx/(Kσ²) · Hᴴ F_RF (F_RFᴴ F_RF)⁻¹ F_RFᴴ H)⁻¹]      (concentrated MSE)
L_sen     = −(1/T) · Σ_t ||b_tᴴ F_RF F_BB||²                            (F_BB detached)
```

The run reported below uses **λ = 0.5**. This follows from the logged values: 0.5·(0.1915/4) + 0.5·(−0.2567) = −0.1044.

**Network change.** The target steering vectors are appended to the per-user inputs, so the shared complex MLP sees K+T = 7 columns. The merge layer input grows from 4·128 to 7·128. By my count this gives about 1.38 M real parameters (about 0.99 M for the communication-only model). The model constructor prints the exact figure.

### Experimental setup

| Item | Value |
|---|---|
| Array / users / targets | M = 128 ULA, K = 4, T = 3, N_RF = 4 |
| Carrier / Rayleigh distance | 100 GHz / 24.2 m |
| Power / noise | P_tx = 1 W, σ² = 0.1 (nominal 10 dB SNR) |
| Optimizer | AdamW, initial LR 7.5e-4, ReduceLROnPlateau (×0.5), grad-clip 5 |
| Epochs run | 1200 (best validation epoch: 648) |
| LR at end of run | 2.34e-5 (five halvings from 7.5e-4) |
| Checkpoint | `models/Best_model_ISAC_128_L3.pt` |
| Training script | `train_128_isac.py` |

### Headline results (held-out test set, best checkpoint)

| Metric | Value |
|---|---|
| Sum rate | **15.507 bps/Hz** |
| Average sensing power | **0.246961** (per target, fraction of P_tx) |
| Target 1 / 2 / 3 | 0.238539 / 0.256215 / 0.246129 |
| Total power on the 3 targets | 0.7409 of P_tx |
| Reference sum rate (logged as "MF baseline") | 30.343 bps/Hz |
| Sum rate relative to reference | 51.1% (loss of 14.84 bps/Hz) |

### Validation and training (final epoch 1200)

| Quantity | Value |
|---|---|
| Total training loss | −0.1045 |
| Communication loss E_c (training) | 0.1914 |
| Sensing loss L_sen (training) | −0.2568 |
| Training sum rate | 17.769 bps/Hz |
| Validation sum rate | 15.310 bps/Hz (best 15.481 at epoch 648) |
| Validation sensing power (avg) | 0.2530 (targets: 0.2514 / 0.2659 / 0.2417) |
| Validation reference rate / percent | 30.318 bps/Hz / 50.5% |
| Early-stopping counter | 552 (no improvement since epoch 648) |

The validation and test numbers differ by about 1 to 2% (15.31 vs 15.51 bps/Hz, 0.253 vs 0.247 W), which is ordinary sampling variation across the two 3000-sample splits.

### Training dynamics

![Total loss](Outputs/ISAC_Outputs_with_NLoS_L3/Figure_1.png)
![Sum rate](Outputs/ISAC_Outputs_with_NLoS_L3/Figure_2.png)
![Sensing loss](Outputs/ISAC_Outputs_with_NLoS_L3/Figure_3.png)
![Communication loss](Outputs/ISAC_Outputs_with_NLoS_L3/Figure_4.png)

Values below are read from the plots, so they are approximate.

1. **Epochs 0 to about 120, communication first.** E_c falls from about 1.17 to about 0.16 and the training sum rate climbs to about 18.8 bps/Hz. The sensing loss barely moves (about −0.008, which is the P_tx/M ≈ 0.0078 level expected for a random constant-modulus beam).
2. **Epochs about 130 to 300, the trade-off appears.** L_sen drops steeply from about −0.02 to about −0.21. E_c rises to about 0.23 and the training sum rate falls to about 16.5 to 17 bps/Hz.
3. **Epochs 300 to 1200, slow co-convergence.** L_sen settles at −0.257 and E_c at 0.191. The training sum rate plateaus at about 17.77 bps/Hz. The total loss goes from about 0.142 to −0.1045.

### Beam patterns (one test sample)

![Transmit beam pattern](Outputs/ISAC_Outputs_with_NLoS_L3/Figure_5.png)
![Near-field 2-D beampattern](Outputs/ISAC_Outputs_with_NLoS_L3/Figure_6.png)

- Three dominant lobes appear at about −56°, +35° and +49° (peaks at about −2.5, −1 and 0 dB). Most sidelobes sit between −20 and −30 dB.
- The 2-D pattern over 2 to 24 m (inside the Rayleigh distance) shows **vertical stripes**. The learned beam is concentrated in angle and shows no visible range focusing in this window.

### Analysis

**Sensing gain.** A random constant-modulus beam delivers P_tx/M = 1/128 ≈ 0.0078 per target. The trained network reaches 0.247, which is **31.6× (about +15 dB)**.

**Upper bound.** For near-orthogonal targets, Σ_t ||b_tᴴ x||² ≤ λ_max(BBᴴ)·P_tx ≈ P_tx, so the per-target ceiling is about 1/T = 0.333. The network reaches **about 74%** of it. Per-target balance is good: max/min = 1.074 (0.3 dB).

**Why the sum rate halves.** F_RF F_BB lies in an N_RF = 4 dimensional subspace that communication already needs in order to serve 4 users, while 3 targets compete for the same subspace. This degrees-of-freedom limit explains the drop, and it suggests N_RF > K should recover rate at the same sensing level.

**Generalization gap.** The training sum rate (17.77) exceeds the validation sum rate (15.31) by 2.46 bps/Hz (13.8% of training). The sensing gap is only 1.5% (0.2568 vs 0.2530). Overfitting is therefore in the communication part. In the communication-only run (λ = 1) the training and validation curves coincide at about 30.3 bps/Hz.

**Loss-to-rate sanity check.** Assuming equal per-user SINR with MSE = 1/(1+SINR): E_c = 0.1914 corresponds to about 17.5 bps/Hz (measured training rate 17.77), and the communication-only E_c ≈ 0.0227 corresponds to about 29.8 bps/Hz (measured about 30.3). The loss and the rate are consistent.

### Assumptions and limitations

- Sensing is measured as one-way beampattern gain at known target positions. It is not detection probability, localization error or the Cramér-Rao bound.
- F_BB is the communication-optimal precoder and is detached in the sensing term, so the gradient with respect to F_RF is approximate. The closed-form sensing term avoids this.
- Normalization: L_sen/P_tx effectively spans [−1/T, 0], not [−1, 0], and E_c/K is small at high SINR, so the nominal λ = 0.5 is not an equal split in practice.
- Only one λ was trained. The communication-sensing Pareto front is not yet measured.
- Range focusing is not visible in the pattern above. In this data most users and targets lie beyond the Rayleigh distance, so the experiment weakly tests the near-field benefit.

### Next steps

- Sweep λ ∈ {0, 0.1, …, 1} and plot sum rate vs sensing power (the Pareto front), including a fully-digital ISAC bound.
- Re-run with N_RF = K + 2 to test the degrees-of-freedom explanation.
- Evaluate with r_max reduced (or M = 256) to expose range focusing, and add a CRB-based sensing metric.
