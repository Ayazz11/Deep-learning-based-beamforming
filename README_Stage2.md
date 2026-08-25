# Near-Field Deep-Unfolded Hybrid Beamforming — Stage 2 (ISAC Extension)

Stage 2 extends the [Stage 1](README_Stage1.md) communication-only network to
joint communication **and sensing** (ISAC): the same analog precoder network
now also has to illuminate a set of near-field sensing targets, alongside
serving the $K=4$ downlink users. $M$, $N_{\rm RF}$, carrier frequency, and
the rest of the array geometry are unchanged from Stage 1.

## What's New vs. Stage 1

| | Stage 1 | Stage 2 |
|---|---|---|
| Input | $\mathbf{H}$ only ($M\times K$) | $\mathbf{H}_{\rm aug} = [\mathbf{H} \,\|\, \mathbf{B}]$ ($M\times(K{+}L)$) |
| Targets | none | $L=3$ near-field sensing targets |
| Network columns processed | $K=4$ | $K+L=7$ |
| Merge layer | ComplexLinear(512→512) | ComplexLinear(896→512) |
| Loss | $L_{\rm com}$ only | $\lambda L_{\rm com} + (1-\lambda)L_{\rm sen}$ |

## Targets and the Augmented Input

Three targets, fixed for every training sample (only $\mathbf{H}$ varies
batch to batch):

| Target | Angle | Range |
|---|---:|---:|
| 1 | 0° (broadside) | 15 m |
| 2 | 30° | 25 m |
| 3 | −20° | 40 m |

Each target's column $\mathbf{b}_\ell = \mathbf{a}_{\rm nf}(\theta_\ell, r_\ell)$
is the same near-field array-response function used for the channel model.
Target 2 sits just outside, and target 1 well inside, the array's Rayleigh
distance (24.2 m for this $M$, $d$, $\lambda$) — deliberately spanning both
sides of the near/far-field boundary.

$\mathbf{H}$ ($M\times K$) and $\mathbf{B}$ ($M\times L$) are concatenated
column-wise into $\mathbf{H}_{\rm aug}$ ($M\times7$) before the network sees
them.

## Architecture Changes

The per-column ComplexMLP backbone (128→512→256→128, ComplexLinear +
ComplexBatchNorm + ComplexTanh at each stage) is **unchanged and shared**
across all 7 columns — it processes user channels and target steering
vectors with identical weights, with no explicit "this is a user" /
"this is a target" flag. The 7 resulting 128-dim feature vectors are
concatenated into a 896-dim global vector, which only then passes through
the merge layer — the single architectural change from Stage 1
(ComplexLinear(512→512) → ComplexLinear(896→512)) — before reshaping and
constant-modulus normalization into $\mathbf{F}_{\rm RF}$, exactly as before.

## Loss Function

**Communication term** $L_{\rm com}$ is untouched from Stage 1 — the same
concentrated-MSE loss, with $\mathbf{F}_{\rm BB}$ still analytically
eliminated via the KKT closed form as a function of $\mathbf{F}_{\rm RF}$
and $\mathbf{H}$ alone.

**Sensing term** $L_{\rm sen}$ is new. With the effective transmit matrix
$\mathbf{TX} = \mathbf{F}_{\rm RF}\mathbf{F}_{\rm BB}$, the power delivered
to target $\ell$ (summed over all $K$ communication beams, since there is no
dedicated sensing-only stream) is $\|\mathbf{b}_\ell^H\mathbf{TX}\|^2$, and

$$L_{\rm sen} = -\frac{1}{L}\sum_{\ell=1}^{L}\big\|\mathbf{b}_\ell^H\mathbf{TX}\big\|^2$$

— negative, since minimizing $L_{\rm sen}$ maximizes total target power.

**A deliberate detail: $\mathbf{F}_{\rm BB}$ is detached before the sensing
loss.** $\mathbf{F}_{\rm BB}$ is still the KKT-optimal digital precoder for
whatever $\mathbf{F}_{\rm RF}$ the network currently proposes, but the KKT
solve is wrapped in `torch.no_grad()`. This means the sensing gradient flows
back to the network **only through $\mathbf{F}_{\rm RF}$**, not through
$\mathbf{F}_{\rm BB}$ — $\mathbf{F}_{\rm BB}$'s job stays exclusively
communication interference-suppression, and the network can't "cheat" by
using the sensing loss to distort the digital stage away from its
communication-optimal solution. All of the sensing/communication tradeoff is
resolved entirely within the $N_{\rm RF}=4$-dimensional analog precoder.

**Combining the two terms** requires normalization first — raw $L_{\rm com}$
and $L_{\rm sen}$ sit on very different numeric scales (roughly $[1,4]$ vs.
$\mathcal{O}(10^{-2})$), so a naive $\lambda=0.5$ mix would be dominated by
$L_{\rm com}$ by over an order of magnitude:

$$L_{\rm com}^{\rm norm} = \frac{L_{\rm com}}{K}, \qquad
L_{\rm sen}^{\rm norm} = \frac{L_{\rm sen}}{-P_t}, \qquad
\mathcal{L} = \lambda\, L_{\rm com}^{\rm norm} + (1-\lambda)\, L_{\rm sen}^{\rm norm}$$

$\lambda=1$ recovers Stage 1 exactly (pure communication); $\lambda=0$ is
pure sensing (targets illuminated, user rates unconstrained). The results
below use $\lambda=0.5$ — equal weight between the two objectives.

## Training

Same optimizer/scheduler setup as Stage 1 (Adam, plateau LR scheduling,
gradient clipping at norm 5.0), run for up to 1200 epochs — noticeably
longer than Stage 1's ~500, consistent with the harder joint optimization
surface.

### Training Curves

![Communication loss vs epoch](Figure_4.png)

$L_{\rm com}^{\rm norm}$ falls from ~1.17 to a floor around **0.19–0.20**,
with a transient bump around epoch 200–250 (likely the sensing gradient
pulling against communication early in training, before the two objectives
settle). Multiplying back by $K=4$ puts the underlying communication MSE at
roughly the same ~0.8 floor Stage 1 reached — i.e. **the network holds
communication performance close to its Stage 1 ceiling even while jointly
optimizing for sensing**, which is the result you'd want to see at
$\lambda=0.5$.

![Sensing loss vs epoch](Figure_3.png)

$L_{\rm sen}$ decreases steadily and smoothly from near 0 to about $-0.26$,
converging by roughly epoch 800–1000 — target illumination power is
climbing throughout training with no instability.

![Combined training loss vs epoch](Figure_1.png)

The combined loss $\mathcal{L}$ goes negative as it converges (~$-0.105$) —
expected, since $L_{\rm sen}^{\rm norm}$ is negative by construction and
comes to dominate the sum as sensing improves. Note the y-axis label
"Concentrated MSE Loss" looks like it's carried over from the Stage 1
plotting code and no longer describes what's actually plotted here (the
combined ISAC loss) — worth relabeling before this goes in the paper.

![Sum rate vs epoch](Figure_2.png)

Training sum rate rises quickly to ~18.8 bps/Hz before settling to ~17.7–17.8;
validation climbs much more slowly, only catching up around epoch 600–700,
plateauing near **15.2–15.3 bps/Hz**. Unlike Stage 1 — where training and
validation converged closely together — there's a **persistent ~2.5 bps/Hz
gap** here that doesn't close. Since the targets are fixed (identical for
every sample) and only $\mathbf{H}$ varies, this gap reflects the network's
communication-side response to varying channels, not target memorization;
worth investigating whether it's standard overfitting from the added
capacity (7-column processing, larger merge layer) or an interaction with
the sensing objective. The absolute sum-rate scale here (~15–18 bps/Hz) is
also well above Stage 1's ~10 bps/Hz for what's described as the same core
config — if $r_{\min}$/$r_{\max}$ for the users differ between the two runs
(e.g. a closer/near-field-only user population for this stage), that alone
would explain the gap and is worth stating explicitly in the write-up so
the two stages' sum-rate numbers aren't read as directly comparable.

### Beam Patterns

![1-D far-field beam pattern cut](Figure_5.png)

This is the same 1-D angular-cut plot (and the same underlying jagged
sidelobe floor) we diagnosed earlier: probing a near-field-focused precoder
with a far-field steering vector is the wrong tool once targets/users sit
inside the Rayleigh distance, which several of yours do here. The four
sharp peaks are real and roughly angle-aligned, but the noisy floor between
them isn't a meaningful sidelobe structure — treat this figure as a rough
sanity check only, not the primary sensing diagnostic.

![2-D near-field beampattern](Figure_6.png)

This is the right visualization for a near-field system, and a good
addition — power plotted jointly over angle **and** range, with the
Rayleigh distance and peak marked. One thing worth a closer look: the
pattern reads as near-vertical stripes, i.e. power at a given angle looks
almost constant across the whole 2.5–24 m range axis rather than
concentrating at the three targets' specific ranges (15, 25, 40 m). That's
actually consistent with something we worked out earlier in this project:
$L_{\rm sen}$ only rewards power *at* each target's exact $(\theta_\ell,
r_\ell)$ — it has no term penalizing power delivered to *other* ranges at
that same angle. Combined with how slowly near-field steering vectors
decorrelate across range at this array size (we measured
$\rho(\mathbf{a}(\theta,5\text{m}),\mathbf{a}(\theta,24\text{m}))\approx0.94$
for $M=128$ earlier), there's essentially no gradient signal pushing the
network toward range-selective focusing — angle-only beams already collect
most of the achievable target power "for free." If genuine range
discrimination matters for the sensing objective, $L_{\rm sen}$ likely needs
an explicit range-selectivity term (e.g. penalizing power at nearby-but-wrong
ranges) rather than reward-at-the-target-point alone.

## Summary

| | Stage 1 | Stage 2 ($\lambda=0.5$) |
|---|---|---|
| Objective | Communication only | Communication + sensing (3 targets) |
| Val. sum rate | ~9.9 bps/Hz | ~15.2–15.3 bps/Hz* |
| Comm. MSE floor | ~0.85 | ~0.8 (via $L_{\rm com}^{\rm norm}\times K$) |
| Train/val gap | closes by convergence | persistent ~2.5 bps/Hz |

\*not directly comparable to Stage 1 if the near-field user range differs
between runs — confirm $r_{\min}$/$r_{\max}$ before quoting both numbers
together in the paper.

## Open Items for Next Pass

- Confirm whether Stage 1 and Stage 2 use the same user range $[r_{\min},
  r_{\max}]$; state explicitly either way before comparing sum rates
- Investigate the persistent train/val sum-rate gap
- Consider a range-selectivity term in $L_{\rm sen}$ if the 2-D beampattern's
  angle-only structure isn't the intended sensing behavior
- Relabel the combined-loss training curve's axis
- Regenerate the 1-D beam pattern with a near-field-matched probe (or drop
  it in favor of the 2-D map) for the final write-up

## Inference Time: Neural Network vs. SOMP (Iterative Baseline)

*Carried over from Stage 1 — still to be completed.*
