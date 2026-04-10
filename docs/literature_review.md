# Literature Review

The review is not a formal systematic survey. It is a focused scan of the lines of work most likely to overlap with this repo's actual scope:

- sim-to-real and dynamics adaptation in RL
- off-dynamics RL
- off-policy instability and value-error literature
- TD-error-based diagnostics that are close to our warning-score idea

Searches were grounded in primary paper sources available through arXiv, OpenReview, PMLR, RSS, IJCAI, and publisher/author-hosted publication pages on April 9, 2026.

## Executive View

- The transfer/adaptation literature is already large. A paper framed as "we adapt robotic RL policies under dynamics shift" would not be novel.
- The instability literature is also mature. A paper framed as "off-policy RL can become unstable because of value error / overfitting / divergence" would also not be novel.
- The narrower combination still appears less crowded: a controlled empirical study of collapse during robotic fine-tuning under domain shift, plus an online recent-buffer TD-based warning signal evaluated by lead time and ROC-style metrics.
- The closest novelty risk is [Efficient Deep Reinforcement Learning Requires Regulating Overfitting](https://openreview.net/forum?id=14-kr46GvP-), because it already treats TD error as a useful training-time signal.
- The safest paper claim is therefore a measurement-and-diagnostics paper, not a new transfer algorithm paper.

## 1. Transfer and Adaptation Under Shift

### Sim-to-real robustness by randomization

- [Tobin et al., 2017, Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)
  This paper established the now-standard idea that aggressive simulator randomization can make the real world look like another training variation. It is important background, but it is mostly about transfer robustness during pretraining, not collapse during later fine-tuning.
- [Peng et al., 2018, Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://doi.org/10.1109/ICRA.2018.8460528)
  This is the more direct dynamics-side precursor. The key idea is to randomize simulator dynamics so a policy transfers without real-world training. Again, the objective is zero-shot robustness, not online collapse detection during adaptation.

### Online adaptation after deployment

- [Nagabandi et al., 2018, Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347)
  This paper learns a dynamics prior that can adapt quickly online using recent data. It is close in spirit because it explicitly targets perturbations, payloads, and terrain changes. The difference is that it proposes a fast adaptation method; our repo is currently set up to characterize when ordinary fine-tuning breaks and whether diagnostics can warn early.
- [Hansen et al., 2021, Self-Supervised Policy Adaptation during Deployment](https://openreview.net/forum?id=o_V-MjyyGV_)
  PAD studies reward-free adaptation after deployment using self-supervision. The deployment-under-shift setting is highly relevant, but the mechanism and claim are different: it is about recovering performance without reward, not about analyzing collapse thresholds or warning signals.
- [Kumar et al., 2021, RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034)
  RMA is a major legged-robot adaptation reference. It trains an adaptation module so a quadruped can handle new terrains, payloads, and wear in fractions of a second. This is important because it shows the robotics community already values fast adaptation, but it does not answer our narrower question about sample-efficient fine-tuning collapse and its detectability.

## 2. Off-Dynamics RL

This is the nearest algorithmic neighborhood to the planned paper.

- [Eysenbach et al., 2021, Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers](https://openreview.net/forum?id=eqBwg3AcIAK)
  DARC modifies rewards using classifiers that distinguish source-domain from target-domain transitions. This is a direct attempt to solve source-to-target dynamics mismatch.
- [Lyu et al., 2024, Cross-Domain Policy Adaptation by Capturing Representation Mismatch](https://arxiv.org/abs/2405.15369)
  PAR penalizes source-domain data using representation mismatch and explicitly studies kinematic and morphology shifts. This is one of the strongest recent overlaps in terms of problem statement.
- [Guo et al., 2024, Off-Dynamics Reinforcement Learning via Domain Adaptation and Reward Augmented Imitation](https://openreview.net/forum?id=k2hS5Rt1N0)
  DARAIL argues that reward modification alone is insufficient and adds imitation to improve target-domain behavior.
- [Daoudi et al., 2024, A Conservative Approach for Few-Shot Transfer in Off-Dynamics Reinforcement Learning](https://www.ijcai.org/proceedings/2024/430)
  This paper focuses on few-shot transfer and penalized conservative behavior when target data are scarce.
- [Lyu et al., 2024, ODRL: A Benchmark for Off-Dynamics Reinforcement Learning](https://openreview.net/forum?id=ap4x1kArGy)
  This benchmark matters for positioning. It argues the field lacked a standard benchmark and reports that no method has universal advantages across shifts. That makes a careful measurement paper more credible than a broad "we solved transfer" claim.

### Implication for us

- If we present the paper as another algorithm for source-to-target adaptation, we will be entering a crowded lane.
- If we present it as a controlled study of collapse behavior and early warning during fine-tuning under shift, the overlap drops substantially.

## 3. Off-Policy Instability and Value Error

### Algorithmic foundations

- [Haarnoja et al., 2018, Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
  SAC is the base learner in this repo. Its relevance is mostly methodological: readers will interpret any failure patterns through the behavior of SAC under replay, entropy regularization, and twin-critic updates.
- [Fujimoto et al., 2018, Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html)
  TD3 is the canonical reference for value overestimation and actor-critic instability from function-approximation error.
- [Fujimoto et al., 2019, Where Off-Policy Deep Reinforcement Learning Fails](https://openreview.net/forum?id=S1zlmnA5K7)
  This paper shows how extrapolation error breaks standard off-policy learning under mismatched data distributions. That is highly relevant to any fine-tuning-under-shift discussion.

### n-step and replay

- [Barth-Maron et al., 2018, Distributed Distributional Deterministic Policy Gradients](https://openreview.net/forum?id=SyZipzbCb)
  D4PG helped normalize `N`-step returns as a practical ingredient in continuous-control actor-critic methods.
- [Fedus et al., 2020, Revisiting Fundamentals of Experience Replay](https://research.google/pubs/revisiting-fundamentals-of-experience-replay/)
  This paper is especially important for the current experiment design because the key result is specifically about uncorrected `n`-step returns in replay-based RL. That is a direct match to this repo's implementation choice: `atlas.nstep.NStepTransitionAggregator` performs naive off-policy aggregation rather than an importance-corrected variant. That makes our `n in {1, 3, 10}` factor scientifically motivated rather than cosmetic.

### Instability, overfitting, and divergence

- [Nikishin et al., 2022, The Primacy Bias in Deep Reinforcement Learning](https://proceedings.mlr.press/v162/nikishin22a.html)
  This paper argues that deep RL can overfit early interactions and discount later evidence.
- [Lyle et al., 2022, Understanding and Preventing Capacity Loss in Reinforcement Learning](https://openreview.net/forum?id=ZkC8wKoLbQ7)
  This work argues that sequentially changing targets can erode the network's ability to fit new functions over time.
- [Cetin et al., 2022, Stabilizing Off-Policy Deep Reinforcement Learning from Pixels](https://proceedings.mlr.press/v162/cetin22a.html)
  This paper introduces the notion of catastrophic self-overfitting in pixel-based off-policy RL.
- [Li et al., 2023, Efficient Deep Reinforcement Learning Requires Regulating Overfitting](https://openreview.net/forum?id=14-kr46GvP-)
  This is the closest direct overlap with our diagnostic story. The paper argues that high validation TD error is the main culprit behind poor performance in sample-efficient deep RL and uses it for online model selection.
- [Hussing et al., 2024, Dissecting Deep RL with High Update Ratios: Combatting Value Divergence](https://openreview.net/forum?id=ofwv9VYp3h)
  This paper revisits primacy-bias-style failures and argues that value divergence is a more central mechanism than simple early-data overfitting in high-UTD settings.

### Implication for us

- We should not claim to have discovered that TD-error behavior is relevant to instability in RL. That is already in the literature.
- We can still contribute if we narrow the claim to a specific fine-tuning-under-shift regime and evaluate a practical online warning score rather than a model-selection heuristic.

## 4. What Appears Open

The following appears underexplored in the sources reviewed above:

- collapse during sample-efficient fine-tuning of a pretrained robotic locomotion policy under a fixed, controlled dynamics shift
- a recent-buffer-only diagnostic that ignores full replay and eval rollouts
- per-eval warning scores assessed by warning lead time and ROC-style prediction of threshold-defined collapse
- a small factorial analysis of backup horizon and critic size as factors affecting both collapse and warning quality

This is an inference from the reviewed literature, not a proof of absence. The reviewed papers strongly suggest adjacency, but I did not find a paper with exactly this combination of setting, metric design, and evaluation target.

## 5. Novelty Risk Assessment

### High-risk claims

- "We propose a new method for transfer under dynamics mismatch."
- "We show robotic RL policies can adapt under dynamics shift."
- "We are the first to use TD error to understand RL instability."

All of those claims are already crowded by prior work.

### Lower-risk claims

- "We provide a controlled empirical study of collapse during fine-tuning under dynamics shift in a legged-locomotion vertical slice."
- "We compare backup horizon and critic width as factors affecting both collapse incidence and warning predictability."
- "We evaluate whether recent-buffer TD diagnostics provide actionable early warning before threshold-defined collapse."

Those claims are much better aligned with the actual gap.

## 6. Recommended Paper Framing

The safest framing is:

- a measurement paper about fine-tuning stability under controlled shift
- with a secondary diagnostics contribution around early warning
- in a deliberately narrow robotic RL vertical slice

The framing to avoid is:

- a universal sim-to-real or off-dynamics adaptation algorithm
- a broad theory paper about all RL collapse
- a claim of first-ever TD-based instability detection

## 7. The Closest Prior Work To Watch

If we want one paper to keep in mind while designing claims, it is:

- [Li et al., 2023, Efficient Deep Reinforcement Learning Requires Regulating Overfitting](https://openreview.net/forum?id=14-kr46GvP-)

Why it matters:

- it already centers TD error as a meaningful training-time signal
- it links that signal to poor performance
- it uses the signal for online model selection

Why our current plan can still be distinct:

- we use fine-tuning under explicit source-to-target shift rather than generic online training
- we restrict the diagnostic to recent fine-tune transitions
- we evaluate warning lead time before threshold-defined collapse, not just model selection utility
- we focus on a robotic locomotion case study and a fixed factorial design

## 8. Design Changes That Would Strengthen Novelty

If we want the paper to feel more publication-ready, the highest-value upgrades are:

1. Add a small sensitivity analysis over pretrain seeds so the results are not conditional on one shared pretrain checkpoint.
2. Lock a primary warning metric suite before the sweep starts: ROC-AUC, lead time, false-positive rate, and usable-warning coverage.
3. Keep one explicit comparison between the recent-buffer warning score and a simpler baseline. This is now the recommended primary comparison, and the cleanest baseline is raw return drop on the same evaluation checkpoints because it is cheap, already supported by the artifact format, and does not cut against the repo's recent-buffer diagnostic design.
4. Add at least a small shift-severity sensitivity story, ideally one milder and one more extreme shift on representative cells, so the results do not hinge entirely on a single point in shift space.
5. Treat the paper as a scoped case study and say so early.

## 9. Bottom Line

The current project does not look like a duplicate of the standard sim-to-real or off-dynamics adaptation papers, provided we keep the paper narrowly framed. The part that appears most defendably new is not "how to adapt," but "how collapse behaves during fine-tuning under shift, and whether a simple recent-buffer TD signal can warn before collapse."

With the current one-environment, one-primary-shift design, the plan looks more naturally matched to a workshop paper or a tightly scoped empirical submission than to a venue that will demand broad generalization claims. Extra task coverage or at least a stronger shift-sensitivity story would materially improve that position.
