# Paper Outline

This document is a manuscript scaffold for the current `fine-tune-stability` vertical slice. It is intentionally grounded in the experiment logic already encoded in the repository so we do not promise claims the current design cannot support.

## Working Title

Early Warning Signals for Collapse During Sample-Efficient Fine-Tuning of Robotic Reinforcement Learning Policies

Alternative title:

Stability Warnings for Off-Policy Fine-Tuning Under Controlled Domain Shift in Quadruped Locomotion

## What This Paper Can Defend

The current codebase is set up to support a focused empirical paper, not a broad framework paper. The defensible scope today is:

- one task family: Go1 flat-terrain joystick locomotion
- one training stack: Brax SAC with MuJoCo Playground integration
- one controlled domain shift family: nominal train-domain randomization followed by a harder fixed fine-tune domain
- one early-warning family: recent-buffer TD-error statistics
- one small factorial sweep over backup horizon and critic capacity

The paper framing is locked: this is a warning-signal measurement paper, with stability experiments as supporting evidence inside a controlled case study of fine-tuning under shift. It should not be framed as a universal claim about all RL fine-tuning regimes or as a new transfer algorithm.

Venue-fit realism:

- with one environment and one primary shift family, the current design is better matched to a workshop paper or a narrowly scoped empirical submission than to a top-tier generalization-heavy paper
- stronger venue ambitions likely require at least one additional task or a broader shift-sensitivity story

## Abstract

[Placeholder]

This paper is a warning-signal measurement study of sample-efficient fine-tuning under controlled domain shift. We pretrain a locomotion policy in a nominal domain, fine-tune it in a shifted domain, and evaluate whether recent-buffer TD-error diagnostics provide actionable warning before threshold-defined collapse. The stability experiments over backup horizon and critic size serve to stress the warning problem under multiple regimes rather than to introduce a new adaptation method. We find [RESULT], with warning signals providing [RESULT] evaluation steps of lead time before collapse. These results suggest [RESULT / IMPLICATION], while also showing that [LIMITATION].

## 1. Introduction

### 1.1 Motivation

- Fine-tuning is attractive because it reuses expensive pretrained policies instead of training from scratch.
- In robotic RL, a policy that transfers well on average can still fail abruptly after a domain shift.
- Practitioners need warning signals that appear before visible collapse, not only post hoc failure detection.

### 1.2 Problem Statement

We ask whether fine-tuning instability under controlled domain shift can be measured systematically enough to support an operational early-warning signal based on recent TD-error behavior.

### 1.3 Why This Matters

- Sample-efficient fine-tuning is often the only practical adaptation path once pretraining has already consumed the larger budget.
- If collapse is predictable before the return metric crosses a failure threshold, operators can stop runs, intervene, or adapt schedules.
- The strongest paper contribution here is not a new transfer algorithm, but a useful measurement of warning quality alongside stability outcomes.

### 1.4 Claimed Contributions

The paper's contribution framing is locked around measurement and warning quality:

1. A controlled measurement protocol for studying fine-tuning collapse under domain shift in a robotic RL vertical slice.
2. A recent-buffer TD-error warning score with an explicit warmup procedure, hold-based trigger, and evaluation against threshold-defined collapse and a simple raw return-drop warning baseline.
3. An empirical comparison of backup horizon and critic width as stressors that change both collapse behavior and warning predictability.
4. A reproducible artifact set with per-eval logs, threshold metadata, and sweep manifests suitable for downstream warning analysis.

## 2. Background and Related Work

A fuller annotated review of the nearest prior work and the novelty risks for this paper lives in [literature_review.md](./literature_review.md).

### 2.1 Fine-Tuning and Transfer in RL

Prior work on transfer in RL mostly tries to make deployment under shift succeed, rather than to measure when adaptation fails. [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) and [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://doi.org/10.1109/ICRA.2018.8460528) train source policies to be robust enough for zero-shot transfer. [Learning to Adapt in Dynamic, Real-World Environments Through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347), [Self-Supervised Policy Adaptation during Deployment](https://openreview.net/forum?id=o_V-MjyyGV_), and [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034) study fast online adaptation after deployment.

### 2.2 Collapse and Instability in Off-Policy RL

This paper sits on top of the off-policy-instability literature. [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) provides the algorithmic base. [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html) and [Where Off-Policy Deep Reinforcement Learning Fails](https://openreview.net/forum?id=S1zlmnA5K7) show why value overestimation and extrapolation error matter. More recent work on [The Primacy Bias in Deep Reinforcement Learning](https://proceedings.mlr.press/v162/nikishin22a.html), [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://openreview.net/forum?id=ZkC8wKoLbQ7), [Stabilizing Off-Policy Deep Reinforcement Learning from Pixels](https://proceedings.mlr.press/v162/cetin22a.html), [Efficient Deep Reinforcement Learning Requires Regulating Overfitting](https://openreview.net/forum?id=14-kr46GvP-), and [Dissecting Deep RL with High Update Ratios: Combatting Value Divergence](https://openreview.net/forum?id=ofwv9VYp3h) argues that overfitting, representation degradation, and Q-value divergence can all destabilize learning.

### 2.3 n-step Returns and Stability Tradeoffs

[Distributed Distributional Deterministic Policy Gradients](https://openreview.net/forum?id=SyZipzbCb) treats `N`-step returns as one of the useful ingredients for strong continuous-control performance. [Revisiting Fundamentals of Experience Replay](https://research.google/pubs/revisiting-fundamentals-of-experience-replay/) is especially relevant here because its non-trivial result is not merely that `n`-step returns can help, but that uncorrected `n`-step returns can be uniquely beneficial in replay-based RL. That matters directly for this repo: `atlas.nstep.NStepTransitionAggregator` performs naive off-policy aggregation rather than an importance-corrected variant. In other words, the `n in {1, 3, 10}` factor is probing a design choice the runtime actually makes, not an abstract hyperparameter axis.

### 2.4 Warning Signals and Predictive Diagnostics

The closest adjacent diagnostic paper we found is [Efficient Deep Reinforcement Learning Requires Regulating Overfitting](https://openreview.net/forum?id=14-kr46GvP-), which uses validation TD error as a model-selection signal for sample-efficient RL. By contrast, the current repo is positioned to study recent-buffer TD diagnostics as an online warning signal during fine-tuning under explicit domain shift, with lead-time and ROC-style evaluation against threshold-defined collapse. That appears materially different from the adaptation papers above, but this is an inference from the reviewed literature rather than a proof that no closer prior paper exists.

### 2.5 Off-Dynamics RL Positioning

The nearest algorithmic literature to the proposed study is off-dynamics RL: [Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers](https://openreview.net/forum?id=eqBwg3AcIAK), [Cross-Domain Policy Adaptation by Capturing Representation Mismatch](https://arxiv.org/abs/2405.15369), [Off-Dynamics Reinforcement Learning via Domain Adaptation and Reward Augmented Imitation](https://openreview.net/forum?id=k2hS5Rt1N0), and [A Conservative Approach for Few-Shot Transfer in Off-Dynamics Reinforcement Learning](https://www.ijcai.org/proceedings/2024/430). [ODRL: A Benchmark for Off-Dynamics Reinforcement Learning](https://openreview.net/forum?id=ap4x1kArGy) is especially relevant because it reports that no method has universal advantages across varied dynamics shifts. That strengthens the case for a careful measurement paper that characterizes failure modes and warning quality instead of claiming a universally better transfer algorithm.

## 3. Research Questions and Hypotheses

### 3.1 Primary Research Questions

RQ1. How does backup horizon (`n in {1, 3, 10}`) affect collapse incidence and time-to-collapse during fine-tuning under shift?

RQ2. How does critic width (`256` vs `1024`, depth fixed at `3`) affect fine-tuning stability?

RQ3. Does the recent-buffer TD diagnostic score provide useful warning lead time before collapse?

### 3.2 Exploratory Question

RQ4. Are warning quality and collapse behavior aligned, or do some settings appear to improve return while degrading predictability?

This question is exploratory rather than primary. The current six-cell factorial is well suited to estimating stability and warning behavior by condition, but it is not designed specifically to guarantee matched-return counterexamples where warning reliability diverges cleanly from performance.

### 3.3 Hypotheses

These should be finalized before running the full sweep. They apply to the primary research questions only.

- H1. Backup horizon changes collapse behavior under shift rather than only changing sample efficiency.
- H2. Critic width changes stability and/or warning behavior even with actor architecture fixed.
- H3. The recent-buffer diagnostic score rises before threshold-defined collapse often enough to provide non-trivial lead time.

## 4. Experimental Design

### 4.1 Environment and Task

- Environment: `Go1JoystickFlatTerrain`
- Training stack: Brax SAC with MuJoCo Playground runtime
- Discount: `gamma = 0.99`
- Soft target update rate: `tau = 0.005`
- Optimizer learning rate for actor, critic, and entropy coefficient: `3e-4`
- Batch size: `256`
- Gradient updates per environment step: `1`
- Reward scaling: `1.0`
- Entropy coefficient: learned online via SAC's standard temperature update, with target entropy set from action dimensionality
- Actor architecture: width `256`, depth `3`
- Critic depth: `3`

These parameters matter for stability and should be reported in the final methods table even when they are not sweep factors.

### 4.2 Domain Shift

Nominal pretraining distribution:

- train friction range: `(0.8, 1.2)`
- train payload range: `(0.8, 1.2)`

Shifted fine-tune domain:

- fine-tune friction: `0.3`
- fine-tune payload: `1.5`

This is the primary shift severity for the main factorial, not the full shift story the paper should aspire to defend. To strengthen the paper without exploding the factorial, add a small shift-severity sensitivity check with at least one milder shift and one more extreme shift on `1-2` representative cells. Exact severity values should be locked before the sweep.

Paper phrasing:

[Placeholder]

We study a controlled dynamics shift that simultaneously lowers ground friction and increases payload relative to the nominal training distribution.

### 4.3 Training and Evaluation Protocol

Planned full-sweep defaults from the repo:

- fine-tune steps per run: `2,000,000`
- eval interval: `100,000` environment steps
- eval episodes per checkpoint: `10`
- seeds per sweep cell: `8`
- collapse baseline threshold episodes in the pilot: `50`

Budget realism note:

- `2,000,000` fine-tune steps is the locked default horizon for the main factorial
- longer horizons require explicit opt-in in the manifest tooling and should only be used after rerunning the pilot with a matching sweep-budget target

Protocol summary:

1. Pretrain a nominal-domain policy.
2. Freeze a baseline return distribution on the shifted evaluation domain.
3. Fine-tune from the pretrained checkpoint in the shifted domain.
4. Emit evaluation diagnostics after warmup-eligible checkpoints.
5. Stop early on first threshold-defined collapse with `stop_on_collapse = True`.

This choice is locked for the paper plan:

- early stopping is retained to avoid wasting compute on runs that have already crossed the collapse threshold
- post-collapse recovery is out of scope for the main study
- time-to-collapse analyses must therefore treat non-collapsing runs as right-censored

Replay and diagnostic-buffer initialization at fine-tune start:

- the main SAC replay buffer is restored from the pretrain checkpoint rather than reset from scratch
- the recent-buffer diagnostic store is reset to empty and then filled only with freshly collected fine-tune transitions

Why this matters:

- optimization at fine-tune start is seeded with pretrain replay contents
- warning diagnostics are intentionally not seeded with pretrain data
- the warmup rows therefore reflect only early fine-tune behavior, not replay carried over from pretraining

Important design note:

The main sweep keeps a shared pretrain checkpoint and varies fine-tune seeds only. Claims from the factorial therefore remain conditional on a fixed pretrained policy. To make that limitation defensible without expanding the full factorial, add a small sensitivity check on one representative cell (`n = 3`, `critic_width = 256`) using `2-3` additional pretrain seeds.

### 4.4 Factorial Sweep

Primary factors:

- backup horizon: `n in {1, 3, 10}`
- critic width: `{256, 1024}`
- fine-tune seeds: `8` per cell

Current full design size:

- `3 x 2 x 8 = 48` fine-tune runs

Actor size is fixed to reduce degrees of freedom and keep the paper centered on stability mechanisms rather than broad architecture search.

### 4.5 Collapse Definition

The current repository defines collapse from a frozen baseline on shifted-domain evaluation returns:

That frozen baseline is computed by evaluating the pretrained policy on the shifted evaluation domain before any fine-tuning updates. In paper language, `mu0` and `sigma0` therefore capture zero-shot performance under shift, not nominal-domain performance.

- `mu0`: baseline mean return
- `sigma0`: baseline return standard deviation
- threshold: `min(mu0 - 2 sigma0, mu0 - 0.2 |mu0|)`

A fine-tune evaluation is labeled collapsed when:

- `return_mean < threshold`

This yields an explicit operational definition that can be reported exactly in the paper.

### 4.6 Warning Signal Definition

The canonical warning signal is `score` in `eval_log.jsonl`.

Current diagnostic semantics:

- diagnostics use recent fine-tune transitions only
- recent-buffer capacity is `10,000` transitions
- the first two eligible eval checkpoints are warmup-only and do not emit rows
- warmup variance is the mean of those two warmup variances
- emitted checkpoints summarize TD-error variance and `q95` absolute TD error
- warning score compares current variance to warmup variance on a log scale
- the default trigger is `score > log(3)` for `2` consecutive evals

Why the buffer size matters:

- it makes the diagnostic explicitly recent rather than replay-wide
- it bounds the temporal window from which TD diagnostics are sampled
- it is part of the method definition, not just an implementation detail

Primary prediction horizon for ROC-style warning evaluation:

- `prediction_horizon_evals = 10`
- at `eval_interval = 100,000`, this is a `1,000,000`-step lookahead

Why `10` evals is the primary horizon:

- it is operationally meaningful for run management, because a million-step warning window leaves time to stop, inspect, or retune a run
- it is still a near-term warning target rather than a diffuse long-horizon risk score
- it matches the current repo default and therefore the existing diagnostic pipeline

The paper should distinguish clearly between:

- the per-eval diagnostic score
- the held warning trigger
- the collapse label based on return threshold
- the prediction horizon used when scoring warning quality

### 4.7 Primary Warning Baseline

The main warning-score comparison is locked:

- primary warning method: recent-buffer TD diagnostic `score`
- primary baseline comparator: raw return-drop rate computed from the same evaluation checkpoints

Recommended baseline definition:

- convert `return_mean` into the locked normalized drop score `(mu0 - return_mean) / max(|mu0|, epsilon)`
- use that return-drop value as the baseline warning score on the same eval schedule as the TD-based score
- define the baseline warning trigger as `2` consecutive evals above `0.5 * threshold_drop_fraction(mu0, threshold)`
- compare both methods with the same downstream metrics: ROC-AUC, lead time, false-positive rate, and usable-warning coverage

Rationale:

- this baseline is cheap
- it is already derivable from the current artifact format
- it gives reviewers a simple question: does the TD-based warning outperform a naive warning based only on observed return degradation?

### 4.8 Primary Outcomes

Primary outcomes:

- collapse incidence by condition
- Kaplan-Meier-style time to first collapse by condition
- environment steps to first warning
- warning lead time before collapse
- global ROC-AUC for predicting collapse within the primary `10`-eval horizon
- comparative warning advantage versus the raw return-drop baseline

Secondary outcomes:

- return trajectories during fine-tuning
- variance and `q95_abs_td` trajectories
- false-positive warning frequency in non-collapsing runs
- interaction between factor settings and warning reliability
- realized-versus-scheduled training budget, reported as a consequence of early stopping rather than a direct quality metric

### 4.9 Pilot Gate and Study Integrity

The repository already includes a pilot gate for deciding whether the full sweep is worth running.

The pilot currently checks:

- at least `2` usable fine-tune seeds
- at least `3` post-warmup eval rows per usable seed
- finite baseline and diagnostic metrics
- conservative `48`-run budget within the configured GPU-hour ceiling
- whether fine-tune drop fractions land in the target band

Suggested paper language:

[Placeholder]

Before the main sweep, we ran a pilot calibration procedure to verify that the shift severity, collapse threshold, and compute budget jointly produced an informative and tractable study regime.

The pilot-to-sweep handoff is a hard gate:

- only a production-profile pilot on the target hardware with `decision = proceed` can authorize manifest generation
- the operator must record the decision note before launching the main factorial or the representative-cell sensitivity bundle
- `adjust` and `fail` both require a new pilot before any sweep or sensitivity manifest is launched

### 4.10 Analysis Plan

[Placeholder to finalize before execution]

Recommended minimum analysis commitments:

- Report per-cell means with uncertainty intervals across seeds.
- Treat each run as the basic unit of analysis.
- For collapse incidence, report binomial uncertainty intervals and effect sizes between cells.
- For time-to-collapse, use Kaplan-Meier-style survival curves with non-collapsing runs right-censored at their final realized evaluation step.
- Report median time-to-collapse only when the survival curve crosses `0.5`; otherwise report restricted-horizon survival summaries.
- For warning quality, report ROC-AUC, mean lead time, false-positive rate, and the count of runs with usable warning-before-collapse events.
- Compare the TD-based warning score against the raw return-drop baseline on the same checkpoints and with the same metrics.
- Treat `prediction_horizon_evals = 10` as the primary horizon and report sensitivity to shorter and longer horizons, using `5`, `10`, and `20` evals as the locked comparison set.
- Treat the primary single-shift factorial as the main result and any added mild/extreme shift runs as sensitivity analysis, not as a second underpowered factorial.
- Do not interpret raw final environment steps or final return as directly comparable across conditions without accounting for early stopping.
- Do not make claims about post-collapse recovery in the main study.
- Treat alignment or misalignment between warning quality and return outcomes as exploratory unless the observed pattern is strong and clearly separated.
- Keep the primary endpoint list short and locked before the sweep begins.

### 4.11 Threats to Validity

Internal validity concerns:

- collapse thresholds depend on finite baseline evaluation samples
- early stopping on collapse changes run length, creates right-censoring, and prevents recovery analysis
- warning metrics are conditioned on sufficient recent-buffer coverage
- private Brax internals may affect future reproducibility if the dependency version changes

External validity concerns:

- one environment and one algorithm family
- one primary shift family in the main factorial
- actor architecture held fixed
- if a shared pretrain is used, pretrain-seed variance is not measured
- with `8` seeds per cell, collapse-incidence uncertainty intervals will still be wide enough that the paper should emphasize large effects and uncertainty rather than fine-grained effect-size claims

## 5. Results Skeleton

This section is intentionally written as a fill-in template.

### 5.1 Setup Verification

[Placeholder]

Report:

- pilot decision
- host/backend summary
- realized run counts
- any deviations from the planned sweep

### 5.2 Main Stability Results

[Placeholder]

Suggested table:

- Table 1. Collapse incidence, Kaplan-Meier-style time-to-collapse summaries, and scheduled-versus-realized training budget by `n-step x critic-width`

Suggested text scaffold:

Under the shifted fine-tuning regime, collapse occurred in [RESULT] of [RESULT] runs. The lowest collapse incidence was observed for [RESULT], while [RESULT] exhibited the highest collapse rate. Kaplan-Meier-style time-to-collapse analysis showed [RESULT]. Because runs stop on first collapse, we do not interpret post-collapse recovery from this study.

### 5.3 Warning-Signal Results

[Placeholder]

Suggested table:

- Table 2. Warning ROC-AUC at the primary `10`-eval horizon, plus false-positive rate, usable-warning coverage, and mean lead time for the TD-based score versus the raw return-drop baseline, by condition

Suggested text scaffold:

At the primary `10`-eval (`1,000,000`-step) horizon, the TD-based warning score achieved a global ROC-AUC of [RESULT], compared with [RESULT] for the raw return-drop baseline. Among runs that eventually collapsed, the first TD-based warning preceded the first collapse by [RESULT] evals on average, versus [RESULT] for the baseline. Sensitivity across `5`, `10`, and `20` eval horizons showed [RESULT]. False positives were concentrated in [RESULT].

### 5.4 Trajectory-Level Analysis

[Placeholder]

Suggested figure:

- Figure 1. Fine-tune return trajectories by condition
- Figure 2. Score trajectories aligned to first collapse
- Figure 3. Variance versus return snapshots over training
- Figure 4. Survival curves for time to first collapse by condition

### 5.5 Ablations and Sensitivity Checks

[Placeholder]

Candidates:

- shared-pretrain versus multiple-pretrain sensitivity on the representative cell (`n = 3`, `critic_width = 256`) with `2-3` alternate pretrain seeds
- mild and extreme shift-severity sensitivity on `1-2` representative cells
- threshold parameter sensitivity (`collapse_c`, `collapse_rho`)
- effect of early stopping on summary metrics
- effect of prediction horizon on warning AUC beyond the locked `5/10/20` horizon set
- optional additional comparator beyond raw return-drop, if one can be added without breaking diagnostic invariants

## 6. Discussion

[Placeholder]

Subsections to fill:

### 6.1 Interpretation

- Which factor most changed stability?
- Did warning quality track true instability or only return degradation?
- Were there settings where better nominal adaptation came with worse predictability?

### 6.2 Implications

- what this means for fine-tuning practice
- whether simple TD diagnostics are operationally useful
- how to use warnings in future adaptive schedulers or stop rules

### 6.3 Limitations

- focused case study scope
- one environment is not enough to support broad generalization claims
- one primary shift severity in the main factorial is thin without added sensitivity checks
- dependence on threshold definition
- lack of real-robot validation
- possible under-measurement of variance if pretrain seeds are fixed
- `8` seeds per cell are adequate for large effects but not generous for precise collapse-incidence estimation

## 7. Conclusion

[Placeholder]

This paper presented a controlled study of fine-tuning collapse under domain shift in a robotic RL vertical slice. Across a factorial comparison of backup horizon and critic capacity, we found [RESULT]. A simple recent-buffer TD diagnostic provided [RESULT] warning utility before threshold-defined collapse. These findings suggest [RESULT], while motivating future work on [RESULT].

## 8. Reproducibility and Artifact Plan

Artifacts already supported by the repo:

- run config files
- fine-tune `eval_log.jsonl`
- `summary.json`
- frozen baseline metadata
- pilot report and decision note
- generated sweep manifest

Paper appendix checklist:

- exact software versions
- exact shift specification
- exact collapse threshold rule
- exact warning-trigger rule
- recent-buffer diagnostic details, including `recent_buffer_capacity = 10,000`
- SAC optimization details, including `gamma`, `tau`, learning rate, batch size, reward scaling, and entropy-coefficient handling
- compute budget accounting
- inclusion and exclusion criteria for runs

## 9. Locked Design Choices

The paper framing is already locked:

- This is a warning-signal paper with stability experiments as supporting evidence.
- The study should be written as a measurement paper about collapse predictability under controlled shift.
- The study should not be framed as a new transfer algorithm, a broad benchmark paper, or a general theory of RL collapse.

The following design choices are locked for the current study plan:

1. Keep the main 48-run sweep on a shared pretrain checkpoint, and add a pretrain-seed sensitivity check on the representative cell (`n = 3`, `critic_width = 256`) with `2-3` alternate pretrain seeds.
2. Keep `stop_on_collapse = True` and analyze time-to-collapse with Kaplan-Meier-style right-censoring. Post-collapse recovery is out of scope for the main study.
3. Use the raw return-drop baseline score `(mu0 - return_mean) / max(|mu0|, epsilon)` and trigger baseline warnings only after `2` consecutive evals above `0.5 * threshold_drop_fraction(mu0, threshold)`.
4. Lock the warning-horizon sensitivity set around the primary `10`-eval horizon. The default plan is `5`, `10`, and `20` evals.
5. Use `2,000,000` fine-tune steps as the default main-sweep horizon. Longer horizons require explicit opt-in in manifest generation and a pilot rerun with a matching sweep-budget target.
6. Lock the pilot-to-sweep handoff rule: no manifest generation without a target-hardware production pilot that returns `decision = proceed` and a recorded decision note.

The remaining design work before launch is narrower:

- Add a shift-severity sensitivity check with at least one milder and one more extreme shift on `1-2` representative cells, and lock those severity values before the sweep.

## 10. Minimal Publishability Checklist

Use this as a pre-launch gate:

- The paper claim matches the actual experimental scope.
- The pilot establishes that the shift is neither trivial nor overwhelmingly destructive.
- The main sweep has enough runs per cell to estimate uncertainty, not just means.
- Warning metrics are evaluated separately from final performance metrics.
- Any fixed-pretrain limitation is stated explicitly.
- All thresholds and stopping rules are written down before the main sweep.
- Planned tables and figures are derivable from existing artifacts.
