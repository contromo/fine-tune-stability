# Figures and Tables

This file contains paste-ready captions, tables, and prose snippets for the post-mortem manuscript in [paper_submission.md](./paper_submission.md).

## Figure Files

- Figure 1 SVG (repo copy): `/Users/manav/code/fine-tune-stability/docs/figures/figure1_warning_pilot_summary.svg`
- Figure 2 SVG (repo copy): `/Users/manav/code/fine-tune-stability/docs/figures/figure2_horizon_final_return.svg`

## Figure 1 Caption

**Figure 1. Warning-signal pilot outcome.** Blue bars show the number of runs with at least one warning event in each pilot diagnostic summary, red bars show the number of runs with threshold-defined collapse, and black points show global ROC-AUC when defined. Both warning-focused pilots were run only on the representative cell `n = 1`, `critic_width = 256`. In `pilot_gate_1m_v3`, the selected severe shift produced no warnings and no collapses across `3` reported runs. In `pilot_gate_1m_v4`, tightening the collapse threshold yielded `3` collapsed runs but still `0` warning-positive runs, with ROC-AUC `0.298`. The sample is small, so the AUC should be read cautiously; the stronger failure signal is simply that no warning-positive rows ever appeared.

## Figure 2 Caption

**Figure 2. Final fine-tune return by backup horizon under the fixed severe shift.** Each point is one seed's final evaluation return after `1,000,000` fine-tune environment steps. Horizontal black markers show the group mean. The `n = 1` group remains broad, with several seeds still near zero, but it is materially separated from the near-zero `n = 3` and `n = 10` groups. This is a secondary robustness observation rather than the paper's lead claim because the design is horizon-matched, unbalanced, and not monotone within the longer-horizon pair; with only `4` seeds each, `n = 3` and `n = 10` should be treated as practically indistinguishable from each other on the scale of the `n = 1` separation.

## Table 1 Caption

**Table 1. Warning-focused pilot outcomes for the representative cell.** Both pilots were run only at `n = 1`, critic width `256`, under the same severe fine-tune shift `(friction = 0.10, payload = 2.2)`. `pilot_gate_1m_v3` solved the compute-budget and degradation-band problem but yielded no warning or collapse events across `3` reported runs. `pilot_gate_1m_v4` created threshold-defined collapses by lowering `collapse_c`, but still produced zero warning-positive runs and ROC-AUC `0.298`, which is best interpreted as uninformative in a tiny-sample regime rather than as a strong signed ranking result.

## Table 1 Markdown

| Pilot | Shift `(friction, payload)` | `collapse_c` | Drop Fraction Mean | Threshold-Drop Mean | Collapsed Runs | Warning-Positive Runs | Global ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilot_gate_1m_v3` | `(0.10, 2.2)` | 2.0 | 0.3427 | 1.9430 | 0 | 0 | undefined |
| `pilot_gate_1m_v4` | `(0.10, 2.2)` | 0.5 | 0.7486 | 0.7077 | 3 | 0 | 0.2982 |

## Table 2 Caption

**Table 2. Secondary robustness comparison by backup horizon.** Values are the final evaluation return means recorded at the end of each `1,000,000`-step fine-tune run under the same severe shift used in the warning pilots. The directional result is strong, but the comparison is secondary because each horizon used its own pretrained checkpoint and the seed counts are unbalanced.

## Table 2 Markdown

| Horizon | Seeds | Mean Final Return | Std. Dev. | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n = 1` | 8 | 0.5310 | 0.5109 | 0.0078 | 1.4597 |
| `n = 3` | 4 | 0.0099 | 0.0043 | 0.0054 | 0.0155 |
| `n = 10` | 4 | 0.0229 | 0.0049 | 0.0169 | 0.0271 |

## Post-Mortem Paragraph For Section 4

The warning design failed in two stages. In `pilot_gate_1m_v3`, the severe shift was finally informative on degradation and acceptable on projected compute budget, but the diagnostic summary still yielded neither warnings nor collapses across `3` reported runs. In `pilot_gate_1m_v4`, lowering `collapse_c` created threshold-defined collapses in all `3` reported runs, yet warnings remained absent and ROC-AUC was only `0.298`. The cleaner reading is not that the warning score was provably inverted; it is that label creation became possible without any corresponding increase in warning utility.

## Mechanism Paragraph For Section 4

The failure is consistent with the geometry of the implementation. The warning score is normalized to the first two shifted-domain warmup evaluations rather than to nominal-domain behavior; those two warmup measurements emit no warning rows; the ROC-AUC label asks whether collapse occurs in any of the next `10` evaluations; and tightening the collapse threshold changes labels more than it changes underlying dynamics. Together, these choices create a regime in which degradation and even threshold-defined collapse can occur without producing usable early-warning separation in the recent-buffer TD score.

## Secondary Observation Paragraph For Section 6

The later horizon comparison is better treated as a secondary observation than as the lead scientific claim. Under the same severe shift, the `n = 1` configuration retained partial performance on average (`8` seeds, mean final return `0.531`), while both longer-horizon configurations finished near zero (`n = 3`: mean `0.0099`; `n = 10`: mean `0.0229`). That directional separation is real, but it is not a clean causal ablation because the design is horizon-matched, unbalanced, and not monotone within the longer-horizon pair.

## Submission Notes

- Lead with Figure 1 and Table 1. They carry the actual post-mortem contribution.
- Keep Figure 2 and Table 2 as secondary support, not as the first result.
- If space is tight, Table 2 can be compressed into prose, but Table 1 should stay because it makes the failure sequence legible.
- The warning score in the manuscript should be defined as variance over a recent-buffer TD snapshot, not just as an abstract `var_t`, because that detail is load-bearing for the post-mortem.
