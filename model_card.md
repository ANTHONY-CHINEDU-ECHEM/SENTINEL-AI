# Model card: Shelf Sentinel detector

This card is generated from `reports/metrics.json` by `scripts/build_assets.py`, so it always describes the committed model.

## Overview

| Item | Detail |
| :-- | :-- |
| Task | Classify a 12.8 second single person track clip as normal shopping or one of seven theft scenarios |
| Model | Histogram gradient boosted trees (scikit learn), eight classes |
| Inputs | 62 interpretable features from pose, item flow, point of sale and context |
| Outputs | Class probabilities, the most likely theft scenario, its risk, and a tier: clear, review or priority |
| Training data | 162,803 complete clips from 26 simulated stores, plus partial clips for streaming aware training |
| Boosting iterations | 120 (early stopping) |
| Calibration | Isotonic calibration tested on validation stores and rejected; test calibration error 0.0059 |
| Alert policy | Per scenario thresholds for 80% (review) and 95% (priority) precision at a theft rate of 0.5%, with a safety margin of 1 phantom false alert on the review tier |
| Artefact | `models/sentinel.joblib` (estimator, policy and per camera reference statistics) |

## Intended use

Ranking clips for human review inside a loss prevention workflow, with the evidence packet and incident brief. The model must not be used to make automated decisions about a person. See [responsible use](responsible_use.md).

## Evaluation data

46,673 clips from 7 stores that were never used for training, calibration or thresholds. Alert metrics are reweighted so that theft is 0.5% of clips.

## Classification by most likely class

Accuracy 0.9906, macro F1 0.9706, macro F1 over the seven theft scenarios 0.9671, theft against normal ROC AUC 0.9990, average precision at the deployment rate 0.9615.

| Scenario | Precision | Recall | F1 | Test clips |
| :-- | --: | --: | --: | --: |
| Normal shopping | 99.2% | 99.7% | 0.995 | 39,773 |
| Concealment in a bag | 97.1% | 92.5% | 0.947 | 1,078 |
| Concealment in clothing | 98.2% | 97.4% | 0.978 | 1,074 |
| Shelf sweep | 98.8% | 95.1% | 0.969 | 933 |
| Self checkout skip scan | 95.6% | 86.6% | 0.909 | 1,001 |
| Ticket switch | 100.0% | 100.0% | 1.000 | 735 |
| Sweethearting at the till | 98.7% | 98.8% | 0.988 | 757 |
| Trolley push out | 98.3% | 97.6% | 0.980 | 1,322 |

## Alert tiers at the deployment theft rate

| Measure | Hand written rules | Review tier | Priority tier |
| :-- | --: | --: | --: |
| Precision | 15.3% | 90.6% | 92.7% |
| Recall | 77.0% | 91.5% | 89.4% |
| Alerts per 10,000 clips | 251.2 | 50.5 | 48.2 |
| False alerts per 10,000 honest clips | 213.71 | 4.78 | 3.52 |

| Scenario | Review recall | Review precision | Priority recall | Priority precision | Review threshold |
| :-- | --: | --: | --: | --: | --: |
| Concealment in a bag | 82.9% | 100.0% | 74.3% | 100.0% | 0.95 |
| Concealment in clothing | 98.0% | 79.3% | 97.4% | 79.3% | 0.38 |
| Shelf sweep | 93.3% | 100.0% | 93.3% | 100.0% | 0.88 |
| Self checkout skip scan | 74.7% | 84.4% | 72.2% | 87.5% | 0.96 |
| Ticket switch | 100.0% | 100.0% | 100.0% | 100.0% | 0.93 |
| Sweethearting at the till | 97.4% | 95.5% | 96.6% | 95.5% | 0.81 |
| Trolley push out | 96.4% | 86.0% | 95.5% | 94.8% | 0.74 |

## Scoring clips as they unfold

Measured on 4,000 fresh clips from the test stores, replayed frame by frame. A live alert must hold for 2 evaluations in a row.

| Measure | Trained on whole clips only | Streaming aware (shipped) |
| :-- | --: | --: |
| False alerts per 1,000 honest clips, scored once complete | 0.29 | 0.29 |
| False alerts per 1,000 honest clips, live, unconfirmed | 42.55 | 8.45 |
| False alerts per 1,000 honest clips, live, confirmed | 17.49 | 4.66 |
| Theft recall, live, confirmed | 92.6% | 93.2% |

| Scenario | Caught inside the clip | Median seconds from act to alert | 90th percentile |
| :-- | --: | --: | --: |
| Concealment in a bag | 83.3% | 1.1 after | 3.9 after |
| Concealment in clothing | 99.3% | 0.9 after | 1.4 after |
| Shelf sweep | 90.7% | 0.8 after | 1.7 after |
| Self checkout skip scan | 83.3% | 1.4 after | 4.7 after |
| Ticket switch | 100.0% | 0.1 before | 0.3 after |
| Sweethearting at the till | 97.3% | 0.0 after | 1.0 after |
| Trolley push out | 100.0% | 6.1 before | 2.6 before |

Before means the alert fired ahead of the moment the act completed.

## Factors that change performance

See `reports/tables/robustness_slices.csv`. Low light cameras reduce recall and raise false alerts; crowding has a smaller effect; store format has little effect. Precision depends strongly on the true theft rate (`reports/tables/precision_vs_prevalence.csv`).

## Limitations

* Trained and tested on synthetic data. Real world accuracy is unknown.
* Thresholds are fitted to rare events and will need refitting per store group on real data.
* Features assume a correct single person track and calibrated camera zones.
* Capture conditions (camera quality, crowding, hour) are deliberately not model inputs, so the model cannot compensate for them explicitly.
* No demographic attributes exist in the data, so fairness across groups of people cannot be measured here and must be tested on real data before any deployment.

## Reproduce

```bash
make train
python scripts/build_assets.py
```
