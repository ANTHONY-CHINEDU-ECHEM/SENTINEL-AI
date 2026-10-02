# Shelf Sentinel AI

Privacy preserving, explainable loss prevention analytics for seven shoplifting scenarios. A system that reconciles items against receipts rather than profiling people.

![Shelf Sentinel AI system overview](./hero.png)

## Overview

Shop theft costs UK retailers over 16 billion pounds annually. While stores deploy extensive camera networks and sensor systems, most footage goes unwatched and detection tools are often brittle or expensive.

Shelf Sentinel AI flips the approach. Instead of asking "Does this person look suspicious?", it asks "Do the items add up?" The system reconciles what people take from shelves, carry in baskets, scan at checkout and walk out with against what was actually purchased.

This repository contains the complete production system: a simulator that generates labeled training data, the feature extraction pipeline, detection models, alert policies, explanation layer and human review workflow.

## Key Performance Metrics

Evaluated on 46,673 clips from seven unseen stores, with theft reweighted to match 1 in 200 deployment baseline.

| Metric | Rule Baseline | Review Tier | Priority Tier |
| :-- | --: | --: | --: |
| Precision (real theft alerts) | 15% | 91% | 93% |
| Recall (theft clips caught) | 77% | 91% | 89% |
| False alerts per 10k honest clips | 213.7 | 4.8 | 3.5 |
| Alerts per store per day | 50.2 | 10.1 | 9.6 |
| Value at risk flagged | 79% | 95% | 94% |
| Net value per store per year | £61,000 | £88,000 | £87,000 |

![Model performance comparison](./model_comparison.png)

## The Seven Shoplifting Scenarios

The system detects seven distinct behaviors. Each section shows what the detector sees: anonymous pose skeleton, item positions, hand pose, bag and basket activity, zone geometry and point of sale transactions.

| Scenario | Location | Detection Signal | Legitimate Lookalike |
| :-- | :-- | :-- | :-- |
| Concealment in bag | Aisle | Item taken from shelf, placed in personal bag with no scan or explanation | Customer using scan and go service, packing own purchases |
| Concealment in clothing | Aisle | Item disappears at waistband or neckline and never reappears | Customer carrying loose items against body before checkout |
| Shelf sweep | Aisle | Multiple items removed in seconds, most unaccounted for | Staff restocking shelves or bulk buyer loading cart |
| Self checkout skip scan | Self checkout | Bagging count rises without matching scans | Customer using own bag on scale or quantity key |
| Ticket switch | Self checkout | Item scanned but product detected differs, price far too low | Loose produce or reduced to clear merchandise |
| Sweethearting at till | Staffed checkout | Cashier passes multiple items to packing area without scanning | Operator using handset scanner or bulk void key |
| Trolley push out | Store exit | Goods visible at doors with no payment link, minimal time in checkout | Failed payment link, click and collect order, or staff moving items |

### Concealment in Bag

![Concealment in bag example](./01_concealment_bag.png)

### Concealment in Clothing

![Concealment in clothing example](./02_concealment_clothing.png)

### Shelf Sweep

![Shelf sweep example](./03_shelf_sweep.png)

### Self Checkout Skip Scan

![Skip scan example](./04_skip_scan.png)

Animated view showing real time detection:

![Skip scan animation](./04_skip_scan.gif)

### Ticket Switch

![Ticket switch example](./05_ticket_switch.png)

### Sweethearting at Till

![Sweethearting example](./06_sweethearting.png)

### Trolley Push Out

![Trolley push out example](./07_push_out.png)

### Legitimate Behavior Correctly Excluded

The system must distinguish genuine shopping from theft. Examples below show honest behavior that triggers no alerts.

![Scan and go legitimate behavior](./08_benign_scan_and_go.png)

Customer using scan and go service, packing items into personal bag immediately after scanning each item.

![Carrying items by hand](./09_benign_carry_in_hand.png)

Customer carrying items in hand before reaching basket or checkout.

![Own bag on scale](./10_benign_own_bag.png)

Customer placing personal shopping bag on self checkout baggage scale.

### Detection Failures and False Alerts

The detector does make mistakes. These examples show realistic failure cases from the test set.

![Missed concealment in low light](./11_missed_concealment.png)

Low light camera in crowded store causes skeleton tracking to drop out, allowing concealment to go undetected.

![False alert on click and collect](./12_false_alert_collection.png)

Click and collect customer picking up pre paid order incorrectly flagged as theft due to minimal time in checkout.

## System Architecture

![System architecture diagram](./architecture.png)

The complete pipeline flows from perception to human decision making.

1. Perception layer runs at store edge emitting anonymous streams: 17 pose keypoints per tracked person, hand and bag position detections, basket and bagging counts, shelf sensor events and till activity.

2. Clip schema packages 12.8 seconds of continuous tracking at 5 frames per second into keypoints of shape (64, 17, 3) plus signals of shape (64, 15) with journey metadata.

3. Feature extraction computes 62 interpretable features across four families: pose kinematics, item flow dynamics, point of sale alignment and contextual factors. Each feature has plain language documentation.

4. Detection model is a gradient boosted tree classifier trained on both complete clips and partial clips to enable live scoring as events unfold.

5. Alert policy sets per scenario thresholds to meet precision targets at deployment theft prevalence. Separate thresholds for review tier (80% precision) and priority tier (95% precision).

6. Evidence packet compiles for each alert: timeline of detected events, features influencing the decision, comparison with normal behavior on that camera and data quality indicators.

7. Language model narrator converts evidence packet into five part structured brief readable in ten seconds, with parsed output validation and content filtering.

8. Human reviewer reads structured brief, examines visual storyboard, rules out innocent explanations using scenario playbook and makes final decision.

## Training Dataset

The corpus contains 250,000 single person track clips representing 16 million frames or 889 hours across 40 simulated stores in three formats (convenience, supermarket and superstore).

![Dataset composition](./dataset_composition.png)

Key dataset design choices:

Hard negatives are first class. The model succeeds or fails on honest behaviors resembling theft: scan and go, own bag use, quantity key, reduced to clear, void transactions, handling by staff and bulk purchases.

Theft is enriched then reweighted. Base theft rate is 13.7 percent ensuring each scenario has over 4,000 examples. All metrics are reweighted to 1 in 200 deployment prevalence for accurate precision reporting.

Stores are split never clipped. 26 stores train the model, 7 calibrate thresholds, 7 remain held out for testing stratified by format. This measures transfer learning not memorization.

Noise is part of the label. Low light, crowding and occlusion degrade perception. Some benign clips are intentionally ambiguous so ceiling performance is naturally below 100 percent.

Reproducible bit for bit. The entire 250,000 clip corpus generates from a single seed in eight minutes on one CPU core. Feature shards are committed so training is deterministic.

The corpus is entirely synthetic because real labeled multi sensor footage of shoplifting cannot be ethically or legally shared. See the data card for documentation of sensor models, behavior scripts and evaluation methodology.

## Research Findings

### Finding 1: Till is Most Informative Sensor

![Sensor ablation chart](./sensor_ablation.png)

Training on expanding sensor sets reveals information distribution. Pose alone reaches macro F1 of 0.58 and misses ticket switches and sweethearting entirely. Till sensors add nearly all remaining signal.

### Finding 2: Learning Beats Handwritten Rules

![Per scenario performance](./per_scenario_performance.png)

Rule baseline achieves respectable 77 percent recall but only 15 percent precision. Weakest on scenarios closest to honest behavior: 6 percent precision on skip scan, 8 percent on sweethearting. Learned model outperforms on every scenario.

### Finding 3: Base Rate Determines Usability

![Precision versus theft rate](./precision_vs_prevalence.png)

Same model with same thresholds delivers different precision at different theft prevalences. At 1 in 1000 theft precision is 66 percent. At 1 in 200 it reaches 91 percent. High base rate substantially improves practical usability.

### Finding 4: Economics Curve Shows Diminishing Returns

![Economics curve](./economics_curve.png)

Net value per store peaks around 13 alerts per day then declines as analyst time and management overhead grow faster than recovered value.

### Finding 5: False Alerts Cluster in Specific Behaviors

![False alert sources](./false_alert_sources.png)

Only 19 of 39,773 honest test clips trigger review tier alerts. These are not random. Click and collect customers are flagged at 20 per 1000, over ten times any other behavior. Targeted handling of this case eliminates most false alerts.

### Finding 6: Lighting and Crowding Impact Recall

![Robustness across conditions](./robustness_slices.png)

Recall drops from 94 percent on HD cameras to 86.5 percent in low light, costing seven points. False alert rate rises five fold from 0.2 to 1.1 per 1000 honest clips. Crowding causes less degradation than lighting.

### Finding 7: Live Scoring is Different Problem

![Live versus batch scoring](./streaming_false_alerts.png)

Model trained only on complete clips raised transient alerts on 43 per 1000 honest clips when scored frame by frame. Adding confirmation rules brought this to 2.7 per 1000. Live scoring requires different training or post processing.

### Finding 8: Detection Latency Enables Action

![Detection latency by scenario](./detection_latency.png)

With confirmation rules in place, concealment alerts fire median 1 second after item disappears and skip scan alerts 1.4 seconds after unscanned item enters bag. This provides actionable time at checkouts.

### Finding 9: Three Honest Caveats

Thresholds set on rare events did not transfer. Validation set precision targets were met on training stores but failed on held out test stores. Alert policy was redesigned to account for this.

Priority tier misses its 95 percent precision target, delivering 93 percent instead. At deployment prevalence 95 percent precision is mathematically difficult to achieve with finite test set and natural noise floor.

Isotonic calibration made validation calibration error worse so raw probabilities were retained. Calibration testing showed no benefit from post processing.

## Explanation and Narrative

A raw risk score of 0.97 means nothing to a store manager. The explanation layer turns each alert into a brief that is comprehensible and actionable in ten seconds.

The layer works through:

Closed evidence architecture. The language model receives structured JSON packet with event timeline, feature values and camera statistics. It never sees video, faces, or unrestricted external knowledge.

Behavior not people. System prompt explicitly bans appearance descriptions, demographic information, accusatory language and personally identifying details. All explanations reframe findings as item reconciliation.

Validated output. All narratives must parse as valid JSON with required fields. Content is checked against language guard that rejects demographic terms or accusatory framing.

Dual narrator design. Template narrator runs offline as default. Claude based narrator optionally connects through Anthropic API for more sophisticated explanation when needed.

## Responsible Deployment

This system can cause harm if deployed without safeguards. Design principles are documented in responsible use guide.

The system models items not people. No faces, no identities, no demographic inputs and no memory of individuals between separate clips.

System produces review requests not accusations. Every alert includes innocent explanations and playbook response includes offer of help before any investigation.

Evaluation prioritizes honest shoppers. Precision targets, false alert rates on legitimate behavior and named sources of false alerts are primary metrics.

System should never be used to detain, search or ban anyone, to score staff performance, or without data protection impact assessment and clear public signage. Staff facing scenarios route through separate adjudication paths.

## Quick Start

Clone the repository and set up environment:

```
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/SENTINEL-AI.git
cd SENTINEL-AI
python m venv .venv
source .venv/bin/activate
pip install r requirements.txt
python quickstart.py
```

The repository includes trained model, feature shards and sample clips so this runs immediately. To rebuild all assets from scratch:

```
make all
```

For rapid iteration on a subset:

```
make quick
```

Use as a Python library:

```python
from shelfsentinel.explain import TemplateNarrator, build_evidence
from shelfsentinel.features import extract_features
from shelfsentinel.models import SentinelModel
from shelfsentinel.schema import ClipBatch

clips = ClipBatch.load("data/sample/clips_sample.npz")
model = SentinelModel.load("models/sentinel.joblib")

assessment = model.assess(extract_features(clips))
evidence = build_evidence(model, clips, index=9)
print(TemplateNarrator().narrate(evidence).to_markdown())
```

## Repository Structure

```
SENTINEL-AI/
├── README.md
├── architecture.md
├── architecture.png
├── data_card.md
├── feature_dictionary.md
├── model_card.md
├── responsible_use.md
├── scenario_playbook.md
├── real_footage.md
├── hero.png
├── sentinel.joblib
├── quickstart.py
├── 01_concealment_bag.png
├── 02_concealment_clothing.png
├── 03_shelf_sweep.png
├── 04_skip_scan.png
├── 04_skip_scan.gif
├── 05_ticket_switch.png
├── 06_sweethearting.png
├── 07_push_out.png
├── 08_benign_scan_and_go.png
├── 09_benign_carry_in_hand.png
├── 10_benign_own_bag.png
├── 11_missed_concealment.png
├── 12_false_alert_collection.png
├── model_comparison.png
├── dataset_composition.png
├── per_scenario_performance.png
├── precision_vs_prevalence.png
├── economics_curve.png
├── false_alert_sources.png
├── robustness_slices.png
├── detection_latency.png
├── sensor_ablation.png
├── streaming_false_alerts.png
├── confusion_matrix.png
├── calibration.png
├── feature_importance.png
├── detection_latency.csv
├── economics_curve.csv
├── false_alert_sources.csv
├── precision_vs_prevalence.csv
├── robustness_slices.csv
├── sensor_ablation.csv
└── metrics.json
```

## Documentation

Full documentation is available in the following files:

Architecture: System design and data flow documentation
Data Card: Dataset composition, behaviors and sensor model specification
Model Card: Training procedure, hyperparameters and validation results
Feature Dictionary: Complete reference of 62 extracted features
Scenario Playbook: Response procedures for each detection type
Responsible Use: Deployment safeguards and ethical considerations
Real Footage Guide: Integration path from synthetic training to production systems

## Limitations and Future Work

The corpus is entirely synthetic. Behaviors are scripted from loss prevention domain practice. Real world perception errors are messier and more varied than the sensor model captures.

Single person tracking. The system handles one tracked individual per clip. Distraction teams, handoffs between people and coordinated theft are not modeled.

Economics use illustrative costs. Analysis shows the shape of cost benefit tradeoff but not a complete business case for deployment.

Live scoring still generates higher false alert rates than batch processing. Two alert tiers currently sit below their precision targets on certain scenarios.

Future work will include validation on real footage from PoseLift and other trackers, sequence models over raw sensor channels, per store group threshold learning with uncertainty quantification and active learning pipelines for continuous model refinement.

## Citation and References

British Retail Consortium Crime Report 2026: Comprehensive analysis of retail theft trends and costs in UK market.

Rashvand and colleagues, Exploring Pose Based Anomaly Detection for Retail Security: A Real World Shoplifting Dataset and Benchmark, WACV Workshops 2025: Foundational work on pose based detection for loss prevention.

## License

Released under MIT License. See LICENSE file for complete terms.

## Contact

For questions or collaboration inquiries please open an issue in the repository.
