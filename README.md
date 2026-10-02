# Shelf Sentinel AI

**Privacy-preserving, explainable loss prevention analytics for seven shoplifting scenarios, with a language model that writes the incident brief and a human who makes the call.**

![Shelf Sentinel AI: four camera contexts with priority alerts](./hero.png)

## Project brief

Shop theft in the United Kingdom is no longer a rounding error on the profit and loss account. Retailers already own the raw material for a better answer: stores are full of cameras, shelf sensors, self-checkout scales and till data, yet almost none of it is watched, and the tools that do watch are often brittle and expensive.

Shelf Sentinel AI treats the problem as reconciliation rather than suspicion. It never asks whether a person looks like a thief. It asks whether the items add up: was everything that left the shelf, hand, basket or till actually accounted for?

This repository is the complete, runnable system behind that idea: a simulator that produces a large labelled corpus of perception streams, the feature extractor, a rule baseline, a learned detector, the alert policy, the explanation layer and the human review workflow.

## Results at a glance

All figures are measured on 46,673 clips from seven stores the model never saw in training, reweighted so that theft makes up one clip in 200, which matches the deployment assumption.

| Measure | Hand written rules | Shelf Sentinel, review tier | Shelf Sentinel, priority tier |
| :-- | --: | --: | --: |
| Precision: alerts that are real theft | 15% | **91%** | **93%** |
| Recall: theft clips that raise an alert | 77% | **91%** | 89% |
| False alerts per 10,000 honest clips | 213.7 | **4.8** | 3.5 |
| Alerts per store per day (2,000 clips a day) | 50.2 | **10.1** | 9.6 |
| Share of value at risk that is flagged | 79% | **95%** | 94% |
| Illustrative net value per store per year | £61,000 | **£88,000** | £87,000 |

![Shelf Sentinel against rule and linear baselines](./model_comparison.png)

## The seven scenarios, with picture examples

Each storyboard below shows what the system sees and nothing more: an anonymous skeleton, item, bag and basket detections, zone outlines and point-of-sale events on a stylised scene.

| Scenario | Camera | What gives it away | The honest lookalike it must not flag |
| :-- | :-- | :-- | :-- |
| Concealment in a bag | Aisle | An item leaves the shelf, then leaves the hand at a personal bag, with no basket, shelf or scan event to explain it | Scan and go shoppers packing their own bag after a purchase |
| Concealment in clothing | Aisle | An item leaves the hand at the waistband and never reappears | A shopper with no basket holding items against the body |
| Shelf sweep | Aisle | Many removals in seconds, most unaccounted for | Colleagues restocking; bulk buyers loading a trolley |
| Self checkout skip scan | Self checkout | More items bagged, or gone from the basket, than scanned | Own bag on the scale; quantity key; a barcode that will not read |
| Ticket switch | Self checkout | The scan happens, but the product seen does not match the barcode and the price is far too low | Loose produce; reduced-to-clear stickers |
| Sweethearting at the till | Staffed till | A cashier passes several items to the packing side without scans | Bulky goods scanned by handset; quantity key |
| Trolley push out | Store exit | Visible goods at the doors, no linked payment, almost no time at a checkout | Paid sales that failed to link; click and collect; colleagues moving trolleys |

### Concealment in a bag

![Storyboard: concealment in a bag](./01_concealment_bag.png)

### Concealment in clothing

![Storyboard: concealment in clothing](./02_concealment_clothing.png)

### Shelf sweep

![Storyboard: shelf sweep](./03_shelf_sweep.png)

### Self checkout skip scan

![Storyboard: self checkout skip scan](./04_skip_scan.png)

![Animated replay: self checkout skip scan](./04_skip_scan.gif)

### Ticket switch

![Storyboard: ticket switch](./05_ticket_switch.png)

### Sweethearting at the till

![Storyboard: sweethearting at the till](./06_sweethearting.png)

### Trolley push out

![Storyboard: trolley push out](./07_push_out.png)

### Honest behaviour that looks similar, correctly left alone

![Storyboard: scan and go, no alert](./08_benign_scan_and_go.png)

![Carry in hand, no alert](./09_benign_carry_in_hand.png)

![Own bag on the scale, no alert](./10_benign_own_bag.png)

### Where it goes wrong

![Missed concealment in low light](./11_missed_concealment.png)

![False alert on a click-and-collect customer](./12_false_alert_collection.png)

## How it works

![Architecture: from anonymous signals to a reviewed alert](./architecture.png)

1. **Perception** runs at the store edge and emits anonymous streams: 17 pose keypoints per tracked person, item-in-hand and bag detections, basket and bagging counts, shelf sensor events, and till activity.
2. **A shared clip schema** holds one tracked person for 12.8 seconds at 5 frames per second: keypoints of shape `(64, 17, 3)`, signals of shape `(64, 15)`, journey context and event metadata.
3. **Sixty-two interpretable features** are computed per clip in four families: pose, item flow, point of sale and context.
4. **The detector** is a gradient-boosted tree classifier over eight classes, trained on complete clips and partial clips so it can score while a clip is still unfolding.
5. **The alert policy** chooses, for every scenario, the lowest threshold that meets a precision target.
6. **An evidence packet** is built for each alert: a timeline of detected events, the features that moved the score and how they compare with normal clips on that camera.
7. **A language model narrator** turns the packet into a five-part brief. The reply is parsed, checked against a schema and passed through a language guard.
8. **A human reviewer** reads the brief, looks at the storyboard, rules out innocent explanations and decides what to do.

## The data

The corpus holds 250,000 single-person track clips: 16 million frames, or 889 hours of tracked footage, across 40 simulated stores in three formats (convenience, supermarket and superstore).

![Corpus composition by behaviour and camera](./dataset_composition.png)

The corpus is simulated because real, labelled, multi-sensor footage of shoplifting cannot be shared. The [data card](./data_card.md) documents the schema, the behaviours, the sensor model and the evaluation assumptions.

## Findings

### 1. The till is the best camera in the building

![Sensor ablation: recall by scenario as sensor families are added](./sensor_ablation.png)

### 2. A learned detector beats rules on the scenarios that matter most to customers

![Recall and precision by scenario](./per_scenario_performance.png)

### 3. The base rate decides whether the product is usable

![Alert precision against theft prevalence](./precision_vs_prevalence.png)

### 4. More alerts stop paying for themselves quickly

![Net value against reviewer workload](./economics_curve.png)

### 5. The remaining false alerts have a name

![False alert rate by benign behaviour](./false_alert_sources.png)

### 6. Low light costs seven points of recall

![Robustness by camera quality, crowding and store format](./robustness_slices.png)

### 7. Scoring a clip live is a different problem from scoring it afterwards

![False alerts when scoring live against scoring complete clips](./streaming_false_alerts.png)

### 8. Where live alerts are used, they arrive in time to act

![Detection latency by scenario](./detection_latency.png)

## The language model layer

A risk score of 0.97 is not something a duty manager can act on. The narrator turns each alert into a brief that can be read in ten seconds.

How the layer is built:

* Closed evidence: the model receives a JSON evidence packet and never sees video or a face.
* Behaviour, never people: the prompt bans descriptions of appearance and demographic attributes.
* Checked output: the reply must parse as JSON with required fields and pass a language guard.
* Two narrators: the default template narrator and optional Claude-based narrator.

## Responsible use

A system like this can do harm if it is deployed carelessly, so the safeguards are part of the design and are documented in [responsible_use.md](./responsible_use.md).

* It models what happens to items, not who a person is.
* It produces review requests, not accusations.
* It is evaluated for honest shoppers first: precision targets, false alerts per honest clip, and named sources of false alerts are headline metrics.
* It should not be used to detain, search or ban anyone, or without data protection impact assessment and clear signage.

## Quick start

```bash
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/SENTINEL-AI.git
cd SENTINEL-AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python quickstart.py
```

If the repo already contains the trained model, feature shards and sample data, the example above works immediately. To rebuild the full corpus and evaluation assets from the seed, use the project’s existing automation and scripts in the repository root.

## Repository map

```text
.
├── architecture.md
├── architecture.png
├── behaviour scripts and simulation code
├── data_card.md
├── feature_dictionary.md
├── hero.png
├── model_card.md
├── quickstart.py
├── real_footage.md
├── responsible_use.md
├── scenario_playbook.md
├── sentinel.joblib
├── 01_concealment_bag.png
├── 02_concealment_clothing.png
├── 03_shelf_sweep.png
├── 04_skip_scan.png
├── 05_ticket_switch.png
├── 06_sweethearting.png
├── 07_push_out.png
├── 08_benign_scan_and_go.png
├── 09_benign_carry_in_hand.png
├── 10_benign_own_bag.png
├── 11_missed_concealment.png
├── 12_false_alert_collection.png
├── metrics.json
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
└── README.md
```

## Further reading in this repository

* [Architecture](./architecture.md)
* [Data card](./data_card.md)
* [Model card](./model_card.md)
* [Scenario playbook](./scenario_playbook.md)
* [Feature dictionary](./feature_dictionary.md)
* [Responsible use](./responsible_use.md)
* [Real footage guide](./real_footage.md)

## License

Released under the MIT licence.
