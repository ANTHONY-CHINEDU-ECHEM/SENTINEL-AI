# Architecture

Shelf Sentinel AI is a pipeline of small, testable stages that share one data contract. This note explains each stage, the design decisions behind it, and where to look in the code.

![Architecture](images/architecture.png)

## One contract: the clip schema

Everything downstream of perception consumes a `ClipBatch` (see `src/shelfsentinel/schema.py`):

| Field | Shape | Content |
| :-- | :-- | :-- |
| `keypoints` | `(N, 64, 17, 3)` | COCO pose keypoints as x, y and confidence, in normalised image coordinates |
| `signals` | `(N, 64, 15)` | Item in hand, shelf events, basket and bagging counts, point of sale quantity, product match, price ratio, personal bag and container boxes, security gate alarm |
| `meta` | `N` rows | Camera context, store, journey context (linked payment, checkout dwell) and, for simulated clips, the ground truth label |

A clip follows one tracked person for 12.8 seconds at 5 frames per second. That window is long enough to contain a pick and a concealment, short enough to keep alerts timely, and cheap to compute on. No pixels are stored anywhere in the system.

Because the contract is narrow, the simulator and a real perception stack are interchangeable. `shelfsentinel.adapters` converts the output of any COCO pose tracker into the same schema.

## Stage by stage

### Simulation (`simulation/`)

* `script.py` is a small keyframe language. A behaviour is written as readable actions (reach to the shelf, hold at chest height, move the hand to the bag) and resampled to frames with eased interpolation.
* `behaviours.py` holds the 32 behaviours: 7 theft scenarios and 25 benign ones. Each is a plain function, registered with its camera context, label and relative frequency.
* `kinematics.py` turns the latent state into a skeleton. Arms use two link inverse kinematics so reach is always physically possible.
* `perception.py` is the sensor model: keypoint jitter and dropout by camera quality, occlusion bursts in crowds, missed item detections when a hand is hidden by the body, shelf sensor misses, counter drift and flicker, saturation of counts in a heaped trolley, till clock skew and missed bag detections.
* `generator.py` builds store profiles and streams the corpus shard by shard, so sixteen million frames never need to be in memory at once.

### Features (`features/`)

`extractor.py` computes 62 features in one vectorised pass. The core idea is accounting: every item that leaves the shelf or the basket should be explained by a basket gain, a shelf return, a scan, a bagging gain, or an item still in the hand. The important derived events are:

* a **vanish**: an item stops being visible in a hand and stays gone for at least 0.8 seconds, which filters detector flicker;
* an **unexplained vanish**: a vanish with no basket, shelf, bagging or scan event near it in time, classified by where the hand was (personal bag, waist, elsewhere);
* a **transfer**: an item carried to the bagging side at a counter, with the time it spent over the scanner.

The extractor works on any clip prefix, which is what makes streaming replay possible. `catalog.py` gives every feature a plain language label that is reused in the documentation and the incident briefs.

### Detection (`models/`)

* `rules.py` is a transparent decision list. It is the baseline, a fallback detector and a useful statement of what a human expert would write.
* `sentinel.py` wraps a histogram gradient boosted classifier over eight classes. Trees were chosen over a deep sequence model because the features are tabular and interpretable, training takes about a minute on one core, and the model file is about one megabyte. Isotonic calibration is fitted on half of the validation stores and kept only if it improves calibration on the other half.
* `policy.py` converts probabilities into tiers. For each scenario it finds the lowest threshold whose precision meets the target when clips are reweighted to the deployment theft rate.

### Streaming aware training

A detector trained only on complete clips misreads partial ones: an item that has been picked up but not yet basketed looks unaccounted for. The corpus therefore includes a second table of partial clips cut at random lengths, labelled by what has happened so far (a concealment clip is still normal shopping until the concealment). Rows cut close to the act are dropped as ambiguous. In live use an alert must also hold for two evaluations in a row before it is raised.

### Evaluation (`evaluation/`)

`metrics.py` reports everything at deployment prevalence: precision, recall, false alerts per honest clip, robustness slices, the sources of false alerts, and an illustrative economic model. `replay.py` replays clips frame by frame to measure detection latency and to compare whole clip scoring with live scoring.

### Explanation (`explain/`)

* `evidence.py` builds the evidence packet: a timeline of detected events and the features that moved the score. A feature's evidence weight is the drop in the log odds of the flagged scenario when that feature is replaced by its typical value for normal clips on the same camera.
* `playbook.py` holds, for each scenario, the innocent explanations and the proportionate response for each tier.
* `prompts.py` and `narrator.py` implement the narrators. The Claude narrator is constrained by a system prompt, a JSON schema check and a language guard, and falls back to the template narrator on any failure.

### Visualisation (`viz/`)

`render.py` draws frames, `storyboard.py` composes four frame storyboards and animated replays, `gallery.py` selects and renders the examples, and `charts.py` draws every report chart from the files written by the pipeline.

## Design principles

1. **Reconcile items; do not judge people.** Features describe what happened to merchandise. Nothing describes who a person is.
2. **Evaluate at the real base rate.** Every headline metric is weighted to the deployment theft rate.
3. **Split by store.** Generalisation to new stores is the question that matters commercially.
4. **Explanations from closed evidence.** The narrator can only restate facts the pipeline detected.
5. **Fail safe.** Any narrator failure yields a deterministic brief; any scenario whose precision target cannot be met is switched off rather than run loose.
6. **Everything is reproducible.** One seed, one config file, one command.
