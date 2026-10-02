# Shelf Sentinel AI

**Privacy preserving, explainable loss prevention analytics for seven shoplifting scenarios, with a language model that writes the incident brief and a human who makes the call.**

![Shelf Sentinel AI: four camera contexts with priority alerts](docs/images/hero.png)

## Project brief

Shop theft in the United Kingdom is no longer a rounding error on the profit and loss account. The British Retail Consortium's 2026 crime report counts 5.5 million detected incidents of shop theft in a single year, at a direct cost of almost £400 million, and it is explicit that the real figure is higher because most incidents are never detected. The same report records around 1,600 incidents of violence and abuse against shop workers every day, and the shop workers' union Usdaw finds that about two thirds of attacks on staff are triggered by theft or armed robbery. Loss prevention is therefore a people problem as much as a margin problem: every confrontation that can be avoided, and every honest customer who is not wrongly challenged, matters.

Retailers already own the raw material for a better answer. Stores are full of cameras, shelf sensors, self checkout scales and till data, yet almost none of it is watched, and the tools that do watch tend to fail in one of two ways. Hand written rules fire constantly because honest shopping is full of behaviour that looks odd in isolation: checking a phone, using scan and go, carrying two items with no basket, buying yellow sticker reductions. Vision systems that claim to spot "suspicious behaviour" create a different risk, because a model that scores how a person looks or moves invites bias, cannot explain itself, and produces accusations rather than evidence. There is also a point that is easy to miss: shoplifting is not one behaviour. Hiding a bottle in a tote bag, sliding a steak past a self checkout scanner, a cashier waving a friend's shopping through, and pushing a full trolley out of the door are four different mechanisms, seen by four different cameras, and each leaves a different trail.

Shelf Sentinel AI treats the problem as **reconciliation rather than suspicion**. It never asks whether a person looks like a thief. It asks whether the items add up: was everything that left the shelf basketed, returned, scanned or still in hand; did everything that reached the bagging area pass a scan; does the barcode match the product; is there a payment behind the trolley at the door. It works on anonymous signals only (skeleton keypoints, item and bag detections, shelf events and point of sale records), so no faces, no video and no demographic attributes are stored or modelled. A detector turns those signals into one of seven named scenarios with a risk score, a precision first policy decides whether the clip is worth a colleague's minute, and a language model writes a short, neutral incident brief from a closed evidence packet, always listing the innocent explanations a reviewer must rule out. A human makes every decision.

This repository is the complete, runnable system behind that idea: a simulator that produces a large labelled corpus of perception streams, the feature extractor, a rule baseline and a learned detector, the alert policy, the evidence and narration layer with guardrails, a renderer for picture examples, and an evaluation that reports results the way an operations director would need them, at a realistic theft rate, in alerts per store per day and pounds per year. The corpus is synthetic, and that is stated wherever a number appears. The engineering, the evaluation design and the insights are the point; the section on real footage explains exactly what is needed to move from simulation to a store.

## Results at a glance

All figures are measured on 46,673 clips from seven stores the model never saw in training, and reweighted so that theft makes up one clip in 200, which is the deployment assumption. Theft is far more common in the training corpus than in a real store, and reporting precision at the training mix would flatter the system.

| Measure | Hand written rules | Shelf Sentinel, review tier | Shelf Sentinel, priority tier |
| :-- | --: | --: | --: |
| Precision: alerts that are real theft | 15% | **91%** | **93%** |
| Recall: theft clips that raise an alert | 77% | **91%** | 89% |
| False alerts per 10,000 honest clips | 213.7 | **4.8** | 3.5 |
| Alerts per store per day (2,000 clips a day) | 50.2 | **10.1** | 9.6 |
| Share of value at risk that is flagged | 79% | **95%** | 94% |
| Illustrative net value per store per year | £61,000 | **£88,000** | £87,000 |

Three sentences summarise the table. The learned detector catches more theft than the rules while sending a reviewer about one fifth as many alerts. Nine of every ten review alerts are real, against roughly one in seven for the rules. Honest shoppers are flagged about 45 times less often, which is the number that protects both customers and colleagues.

Across all eight classes the detector reaches a macro F1 of 0.971 and an accuracy of 99.1 percent, with an area under the ROC curve of 0.999 for theft against normal behaviour. Those are results on simulated data and should be read as evidence that the pipeline is sound, not as a forecast of accuracy in a real store.

![Shelf Sentinel against rule and linear baselines](docs/images/model_comparison.png)

## The seven scenarios, with picture examples

Each storyboard below shows what the system sees and nothing more: an anonymous skeleton, item, bag and basket detections, zone outlines and point of sale events on a stylised scene. The four frames are chosen automatically from the detected timeline, and the chart underneath is the risk score from streaming replay, so you can see the moment the evidence arrives. The pictures are rendered from simulated clips. They are not photographs and no real person appears in them.

| Scenario | Camera | What gives it away | The honest lookalike it must not flag |
| :-- | :-- | :-- | :-- |
| Concealment in a bag | Aisle | An item leaves the shelf, then leaves the hand at a personal bag, with no basket, shelf or scan event to explain it | Scan and go shoppers packing their own bag after a handset scan |
| Concealment in clothing | Aisle | An item leaves the hand at the waistband and never reappears | A shopper with no basket holding items against the body; a phone going back in a pocket |
| Shelf sweep | Aisle | Many removals in seconds, most unaccounted for | Colleagues restocking; bulk buyers loading a trolley |
| Self checkout skip scan | Self checkout | More items bagged, or gone from the basket, than scanned | Own bag on the scale; quantity key; a barcode that will not read |
| Ticket switch | Self checkout | The scan happens, but the product seen does not match the barcode and the price is far too low | Loose produce; reduced to clear stickers |
| Sweethearting at the till | Staffed till | A cashier passes several items to the packing side without scans | Bulky goods scanned by handset; quantity key |
| Trolley push out | Store exit | Visible goods at the doors, no linked payment, almost no time at a checkout | Paid sales that failed to link; click and collect; colleagues moving trolleys |

### Concealment in a bag

The score stays near zero while the item is taken and inspected, and jumps when the item is last seen at the bag at 6.0 seconds with nothing to account for it. A second concealment at 10.8 seconds confirms it.

![Storyboard: concealment in a bag](docs/images/scenarios/01_concealment_bag.png)

### Concealment in clothing

![Storyboard: concealment in clothing](docs/images/scenarios/02_concealment_clothing.png)

### Shelf sweep

![Storyboard: shelf sweep](docs/images/scenarios/03_shelf_sweep.png)

### Self checkout skip scan

Nothing about the pose is unusual here. The alert comes entirely from reconciliation: the bagging count keeps rising with no scan in the two seconds before.

![Storyboard: self checkout skip scan](docs/images/scenarios/04_skip_scan.png)

![Animated replay: self checkout skip scan](docs/images/scenarios/04_skip_scan.gif)

### Ticket switch

![Storyboard: ticket switch](docs/images/scenarios/05_ticket_switch.png)

### Sweethearting at the till

![Storyboard: sweethearting at the till](docs/images/scenarios/06_sweethearting.png)

### Trolley push out

In this clip the alert fires two seconds after the person appears on the exit camera and more than six seconds before they reach the doors, because the journey context (no linked payment, seconds rather than minutes at a checkout, a full trolley) is already known.

![Storyboard: trolley push out](docs/images/scenarios/07_push_out.png)

### Honest behaviour that looks similar, correctly left alone

A shopper using scan and go puts items straight into a personal bag, exactly like concealment, except that a handset scan follows each pick. The detector stays clear.

![Storyboard: scan and go, no alert](docs/images/scenarios/08_benign_scan_and_go.png)

Two more hard negatives are in the gallery: a shopper [carrying items in hand with no basket](docs/images/scenarios/09_benign_carry_in_hand.png) and a shopper who [puts their own bag on the bagging area](docs/images/scenarios/10_benign_own_bag.png). Animated replays of all seven theft scenarios are in [docs/images/scenarios](docs/images/scenarios).

### Where it goes wrong

The examples above are curated for clarity, so here are two the detector gets wrong. The first is a concealment on a low light camera in a crowded store. Half the skeleton drops out, the personal bag is never detected, and the release of the item cannot be tied to anything, so it reads like the detector flicker that honest clips are full of and the score stays low. The second is an honest customer leaving with a click and collect order: goods in view, no linked payment and no checkout time is precisely the push out signature.

![Storyboard: missed concealment in low light](docs/images/scenarios/11_missed_concealment.png)

![Storyboard: false alert on a click and collect customer](docs/images/scenarios/12_false_alert_collection.png)

## How it works

![Architecture: from anonymous signals to a reviewed alert](docs/images/architecture.png)

1. **Perception** runs at the store edge and emits anonymous streams: 17 pose keypoints per tracked person, item in hand and bag detections, basket and bagging counts, shelf sensor events, and the till and security gate feeds. In this repository perception is simulated, with a sensor model that adds keypoint jitter, dropped joints, missed item detections, counter drift, clock skew and occlusion by other shoppers.
2. **A shared clip schema** holds one tracked person for 12.8 seconds at 5 frames per second: keypoints of shape `(64, 17, 3)`, signals of shape `(64, 15)` and a little journey context. The simulator, the adapter for real pose data, the feature extractor and the renderer all speak this one format.
3. **Sixty two interpretable features** are computed per clip, in four families: pose, item flow, point of sale and context. Every feature has a plain language label in the [feature dictionary](docs/feature_dictionary.md).
4. **The detector** is a gradient boosted tree classifier over eight classes, trained on complete clips and on partial clips so that it can be scored while a clip is still unfolding. A hand written rule list is kept beside it as a baseline and a fallback. Calibration is tested rather than assumed.
5. **The alert policy** chooses, for every scenario, the lowest threshold that meets a precision target at the deployment theft rate: 80 percent for the review tier and 95 percent for the priority tier.
6. **An evidence packet** is built for each alert: a timeline of detected events, the features that moved the score and how they compare with normal clips on that camera, and data quality flags.
7. **A language model narrator** turns the packet into a five part brief. The reply is parsed, checked against a schema and passed through a language guard. If anything fails, a deterministic template brief is used instead.
8. **A human reviewer** reads the brief, looks at the storyboard, rules out the innocent explanations and decides what to do. The [scenario playbook](docs/scenario_playbook.md) keeps the response proportionate and service led.

## The data

The corpus holds **250,000 single person track clips: 16 million frames, or 889 hours of tracked footage, across 40 simulated stores** in three formats (convenience, supermarket and superstore), each with its own camera quality mix, crowding and theft profile. It contains 32 scripted behaviours: 7 theft scenarios and 25 benign behaviours, many of them deliberate lookalikes. A further 50,000 clips are cut at random points into 97,391 labelled partial clips for streaming aware training.

![Corpus composition by behaviour and camera](docs/images/dataset_composition.png)

Design choices that matter:

* **Hard negatives are first class.** A loss prevention model earns its keep on the honest behaviours that resemble theft, so scan and go, own bag on the scale, quantity key, reductions, void, handset scanning, restocking, bulk buying, phone checks, carrying without a basket, click and collect and failed payment linkage are all simulated on purpose.
* **Theft is enriched, then reweighted.** Theft is 13.7 percent of the corpus so that every scenario has more than four thousand examples. Every headline metric is reweighted to a deployment rate of 0.5 percent.
* **Stores are split, never clips.** 26 stores train the model, 7 are used for calibration and thresholds, and 7 are held out for testing, stratified by format. The test therefore measures transfer to unseen stores, cameras and theft mixes.
* **Noise is part of the label.** Low light cameras, crowding and occlusion degrade the signals, and a few benign clips are built to be truly ambiguous, so the ceiling is below 100 percent by design.
* **It is reproducible bit for bit.** The whole corpus is generated from one seed in about eight minutes on a single core, and the feature shards (60 parquet files, 33 MB) are committed, so training runs straight after cloning.

The corpus is simulated because real, labelled, multi sensor footage of shoplifting cannot be shared. The [data card](docs/data_card.md) documents the schema, the behaviours, the sensor model and the known gaps between the simulation and a store.

## Findings

### 1. The till is the best camera in the building

Training the same model on growing sets of sensors shows where the information lives. With pose alone the detector reaches a macro F1 of 0.58 and is blind to ticket switching and sweethearting (recall near zero), because those acts look exactly like honest scanning. Adding item flow lifts it to 0.76. Adding point of sale reconciliation lifts it to 0.97. The practical reading for a retailer is that integrating till and scale data with the video system is worth more than a better camera or a bigger vision model.

![Sensor ablation: recall by scenario as sensor families are added](docs/images/sensor_ablation.png)

### 2. A learned detector beats rules on the scenarios that matter most to customers

The rule list is respectable on recall (77 percent) but only 15 percent of its alerts are real. Its weakest rules are the ones closest to honest behaviour: 6 percent precision on skip scan and 8 percent on clothing concealment, because own bags, quantity keys and items held against the body trip the same conditions. The learned detector weighs those lookalike signals together and holds 84 and 79 percent precision on the same two scenarios. Its recall is higher than the rules on every scenario except skip scan, where it gives up five points of recall for roughly a fifteenfold gain in precision. A logistic regression given the same alert budget reaches 72 percent precision and 73 percent recall, so the interactions between features, not just the features, carry real value.

![Recall and precision by scenario](docs/images/per_scenario_performance.png)

### 3. The base rate decides whether the product is usable

The same detector with the same thresholds delivers very different precision depending on how common theft is. At one theft clip in 1,000 precision is 66 percent; at one in 200 it is 91 percent; at one in 50 it is 97.5 percent. A pilot in a high theft store will look better than the rollout to a quiet one, and thresholds have to be set per store group, not once.

![Alert precision against theft prevalence](docs/images/precision_vs_prevalence.png)

### 4. More alerts stop paying for themselves quickly

With illustrative costs (45 pence of analyst time per alert, 30 percent of flagged value recovered or deterred), net value peaks at about 13 alerts per store per day and declines after that, because each extra alert is increasingly likely to be an honest shopper. The review tier sits close to that peak at 10 alerts a day and flags 95 percent of the value at risk. The rules send 50 alerts a day for less value. The average simulated incident is worth about £86, the same order of magnitude as the roughly £70 per detected incident implied by the British Retail Consortium figures.

![Net value against reviewer workload](docs/images/economics_curve.png)

### 5. The remaining false alerts have a name

Only 19 of 39,773 honest test clips raise a review alert, and they are not random. Click and collect customers are flagged at 20 per 1,000, more than ten times the rate of any other behaviour: a quarter of all false alerts come from a behaviour that makes up less than one percent of honest clips. That points to a specific integration fix, linking collection orders to the exit track, which would do more for customers than any amount of model tuning.

![False alert rate by benign behaviour](docs/images/false_alert_sources.png)

### 6. Low light costs seven points of recall

Recall falls from 94 percent on HD cameras to 86.5 percent in low light, and the false alert rate rises about fivefold, from 0.2 to 1.1 per 1,000 honest clips. Crowding costs less (93 percent in a quiet store, 90 percent in a crowded one) and store format barely matters. The missed concealment shown earlier is this effect in one picture. For a rollout, camera quality is the audit to run first.

![Robustness by camera quality, crowding and store format](docs/images/robustness_slices.png)

### 7. Scoring a clip live is a different problem from scoring it afterwards

The first version of the detector was trained only on complete clips, and it looked excellent until it was replayed frame by frame. Scored live, it raised a transient alert on 43 of every 1,000 honest clips, about 150 times the rate it showed on complete clips (0.3 per 1,000). The cause is simple: a shopper who has picked an item up and not yet put it in the basket looks, for a second or two, exactly like an item unaccounted for. Two changes fixed most of it. Training on partial clips, labelled by what has happened so far, cut live false alerts to 8.5 per 1,000 with no loss of accuracy on complete clips. Requiring an alert to hold for two evaluations in a row cut it again to 4.7. Live recall is 93 percent.

That is still about sixteen times the false alert rate of scoring a completed clip, so the honest recommendation is to use live alerts only where seconds matter, at the exit and the tills, and to score aisle clips when the window closes.

![False alerts when scoring live against scoring complete clips](docs/images/streaming_false_alerts.png)

### 8. Where live alerts are used, they arrive in time to act

With the confirmation rule in place, concealment alerts fire a median of about one second after the item disappears and skip scan alerts 1.4 seconds after the unscanned item is bagged. Ticket switch and sweethearting alerts fire at about the moment the item reaches the packing side. Push out alerts fire a median of six seconds before the person reaches the doors. Between 83 and 100 percent of theft clips are caught inside the clip.

![Detection latency by scenario](docs/images/detection_latency.png)

### 9. Three honest caveats from the evaluation

* **Thresholds set on rare events did not transfer, and the policy was changed because of it.** In the first version every scenario met its 80 percent precision target on the validation stores, but on the test stores sweethearting came in at 59 percent and clothing concealment at 77 percent. At a theft rate of 0.5 percent one honest clip carries the weight of about 32 theft clips, so each threshold was resting on a handful of validation errors. The policy now requires a review threshold to meet its target even if one more honest validation clip had been flagged. With that margin sweethearting is at 96 percent, clothing concealment at 79 percent (still just under target) and the tier as a whole at 91 percent, for a cost of one point of recall. The margin was added after seeing the gap on the test stores, so its test figures are slightly optimistic and should be confirmed on fresh stores.
* **The priority tier misses its target.** It was set for 95 percent precision and delivers 93 percent on the test stores. The same arithmetic explains why: 95 percent with a safety margin cannot be certified on a validation set of this size, so the margin is applied to the review tier only. More validation data per scenario is the fix.
* **Calibration was tested and the extra step was rejected.** Isotonic calibration made validation calibration error worse (0.0070 against 0.0051), so the pipeline kept the raw probabilities. Test calibration error is 0.006, which means the risk score can be read as a probability.

The confusion matrix, feature importance and calibration charts are in [docs/images](docs/images), the full tables are in the [model card](docs/model_card.md), and every number in this section comes from [reports/metrics.json](reports/metrics.json) and the tables beside it.

## The language model layer

A risk score of 0.97 is not something a duty manager can act on. The narrator turns each alert into a brief that can be read in ten seconds:

> **Priority alert: possible concealment in a bag on the aisle camera.**
>
> **What was observed**
> * At 5.2 s: An item was taken from the shelf.
> * At 6.0 s: Item in the right hand was last seen at a personal bag; no basket, shelf or scan event explains it.
> * At 10.8 s: Item in the right hand was last seen at a personal bag; no basket, shelf or scan event explains it.
> * Unexplained disappearances was 2; the typical value for normal clips on this camera is 0 (range 0 to 2).
> * Fastest gap between picks was 3.8; the typical value for normal clips on this camera is 12.8 (range 1 to 12.8).
>
> **Innocent explanations to rule out**
> * The shopper is using scan and go and the handset scan was not recorded.
> * The object was a personal item such as a phone, list or purse.
> * The bag is the shopper's own shopping bag and they intend to pay at the till.
>
> **Recommended action** Send a colleague to offer a basket and assistance now. Check whether the item is presented at checkout before any further step.
>
> **Confidence** Model risk is 1.00, which is strong evidence for this scenario. Pose tracking was clear throughout the clip.

How the layer is built:

* **Closed evidence.** The model receives a JSON evidence packet and, optionally, four skeleton key frames. It never sees video or a face, and the system prompt forbids any fact that is not in the packet.
* **Behaviour, never people.** The prompt bans descriptions of appearance, age, gender, ethnicity or clothing, bans words such as thief, stole and guilty, and requires innocent explanations and a proportionate, service led action.
* **Checked output.** The reply must parse as JSON with five required fields and must pass a language guard that rejects accusatory or demographic terms. Any failure, including an API error, falls back to a deterministic template brief, and the reason is recorded.
* **Two narrators, one interface.** `TemplateNarrator` runs offline and is the default. `AnthropicNarrator` calls a Claude model through the Anthropic Messages API. To switch it on, set `narrator.provider` to `anthropic` in the config and export `ANTHROPIC_API_KEY`; see [examples/claude_narrator.py](examples/claude_narrator.py).

The briefs committed under [reports/incidents](reports/incidents) were written by the template narrator, because the build environment had no API key. The Claude path has not been run against the live API here. It is exercised in the test suite with a stub client, including the cases where the model returns accusatory language, a demographic description, a missing field or no JSON at all.

## Responsible use

A system like this can do harm if it is deployed carelessly, so the safeguards are part of the design and are documented in [docs/responsible_use.md](docs/responsible_use.md).

* It models what happens to items, not who a person is. There are no faces, no identities, no demographic inputs and no memory of individuals between clips.
* It produces review requests, not accusations. Every alert lists innocent explanations, and the playbook's first response is almost always an offer of help.
* It is evaluated for honest shoppers first: precision targets, false alerts per honest clip, and named sources of false alerts are headline metrics.
* It should not be used to detain, search or ban anyone, to score staff performance, or without a data protection impact assessment and clear signage. Staff scenarios such as sweethearting go to a manager for quiet review, never to the shop floor.

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/shelf_sentinel_ai.git
cd shelf_sentinel_ai
make install                    # editable install with test and lint tools
make test                       # 53 tests in under a minute
python examples/quickstart.py   # score the sample clips and print a brief
```

The trained model, the feature shards and a sample of 64 raw clips are committed, so the example above works immediately. To rebuild everything from the seed:

```bash
make all      # corpus, training, evaluation, charts and gallery: about fifteen minutes on one core
make quick    # the same pipeline on 20,000 clips for a first look
```

Using the library directly:

```python
from shelfsentinel.explain import TemplateNarrator, build_evidence
from shelfsentinel.features import extract_features
from shelfsentinel.models import SentinelModel
from shelfsentinel.schema import ClipBatch

clips = ClipBatch.load("data/sample/clips_sample.npz")
model = SentinelModel.load("models/sentinel.joblib")

assessment = model.assess(extract_features(clips))   # scenario, risk and tier per clip
evidence = build_evidence(model, clips, index=9)     # the facts behind one alert
print(TemplateNarrator().narrate(evidence).to_markdown())
```

## Repository map

```text
src/shelfsentinel
    schema.py            clip schema, scenarios, scenes, signal channels
    scenes.py            camera zone geometry
    config.py            typed configuration loaded from YAML
    simulation           behaviour scripts, kinematics, sensor model, corpus generator
    features             feature extractor and plain language catalog
    models               rule baseline, gradient boosted detector, alert policy
    evaluation           deployment weighted metrics, economics, streaming replay
    explain              evidence packets, playbook, prompts, narrators and guard
    viz                  frame renderer, storyboards, gallery, charts
    adapters             bridge from real pose trackers to the clip schema
    pipeline.py          training and evaluation from end to end
    cli.py               command line entry point
configs                  default (250,000 clips) and quick (20,000 clips)
data/generated           the committed corpus: complete clip and partial clip feature shards
data/sample              64 raw clips, two per behaviour
models                   the trained detector with its alert policy
reports                  metrics.json, result tables and incident briefs
docs                     data card, model card, playbook, feature dictionary, images
tests                    53 unit and integration tests
scripts, examples        asset builder and runnable examples
```

## From simulation to a store

The detector does not care where its inputs come from, only that they arrive in the clip schema. The [real footage guide](docs/real_footage.md) sets out the path:

* **Pose** from any COCO 17 keypoint tracker. `shelfsentinel.adapters` converts per frame detections into clips, resamples to the working frame rate and marks tracking gaps as unseen. PoseLift, a public dataset of anonymised pose sequences from a real store, is the natural first test for the pose features. That test has not been run here.
* **Item flow** from a product and bag detector plus hand association, and from shelf weight or vision sensors where they exist.
* **Point of sale** from the till and self checkout event feeds, joined on time and lane.
* **Labels** from reviewed alerts, which is why the reviewer's decision is logged.

Without item flow and point of sale channels the system degrades to the pose only row of the ablation, which is a useful way to set expectations for a camera only pilot.

## Limitations and next steps

* The corpus is synthetic. The behaviours are scripted from loss prevention practice, not learned from footage, and real perception errors are messier than the sensor model. Real world accuracy is unknown until the system is tested on real data.
* Clips follow one tracked person. Distraction teams, handoffs between people and accomplices outside the cashier scenario are not modelled.
* The economics use illustrative costs and a fixed recovery rate. They show the shape of the trade off, not a business case.
* Live scoring still raises far more false alerts than scoring complete clips, and two tiers sit under their precision targets on some scenarios.
* Next steps: validate the pose features on PoseLift; add a sequence model over raw channels beside the feature model; learn thresholds per store group with uncertainty bounds; add an active learning loop from reviewer decisions; run the Claude narrator against a red team set of adversarial evidence packets.

## Further reading in this repository

* [Architecture](docs/architecture.md)
* [Data card](docs/data_card.md)
* [Model card](docs/model_card.md)
* [Scenario playbook](docs/scenario_playbook.md)
* [Feature dictionary](docs/feature_dictionary.md)
* [Responsible use](docs/responsible_use.md)
* [Real footage guide](docs/real_footage.md)

## Sources

* British Retail Consortium, [Crime Report 2026](https://brc.org.uk/news-and-events/news/corporate-affairs/2026/ungated/hard-won-progress-on-crime-but-job-far-from-done/): detected shop theft incidents and cost, violence and abuse against retail workers, and the Usdaw finding on attacks triggered by theft.
* Rashvand and colleagues, [Exploring Pose Based Anomaly Detection for Retail Security: A Real World Shoplifting Dataset and Benchmark](https://arxiv.org/pdf/2501.06591), WACV Workshops 2025: the PoseLift dataset.

Released under the MIT licence.
