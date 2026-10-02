# Data card: the Shelf Sentinel synthetic corpus

## Summary

| Property | Value |
| :-- | :-- |
| Unit of data | One clip: a single tracked person for 12.8 seconds at 5 frames per second |
| Size | 250,000 clips, 16,000,000 frames, 889 hours of tracked footage |
| Stores | 40 simulated stores in three formats: convenience, supermarket, superstore |
| Camera contexts | Aisle (149,612 clips), self checkout (42,684), staffed till (25,840), store exit (31,864) |
| Behaviours | 32 scripted behaviours: 7 theft scenarios and 25 benign behaviours |
| Labels | Scenario (8 classes), behaviour subtype, variant, time of the theft act, value at risk |
| Theft share | 13.7 percent of clips (enriched for learning; evaluation reweights to 0.5 percent) |
| Camera quality | HD 42 percent, SD 40 percent, low light 18 percent |
| Splits | By store: 26 train (162,803 clips), 7 validation (40,524), 7 test (46,673), stratified by format |
| Partial clips | A further 50,000 clips cut at random lengths (97,391 labelled rows) for streaming aware training |
| Storage | 50 parquet shards of clip features (25 MB) plus 10 shards of partial clip features (8 MB); 64 raw sample clips |
| Generation | Deterministic from one seed; about eight minutes on one core |
| Personal data | None. Every value is simulated |

## Why synthetic

Real labelled footage of shoplifting that also carries shelf, scale and till data is not publicly available, and it could not be published if it were. Simulation gives exact labels for seven distinct scenarios, full control over the honest lookalikes, and a corpus that anyone can regenerate. The cost is realism, and the limits are listed at the end of this card.

## Scenario counts

| Scenario | Camera | Clips | What happens |
| :-- | :-- | --: | :-- |
| Normal shopping | All | 215,703 | 25 benign behaviours |
| Concealment in a bag | Aisle | 5,427 | Item taken, then placed in a personal bag |
| Concealment in clothing | Aisle | 5,345 | Item taken, then tucked at the waistband |
| Shelf sweep | Aisle | 4,415 | Five to ten items cleared in seconds, usually into a bag |
| Self checkout skip scan | Self checkout | 5,273 | One to three items bagged without a scan: covered barcode, lifted over the scanner, two items stacked, or straight into a shoulder bag |
| Ticket switch | Self checkout | 4,219 | One or two items scanned under a cheaper barcode |
| Sweethearting at the till | Staffed till | 4,144 | A cashier passes two to four items without scanning |
| Trolley push out | Store exit | 5,474 | Leaves with goods and no payment; four in five with a trolley, one in five carrying items |

The 25 benign behaviours are listed with their purpose in the [scenario playbook](scenario_playbook.md).

## How a clip is made

1. **Script.** A behaviour function writes a sequence of actions for one person: where the body moves, where each hand goes, when an item is picked up or released, what the till records.
2. **Kinematics.** The script is resampled to 64 frames and turned into 17 keypoints with two link arm kinematics and a simple gait.
3. **Sensor model.** Ground truth is degraded according to the camera quality and crowding of the store:

| Effect | HD | SD | Low light |
| :-- | --: | --: | --: |
| Keypoint jitter (share of image width) | 0.4% | 0.8% | 1.3% |
| Keypoint dropout per joint per frame | 2% | 5% | 10% |
| Missed item in hand per frame | 5% | 10% | 18% |
| False item in hand per frame | 0.8% | 1.5% | 3% |
| Missed shelf event | 4% | 7% | 12% |
| Clip with a drifting item counter | 5% | 8% | 12% |
| Clip where the personal bag is not detected | 5% | 8% | 14% |

In addition: a hand held in front of the torso is hidden from a camera behind the shopper (30 percent extra miss rate); crowded stores hide one arm for one to three seconds in up to 45 percent of clips; item counters saturate in a heaped trolley; the till clock is skewed by up to 0.4 seconds; payment linkage fails for 6 percent of paying customers.

4. **Features.** Sixty two features are extracted and written to parquet with the labels and context.

## Fields in the feature shards

| Column | Meaning |
| :-- | :-- |
| `clip_id` | Shard and clip number |
| `store_id`, `store_format` | Simulated store and its format |
| `scene` | Camera context: 0 aisle, 1 self checkout, 2 staffed till, 3 exit |
| `scenario` | Label: 0 normal, 1 to 7 theft scenarios |
| `subtype`, `variant` | Behaviour name and, where relevant, its variant |
| `camera_quality`, `crowding`, `hour` | Capture conditions (used for slicing, never as model inputs) |
| `body_scale` | Apparent height of the person in the image |
| `event_time_s` | Time of the theft act in seconds, empty for normal clips |
| `value_at_risk_gbp` | Simulated value of the goods, zero for normal clips |
| 62 feature columns | See the [feature dictionary](feature_dictionary.md) |

The partial clip shards add `prefix_frames` (how much of the clip was seen) and `scenario_full_clip` (the label of the complete clip). Their `scenario` column is the label of the partial clip: a clip that will contain a concealment is labelled normal until the concealment has happened, and rows cut within about a second of the act are dropped.

## Intended use

* Developing and testing the Shelf Sentinel pipeline end to end.
* Studying evaluation at low prevalence, alert budgets and hard negatives.
* Teaching: the simulator is small enough to read in an afternoon.

## Out of scope

* Estimating real world accuracy.
* Training a model for use on real customers without real data validation.
* Any conclusion about real shoppers. The behaviours are scripts.

## Known limitations

* **Scripted behaviour.** Variation comes from random timings, positions and choices inside 32 scripts. Real behaviour has a longer tail.
* **Simplified perception.** Noise is independent across frames apart from occlusion bursts and counter drift. Real errors are more structured.
* **One person per clip.** Groups, handoffs and identity switches are absent.
* **Label noise is light.** About 1 percent of skip scan clips and 2 percent of sweethearting clips have the theft act at or beyond the edge of the window. They keep their theft label and act as a small, realistic ceiling on recall.
* **No demographic attributes.** This is deliberate, and it means fairness across groups of people cannot be assessed with this corpus.
* **Assumed prevalence.** The deployment theft rate of 0.5 percent is an assumption. The precision against prevalence chart shows how results move if it is wrong.
