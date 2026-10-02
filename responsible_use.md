# Responsible use

Loss prevention technology touches customers, colleagues and the law. This note sets out what Shelf Sentinel AI is designed to do, what it must not be used for, and the checks a retailer should complete before any live use. It is engineering guidance, not legal advice.

## What the system does and does not process

| It processes | It does not process |
| :-- | :-- |
| Skeleton keypoints for a tracked person within one short clip | Faces, face embeddings or any biometric identifier |
| Item, bag and basket detections and counts | Age, gender, ethnicity, clothing or any description of a person |
| Shelf sensor and security gate events | Identity, loyalty card or payment card details |
| Till events: quantity, product match, price ratio | Any memory of an individual between clips or visits |

Because nothing identifies a person, the model cannot learn that a kind of person is a risk. It can only learn that a pattern of item movement is unaccounted for.

## Safeguards built into the design

* **Alerts are review requests.** The output is "this clip is worth a look", with a scenario, a risk score and the evidence. It is never a statement that someone has committed an offence.
* **Innocent explanations are mandatory.** Every brief lists them, and the narrator schema rejects a brief that has none.
* **Language guard.** Briefs are rejected if they contain accusatory words or personal descriptors, whether written by the template or by a language model.
* **Precision first policy.** Thresholds are set to meet a precision target at the real theft rate. A scenario that cannot meet its target is switched off.
* **Service led playbook.** The first response is an offer of help. The playbook never recommends detaining, searching or following a person.
* **Staff scenarios are escalated quietly.** Sweethearting alerts go to a manager with the receipt and the till log. They are never shown on the shop floor.
* **Transparency of failure.** The gallery and the evaluation show missed thefts and false alerts, and name the honest behaviours most often flagged.

## Uses this project does not support

* Automated decisions about a person with no human review.
* Building watchlists, recognising returning individuals, or sharing data about individuals between stores.
* Scoring or ranking colleagues, or monitoring productivity.
* Any use on footage of places where people have a heightened expectation of privacy, such as fitting rooms.
* Deployment on real customers on the strength of the synthetic results in this repository.

## Before a live pilot

1. **Data protection impact assessment.** Video analytics in a shop is high risk processing under UK data protection law. Complete an assessment, identify the lawful basis, and consult the data protection officer.
2. **Signage and transparency.** Update CCTV signage and the privacy notice to describe analytics, not just recording.
3. **Data minimisation.** Run perception at the edge and discard video once keypoints and events are extracted, except for clips attached to a reviewed alert, which need a retention schedule.
4. **Fairness testing on real data.** Pose estimators and item detectors can perform differently across body types, mobility aids, clothing and lighting. Measure false alert rates across those conditions before go live. In this repository only camera quality, crowding and store format are tested, because the simulator has no demographic attributes by design.
5. **Human review standards.** Train reviewers, require a second look before any approach, and log every decision and its outcome.
6. **Colleague consultation.** Staff facing scenarios need consultation with colleagues and their representatives.
7. **Complaints and redress.** Give customers a route to challenge an intervention, and feed upheld complaints back into threshold setting.
8. **Ongoing monitoring.** Track precision, false alerts per honest clip and alert volume per store every week. Theft rates and store layouts drift, and so will the thresholds.

## Language model specific risks

| Risk | Mitigation in this project |
| :-- | :-- |
| The model invents a fact | Closed evidence packet; the prompt forbids facts outside it; key frames are skeleton overlays only |
| The model describes a person | Prompt rule; language guard; fallback to the template brief |
| The model states guilt | Prompt rule; language guard; headline pattern uses "possible" |
| The model is unavailable or slow | Any exception falls back to the template brief and records the reason |
| Personal data leaves the store | The packet contains no personal data: times, counts, zones and scores only |
| Prompt injection through data | The packet is built from numeric detections, not from free text supplied by anyone |
