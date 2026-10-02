# From simulation to a store

The detector consumes a schema, not a simulator. This guide lists what a real deployment has to supply for each part of the schema, what can be tested today with public data, and a staged plan for a pilot.

## What each channel needs

| Schema element | Real world source | Notes |
| :-- | :-- | :-- |
| `keypoints` | Any COCO 17 keypoint estimator with tracking | Convert with `shelfsentinel.adapters.tracks_from_frames`, `resample_track` and `clips_from_track` |
| `hand_item_l`, `hand_item_r` | Product detector plus hand association | The largest perception gap: small, occluded objects are hard. Expect miss rates above the simulated ones |
| `shelf_delta` | Shelf weight sensors, or shelf vision | Optional; without it unaccounted items cannot be computed in the aisle |
| `container_count`, `bagging_count` | Basket, trolley and bagging area item counters; self checkout scale | The scale is usually the most reliable of these |
| `pos_qty`, `price_ratio` | Till and self checkout event feed | Join on lane and time; allow for clock skew |
| `scan_match` | Product recognition at the scanner compared with the scanned barcode | Available from several self checkout vendors |
| `bag_*`, `cont_*` | Object detector classes for bags, baskets and trolleys | |
| `txn_linked`, `checkout_dwell_s` | Multi camera tracking joined to the sales feed | Linkage failures are a named source of false alerts |
| `eas_alarm` | Security tag gate | |
| Zones | One time camera calibration | Replace the fixed rectangles in `scenes.py` with per camera polygons |

## A minimal example with a pose tracker

```python
from shelfsentinel.adapters import clips_from_track, resample_track, tracks_from_frames
from shelfsentinel.features import extract_features
from shelfsentinel.schema import Scene

# frames: [{"frame": 0, "persons": [{"track_id": 7, "keypoints": [[x, y, conf], ...]}]}, ...]
tracks = tracks_from_frames(frames, image_size=(1920, 1080))
for track_id, (frame_numbers, keypoints) in tracks.items():
    track = resample_track(frame_numbers, keypoints, fps_in=25.0)
    clips = clips_from_track(track, Scene.AISLE, track_id=track_id)
    features = extract_features(clips)   # pose features are live; item and till features are zero
```

With pose only, the system behaves like the pose only row of the sensor ablation: useful for concealment and push out, blind to ticket switching and sweethearting.

## What can be validated with public data today

* **PoseLift** is a public dataset of anonymised pose sequences, bounding boxes and track identifiers from a real store, with shoplifting and normal behaviour. It is the right first test for the pose features and the adapter, and a check on how far simulated kinematics are from real ones.
* **Video anomaly datasets with a shoplifting class** can exercise a pose estimator in front of the adapter, but they are short, staged or sparsely labelled, and none carries till data.

No public dataset combines pose, item flow and point of sale, which is why the corpus here is simulated and why a retailer partnership is needed for real validation.

## Staged pilot plan

1. **Shadow mode, pose and till only.** Run on a few cameras with alerts hidden from staff. Compare alerts with existing loss prevention records. Measure alert volume per day and review a random sample of clear clips for misses.
2. **Label by review.** Reviewer decisions become labels. A few thousand reviewed clips per scenario are enough to recalibrate thresholds and to fine tune the model on real features.
3. **Add item flow.** Introduce the item detector on self checkout first, where the camera is close and the scale gives a second opinion.
4. **Assisted mode.** Show review tier alerts to a trained reviewer with the brief and the storyboard. Track precision, the time to review and the outcome of each intervention.
5. **Scale by store group.** Set thresholds per group of similar stores, with a margin above the precision target, and monitor weekly.

## Known gaps between the simulation and reality

* Real pose errors are correlated in time and with clothing, bags and lighting. The sensor model uses simpler noise.
* Real shoppers are more varied than 25 scripted behaviours. Children, pushchairs, mobility aids, couples sharing a trolley and staff interactions are not modelled.
* Simulated thieves follow scripts. Real ones adapt to the system.
* The simulation assumes one tracked person per clip and a correct track. Identity switches in crowds will break item accounting.
