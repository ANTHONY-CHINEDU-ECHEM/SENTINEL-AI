"""Plain language dictionary for every feature.

The same text is used in three places: the generated feature dictionary in the
docs, the evidence packet handed to the narrator, and the review screen. One
source of truth keeps the model, the paperwork and the explanations aligned.
"""

from __future__ import annotations

from .extractor import FEATURE_GROUPS, FEATURE_NAMES

FEATURE_LABELS: dict[str, tuple[str, str]] = {
    # pose
    "speed_mean": ("Average walking speed", "Body heights per second, averaged over the clip"),
    "speed_max": ("Peak walking speed", "Fastest sustained movement in the clip, in body heights per second"),
    "still_frac": ("Time standing still", "Share of the clip with almost no movement"),
    "path_length": ("Distance covered", "Straight line distance between first and last position, in body heights"),
    "width_mean": ("Average torso width", "Apparent shoulder width; low values mean side on to the camera"),
    "width_min": ("Narrowest torso width", "Tenth percentile of apparent shoulder width"),
    "width_range": ("Turning", "Change in apparent shoulder width, a sign of turning away from the camera"),
    "yaw_abs_p90": ("Head turn size", "Ninetieth percentile of sideways head offset"),
    "glance_count": ("Head turns", "Number of distinct sideways head turns"),
    "face_visible_frac": ("Face towards camera", "Share of frames where the face keypoints are confidently seen"),
    "reach_count": ("Reaches", "Number of times an arm extended towards the shelf"),
    "reach_frac": ("Time reaching", "Share of the clip spent with an arm extended"),
    "near_waist_frac": ("Hand at waist", "Largest share of the clip either hand spent at the waistband"),
    "near_bag_frac": ("Hand at personal bag", "Largest share of the clip either hand spent at a personal bag"),
    "hands_together_frac": ("Hands together", "Share of the clip with both wrists close together"),
    "kp_conf_mean": ("Pose confidence", "Mean confidence of the upper body keypoints"),
    "arm_occluded_frac": ("Arm hidden", "Share of frames where at least one wrist was not seen"),
    # item flow
    "shelf_picks": ("Items taken from shelf", "Shelf sensor removal events"),
    "shelf_returns": ("Items put back on shelf", "Shelf sensor return events"),
    "min_pick_interval_s": ("Fastest gap between picks", "Shortest time between two removals, in seconds"),
    "item_frames_frac": ("Share of clip holding an item", "Share of the clip with merchandise seen in a hand"),
    "item_onsets": ("Times an item appeared in hand", "Count of item in hand episodes"),
    "mean_hold_s": ("Average hold time in seconds", "Mean duration of an item in hand episode, in seconds"),
    "hand_item_end": ("Item still in hand at end", "One when merchandise is visible in a hand as the clip ends"),
    "vanish_total": ("Items leaving the hand", "Times a held item stopped being visible and stayed gone"),
    "vanish_unexplained": ("Unexplained disappearances", "Items that left the hand with no basket, shelf, bagging or scan event to explain it"),
    "vanish_near_bag": ("Items released at personal bag", "Items that left the hand while the hand was at a personal bag"),
    "vanish_near_waist": ("Items released at waist", "Items that left the hand while the hand was at the waistband"),
    "vanish_unexplained_near_bag": ("Unexplained release at bag", "Item vanished at a personal bag with nothing to explain it"),
    "vanish_unexplained_near_waist": ("Unexplained release at waist", "Item vanished at the waistband with nothing to explain it"),
    "container_start": ("Basket count at start", "Visible items in the basket, trolley or belt when the clip begins"),
    "container_gain": ("Basket count change", "Change in visible basket or trolley items over the clip"),
    "container_max": ("Largest basket count", "Highest visible item count in the basket or trolley"),
    "bagging_gain": ("Bagging area change", "Change in the item count on the bagging or packing area"),
    "unaccounted_items": ("Unaccounted items", "Items taken minus items returned, basketed, scanned or still in hand"),
    "transfers": ("Items carried to bagging", "Item in hand episodes that ended at the bagging area"),
    "bypass_transfers": ("Items that bypassed the scanner", "Transfers to bagging that never entered the scanner zone"),
    "quick_pass_transfers": ("Items rushed past the scanner", "Items lifted from the input side that spent at most one frame over the scanner"),
    "scan_dwell_mean_s": ("Scanner dwell per item", "Seconds an item spent over the scanner per transfer"),
    "bag_present": ("Personal bag visible", "Share of frames with a personal bag detected"),
    "container_present": ("Basket or trolley visible", "Share of frames with a basket or trolley detected"),
    # pos
    "pos_total": ("Quantity scanned", "Net quantity registered by the point of sale"),
    "pos_events": ("Scan events", "Number of scan events"),
    "pos_voids": ("Voids", "Number of void events"),
    "unscanned_bagging": ("Bagged but not scanned", "Bagging area gain minus quantity scanned"),
    "flow_gap": ("Left the basket but not scanned", "Items that left the input side minus quantity scanned"),
    "scans_per_transfer": ("Scans per transfer", "Scan events divided by items carried to bagging"),
    "scan_match_min": ("Weakest product match", "Lowest similarity between the product seen and the barcode scanned"),
    "scan_match_mean": ("Average product match", "Mean similarity between product seen and barcode scanned"),
    "price_ratio_min": ("Lowest price ratio", "Scanned price divided by the price expected for the product seen"),
    "suspect_scans": ("Mismatched cheap scans", "Scans where the product match was weak and the price was far below expectation"),
    "low_match_scans": ("Weak match scans", "Scans with product similarity under one half"),
    "low_price_scans": ("Cheap scans", "Scans priced under sixty percent of expectation"),
    "scan_before_bag": ("Scanned before bagging", "Items placed in a personal bag after a handset scan"),
    "txn_linked": ("Paid transaction linked", "One when a completed sale is linked to this person"),
    "checkout_dwell_s": ("Seconds spent at checkout", "Seconds the multi camera tracker saw this person in a checkout zone"),
    "eas_any": ("Security tag alarm", "One when the exit gate alarm sounded"),
    # context
    "scene_aisle": ("Aisle camera", "Camera context flag"),
    "scene_self_checkout": ("Self checkout camera", "Camera context flag"),
    "scene_staffed_till": ("Staffed till camera", "Camera context flag"),
    "scene_exit": ("Exit camera", "Camera context flag"),
    "reached_exit": ("Reached the doors", "One when the person crossed into the exit zone"),
}

missing = [n for n in FEATURE_NAMES if n not in FEATURE_LABELS]
if missing:  # pragma: no cover - guards against drift between extractor and catalog
    raise RuntimeError(f"features without a catalog entry: {missing}")


def feature_dictionary_markdown() -> str:
    """Render the catalog as a markdown document."""
    titles = {"pose": "Pose", "item_flow": "Item flow", "pos": "Point of sale and journey", "context": "Context"}
    lines = ["# Feature dictionary", "", f"The detector reads {len(FEATURE_NAMES)} interpretable features per clip.", ""]
    for group, names in FEATURE_GROUPS.items():
        lines += [f"## {titles[group]}", "", "| Feature | Label | Meaning |", "| :-- | :-- | :-- |"]
        lines += [f"| `{n}` | {FEATURE_LABELS[n][0]} | {FEATURE_LABELS[n][1]} |" for n in names]
        lines.append("")
    return "\n".join(lines)
