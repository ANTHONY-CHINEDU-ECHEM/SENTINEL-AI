"""Scenario playbook: what each alert means and what a colleague should do.

The actions follow the customer service approach used across retail loss
prevention: the system never accuses anyone. An alert asks a trained person to
look, and the first response is almost always to offer help.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import Scenario


@dataclass(frozen=True)
class Play:
    summary: str
    benign_explanations: tuple[str, ...]
    review_action: str
    priority_action: str
    verify: tuple[str, ...]


PLAYBOOK: dict[Scenario, Play] = {
    Scenario.CONCEALMENT_BAG: Play(
        summary="An item taken from the shelf appears to have gone into a personal bag without being scanned or basketed.",
        benign_explanations=(
            "The shopper is using scan and go and the handset scan was not recorded.",
            "The object was a personal item such as a phone, list or purse.",
            "The bag is the shopper's own shopping bag and they intend to pay at the till.",
        ),
        review_action="Watch the clip. If the release into the bag is confirmed, ask a colleague to offer help in the aisle and note the track for the checkout team.",
        priority_action="Send a colleague to offer a basket and assistance now. Check whether the item is presented at checkout before any further step.",
        verify=("Was a shelf removal recorded?", "Did the item reappear in a basket or at a till?", "Is scan and go active for this visit?"),
    ),
    Scenario.CONCEALMENT_CLOTHING: Play(
        summary="An item taken from the shelf appears to have been tucked under clothing at the waist.",
        benign_explanations=(
            "The shopper is carrying the item against their body because they have no basket.",
            "The hand went to a pocket for a phone or keys and the item detector lost the product.",
            "The camera view was blocked by another shopper at the key moment.",
        ),
        review_action="Watch the clip. Look for the item in the hand after the waist movement. If it is gone, ask a colleague to offer help in the aisle.",
        priority_action="Send a colleague to greet the shopper and offer a basket. Do not stop or search. Review the footage in full before any further step.",
        verify=("Is the item visible again later in the clip?", "Did the hand stay at the waist or return to rest?", "Was the view occluded?"),
    ),
    Scenario.SHELF_SWEEP: Play(
        summary="Many items were removed from one shelf in a few seconds and most cannot be accounted for.",
        benign_explanations=(
            "A colleague is rotating or removing stock.",
            "A customer is buying in bulk and loading a trolley.",
            "A shelf sensor fault produced repeated removal events.",
        ),
        review_action="Watch the clip and check the product line. Alert the duty manager if the items went into a bag rather than a trolley.",
        priority_action="Alert the duty manager and security now. Keep a safe distance and do not approach. Preserve footage of the aisle and the exit for the evidence pack.",
        verify=("Is this person a colleague?", "Did the items go into a trolley?", "Is the product a known target line?"),
    ),
    Scenario.SKIP_SCAN: Play(
        summary="More items reached the bagging area or left the basket than were scanned.",
        benign_explanations=(
            "The shopper placed their own bag on the bagging area.",
            "One scan covered several identical items using the quantity key.",
            "A barcode failed to read and the shopper has not yet noticed.",
        ),
        review_action="Ask the self checkout host to offer help and run a friendly basket check before payment.",
        priority_action="Ask the host to pause the session and help the shopper rescan. Treat it as an honest mistake unless the review shows otherwise.",
        verify=("Compare bagged items with the receipt lines.", "Was the quantity key used?", "Is there an own bag on the scale?"),
    ),
    Scenario.TICKET_SWITCH: Play(
        summary="A scan registered a product and price that do not match the item seen at the scanner.",
        benign_explanations=(
            "The product recogniser is wrong about a new or seasonal pack.",
            "The item carries a genuine reduced to clear label.",
            "Loose produce was looked up under a similar product.",
        ),
        review_action="Ask the host to check the item against the screen and correct the line if needed.",
        priority_action="Ask the host to verify the line before payment and keep the label for the duty manager.",
        verify=("Does the receipt line match the product in the bag?", "Is there a reduction sticker?", "Is the product new to range?"),
    ),
    Scenario.SWEETHEARTING: Play(
        summary="A cashier passed several items to the packing side without a matching scan.",
        benign_explanations=(
            "Bulky items were scanned in the trolley with a handset.",
            "The quantity key was used for identical items.",
            "The item counter miscounted items that were stacked.",
        ),
        review_action="Do not approach the till. Send the clip and the receipt to the store manager for a quiet review.",
        priority_action="Refer to the store manager and people team with the clip, the receipt and the till log. Follow the internal investigation procedure.",
        verify=("Compare the receipt with the items packed.", "Check the till log for quantity key use.", "Look for a repeat pattern for this till and shift."),
    ),
    Scenario.PUSH_OUT: Play(
        summary="A person is leaving with visible merchandise and no linked payment.",
        benign_explanations=(
            "Payment was made but the sale did not link to the track.",
            "The goods are a click and collect or prepaid order.",
            "A colleague is moving stock or trolleys outside.",
        ),
        review_action="Ask the door colleague to offer a receipt check with a greeting. Check for a recent sale at the tills.",
        priority_action="Alert door security now for a receipt check. Do not block the exit. Preserve footage and note any vehicle.",
        verify=("Is there a sale in the last few minutes that fits?", "Was there a collection order?", "Did the security gate alarm sound?"),
    ),
}
