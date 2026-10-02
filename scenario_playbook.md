# Scenario playbook

What each alert means, the innocent explanations a reviewer must rule out, and the proportionate response.
The system never accuses anyone. An alert asks a trained colleague to look.

## Concealment in a bag

**Camera context** Aisle

An item taken from the shelf appears to have gone into a personal bag without being scanned or basketed.

**Innocent explanations to rule out**

* The shopper is using scan and go and the handset scan was not recorded.
* The object was a personal item such as a phone, list or purse.
* The bag is the shopper's own shopping bag and they intend to pay at the till.

**Review tier response** Watch the clip. If the release into the bag is confirmed, ask a colleague to offer help in the aisle and note the track for the checkout team.

**Priority tier response** Send a colleague to offer a basket and assistance now. Check whether the item is presented at checkout before any further step.

**Checks for the reviewer**

* Was a shelf removal recorded?
* Did the item reappear in a basket or at a till?
* Is scan and go active for this visit?

## Concealment in clothing

**Camera context** Aisle

An item taken from the shelf appears to have been tucked under clothing at the waist.

**Innocent explanations to rule out**

* The shopper is carrying the item against their body because they have no basket.
* The hand went to a pocket for a phone or keys and the item detector lost the product.
* The camera view was blocked by another shopper at the key moment.

**Review tier response** Watch the clip. Look for the item in the hand after the waist movement. If it is gone, ask a colleague to offer help in the aisle.

**Priority tier response** Send a colleague to greet the shopper and offer a basket. Do not stop or search. Review the footage in full before any further step.

**Checks for the reviewer**

* Is the item visible again later in the clip?
* Did the hand stay at the waist or return to rest?
* Was the view occluded?

## Shelf sweep

**Camera context** Aisle

Many items were removed from one shelf in a few seconds and most cannot be accounted for.

**Innocent explanations to rule out**

* A colleague is rotating or removing stock.
* A customer is buying in bulk and loading a trolley.
* A shelf sensor fault produced repeated removal events.

**Review tier response** Watch the clip and check the product line. Alert the duty manager if the items went into a bag rather than a trolley.

**Priority tier response** Alert the duty manager and security now. Keep a safe distance and do not approach. Preserve footage of the aisle and the exit for the evidence pack.

**Checks for the reviewer**

* Is this person a colleague?
* Did the items go into a trolley?
* Is the product a known target line?

## Self checkout skip scan

**Camera context** Self checkout

More items reached the bagging area or left the basket than were scanned.

**Innocent explanations to rule out**

* The shopper placed their own bag on the bagging area.
* One scan covered several identical items using the quantity key.
* A barcode failed to read and the shopper has not yet noticed.

**Review tier response** Ask the self checkout host to offer help and run a friendly basket check before payment.

**Priority tier response** Ask the host to pause the session and help the shopper rescan. Treat it as an honest mistake unless the review shows otherwise.

**Checks for the reviewer**

* Compare bagged items with the receipt lines.
* Was the quantity key used?
* Is there an own bag on the scale?

## Ticket switch

**Camera context** Self checkout

A scan registered a product and price that do not match the item seen at the scanner.

**Innocent explanations to rule out**

* The product recogniser is wrong about a new or seasonal pack.
* The item carries a genuine reduced to clear label.
* Loose produce was looked up under a similar product.

**Review tier response** Ask the host to check the item against the screen and correct the line if needed.

**Priority tier response** Ask the host to verify the line before payment and keep the label for the duty manager.

**Checks for the reviewer**

* Does the receipt line match the product in the bag?
* Is there a reduction sticker?
* Is the product new to range?

## Sweethearting at the till

**Camera context** Staffed till

A cashier passed several items to the packing side without a matching scan.

**Innocent explanations to rule out**

* Bulky items were scanned in the trolley with a handset.
* The quantity key was used for identical items.
* The item counter miscounted items that were stacked.

**Review tier response** Do not approach the till. Send the clip and the receipt to the store manager for a quiet review.

**Priority tier response** Refer to the store manager and people team with the clip, the receipt and the till log. Follow the internal investigation procedure.

**Checks for the reviewer**

* Compare the receipt with the items packed.
* Check the till log for quantity key use.
* Look for a repeat pattern for this till and shift.

## Trolley push out

**Camera context** Store exit

A person is leaving with visible merchandise and no linked payment.

**Innocent explanations to rule out**

* Payment was made but the sale did not link to the track.
* The goods are a click and collect or prepaid order.
* A colleague is moving stock or trolleys outside.

**Review tier response** Ask the door colleague to offer a receipt check with a greeting. Check for a recent sale at the tills.

**Priority tier response** Alert door security now for a receipt check. Do not block the exit. Preserve footage and note any vehicle.

**Checks for the reviewer**

* Is there a sale in the last few minutes that fits?
* Was there a collection order?
* Did the security gate alarm sound?

## Benign behaviours the detector is trained to ignore

| Camera | Behaviour | Why it is in the corpus |
| :-- | :-- | :-- |
| Aisle | `browse` | Looks along the shelf, may touch products |
| Aisle | `pick_to_basket` | Takes items and puts them in a basket or trolley |
| Aisle | `pick_and_return` | Inspects an item then puts it back |
| Aisle | `phone_or_pocket` | Takes a phone from a pocket and puts it back |
| Aisle | `own_bag_adjust` | Reaches into own bag for a list, wallet or phone |
| Aisle | `scan_and_go` | Scans items with a handset and packs them into own bag |
| Aisle | `staff_restock` | Colleague moves stock from a cage onto the shelf |
| Aisle | `bulk_buyer` | Loads many items quickly into a trolley |
| Aisle | `carry_in_hand` | No basket: carries one or two items held against the body |
| Self checkout | `sco_normal` | Scans and bags every item |
| Self checkout | `sco_rescan` | Barcode will not read, shopper retries |
| Self checkout | `sco_produce` | Weighs loose produce that has no barcode |
| Self checkout | `sco_own_bag` | Places own shopping bag on the bagging area |
| Self checkout | `sco_void` | Changes mind and has a scanned item removed |
| Self checkout | `sco_multibuy` | Scans one item and keys the quantity for the rest |
| Self checkout | `sco_markdown` | Buys reduced to clear items with yellow stickers |
| Staffed till | `till_normal` | Cashier scans every item |
| Staffed till | `till_bulky` | Cashier scans bulky goods in the trolley with a handset |
| Staffed till | `till_multibuy` | Cashier scans one and keys the quantity |
| Staffed till | `till_rescan` | Cashier struggles with a damaged barcode |
| Store exit | `exit_paid_till` | Leaves after paying at a staffed till |
| Store exit | `exit_paid_sco` | Leaves after paying at self checkout |
| Store exit | `exit_no_purchase` | Leaves without buying anything |
| Store exit | `exit_staff` | Colleague takes trolleys or a cage outside |
| Store exit | `exit_collection` | Leaves with a click and collect or prepaid order |
