# FreightTiger Shipping Cost Assistant

AI-assisted shipping cost anomaly detection with evidence-based explanations

## What it does
This project calculates weekly route cost per tonne-km, compares each route with its own trailing history and same-week routes of the same `route_type`, then flags unusual increases. Context notes are only allowed to justify a flag when both route scope and date scope match and the note explicitly describes an increase.

The numeric pipeline and verdicts are deterministic. No LLM is called, so token usage and LLM cost are both zero for this run. This is an intentional trust-first trade-off for a small, fixed set of ten notes: a language model is not needed to decide numeric results, and it should not be allowed to invent a cause. Plain-English reasons are built from calculated values and verbatim supporting note text.

## Inputs
- `data/shipment_records.csv`
- `data/context_notes.csv`
- `data/sample_output_format_v2.csv` (column contract/examples supplied by the challenge)

## Run
Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python run.py
```

Output files:
- `outputs/freight_analysis.csv`
- `outputs/evaluation_report.json`
- `outputs/run_cost_log.json`

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Core calculations
1. `week_of` is the Monday of the shipment's Monday-Sunday week.
2. Weekly cost per tonne-km is `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`. This correctly weights shipment ratios by tonne-km rather than taking a simple mean of shipment-level ratios.
3. Each route is the exact origin-destination pair and is grouped together with its provided `route_type`.
4. Own-history baseline is the arithmetic mean of up to the previous eight *observed route-weeks*, strictly before the current week. If fewer than eight prior observations exist, all available prior observations are used; no padding or look-ahead occurs.
5. Peer baseline is the mean cost for all other routes with the same `route_type` in the same week. The current route is excluded. If no peer route has an observation that week, the comparison is unavailable.
6. A route is anomalous if its current cost is at least 15% above either available baseline. The 15% threshold is an explicit implementation choice because the brief does not prescribe one. Change `THRESHOLD_PCT` in `src/analysis.py` to test alternatives.

## Evidence and verdict rules
- A positive cost-impact note must match the exact route or `All Routes`.
- Note date scope is conservative. N001 applies to the weeks beginning 2025-02-24 and 2025-03-03, because the note says disruption ran Feb 24 through Mar 8. N002 applies to the week beginning 2025-01-20. N003 says the diesel rise starts "this week", so it is only used for the week beginning 2025-05-05 rather than being extrapolated indefinitely.
- N004 through N010 are not used to justify increases: their wording either says the affected routes are outside the dataset, costs were not significantly affected, conditions were normal/stable, or costs were absorbed without a rate change.
- If an anomaly has an exact applicable positive note, `flagged` is `No (justified)` and `matched_note_id` contains that note ID. If no valid evidence is found, it remains `Yes` and the note ID is blank. Non-anomalies are `No`.
- No note is treated as evidence just because it is semantically similar. In a larger version, use TF-IDF or sentence embeddings to retrieve candidates, then retain these same strict metadata and evidence gates. A vector database is not necessary for ten short notes.

## Reproducibility and cost
`run.py` runs deterministic Pandas calculations and deterministic evidence rules. The generated run log includes shipment count, weekly row count, flags, threshold, and LLM token/cost fields. Since no LLM is called, input tokens = 0, output tokens = 0, LLM calls = 0, estimated LLM cost = $0.00. This does not claim that general compute or developer time is free.

To independently verify repeatability, run `python verify_reproducibility.py`. It runs the analysis three times to temporary files and compares the CSV outputs byte-for-byte. Wording, numbers, verdicts, and matched note IDs should match exactly.

## Limitations and next steps
- The 15% threshold is a transparent starting assumption, not a learned or calibrated business threshold. Validate it with a labeled review set before production use.
- A peer baseline is unavailable if no other route of the same type has data in that week.
- The route/date rules for notes are deliberately conservative. For production use, store explicit `valid_from`, `valid_to`, route IDs, impact direction, and confidence/approval fields in structured note metadata.
- The optional natural-language Q&A stretch goal is not implemented; it should answer only from the analysis CSV and cited context notes.
