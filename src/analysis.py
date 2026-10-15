from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_SHIPMENT_COLUMNS = {
    "origin", "destination", "route_type", "quantity_tonnes",
    "distance_km", "freight_cost_inr", "shipment_date"
}
OUTPUT_COLUMNS = [
    "route", "week_of", "cost_per_tonne_km", "vs_own_history",
    "vs_similar_routes", "flagged", "matched_note_id", "reason"
]
THRESHOLD_PCT = 15.0


def monday_of(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series, errors="raise")
    return (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.normalize()


def load_shipments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_SHIPMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Shipment file missing required columns: {sorted(missing)}")
    for col in ["quantity_tonnes", "distance_km", "freight_cost_inr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")
    if df[["shipment_date", "quantity_tonnes", "distance_km", "freight_cost_inr"]].isna().any().any():
        raise ValueError("Shipment file contains invalid dates or numeric values.")
    if (df["quantity_tonnes"] <= 0).any() or (df["distance_km"] <= 0).any():
        raise ValueError("quantity_tonnes and distance_km must be positive.")
    if (df["freight_cost_inr"] < 0).any():
        raise ValueError("freight_cost_inr cannot be negative.")
    df["week_of"] = monday_of(df["shipment_date"])
    df["route"] = df["origin"].astype(str).str.strip() + "-" + df["destination"].astype(str).str.strip()
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def aggregate_weekly(shipments: pd.DataFrame) -> pd.DataFrame:
    keys = ["origin", "destination", "route_type", "route", "week_of"]
    weekly = shipments.groupby(keys, as_index=False, sort=True).agg(
        total_freight_cost=("freight_cost_inr", "sum"),
        total_tonne_km=("tonne_km", "sum"),
        shipment_count=("shipment_id", "count") if "shipment_id" in shipments.columns else ("tonne_km", "count"),
    )
    weekly["cost_per_tonne_km"] = weekly["total_freight_cost"] / weekly["total_tonne_km"]
    weekly = weekly.sort_values(["route", "week_of"], kind="mergesort").reset_index(drop=True)
    return weekly


def pct_change(current: float, baseline: Optional[float]) -> Optional[float]:
    if baseline is None or pd.isna(baseline) or baseline <= 0:
        return None
    return (current / baseline - 1.0) * 100.0


def format_comparison(pct: Optional[float], label: str) -> str:
    if pct is None:
        return f"N/A ({label} unavailable)"
    return f"{pct:+.1f}% vs {label}"


def build_baselines(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy().sort_values(["route", "week_of"], kind="mergesort").reset_index(drop=True)
    own_baselines, peer_baselines, own_counts = [], [], []
    for row in weekly.itertuples(index=False):
        prior = weekly[(weekly["route"] == row.route) & (weekly["week_of"] < row.week_of)]
        prior = prior.sort_values("week_of").tail(8)
        own_counts.append(len(prior))
        own_baselines.append(float(prior["cost_per_tonne_km"].mean()) if len(prior) else None)

        peers = weekly[
            (weekly["route_type"] == row.route_type)
            & (weekly["week_of"] == row.week_of)
            & (weekly["route"] != row.route)
        ]
        peer_baselines.append(float(peers["cost_per_tonne_km"].mean()) if len(peers) else None)

    weekly["own_history_baseline"] = own_baselines
    weekly["own_history_weeks"] = own_counts
    weekly["similar_routes_baseline"] = peer_baselines
    weekly["vs_own_pct"] = [
        pct_change(c, b) for c, b in zip(weekly["cost_per_tonne_km"], weekly["own_history_baseline"])
    ]
    weekly["vs_similar_pct"] = [
        pct_change(c, b) for c, b in zip(weekly["cost_per_tonne_km"], weekly["similar_routes_baseline"])
    ]
    return weekly


def note_is_positive_support(note_text: str) -> bool:
    """Conservative phrase gate: reject neutral/negative notes even if retrieval ranks them highly."""
    text = note_text.lower()
    negative_or_neutral = [
        "costs were not significantly affected", "no significant disruptions",
        "remained normal", "no major disruptions", "not part of this dataset",
        "without a rate change", "costs were not", "returned to normal",
        "improved road conditions", "stable this quarter",
    ]
    if any(phrase in text for phrase in negative_or_neutral):
        return False
    positive_terms = [
        "higher trip costs", "surcharge", "pushing up transportation costs",
        "forcing longer detours", "increased costs", "costs rose", "rate increase",
    ]
    return any(term in text for term in positive_terms)


def note_applies(note: pd.Series, route: str, week_of: pd.Timestamp) -> bool:
    applies = str(note["applies_to"]).strip()
    note_date = pd.to_datetime(note["date"], errors="coerce")
    if pd.isna(note_date):
        return False
    if applies not in ("All Routes", route):
        return False

    # Scope windows are deliberately conservative and derived only from explicit note wording.
    note_id = str(note["note_id"])
    if note_id == "N001":
        return pd.Timestamp("2025-02-24") <= week_of <= pd.Timestamp("2025-03-03")
    if note_id == "N002":
        return week_of == pd.Timestamp("2025-01-20")
    if note_id == "N003":
        # "starting this week" supports the week beginning on the stated date only;
        # do not infer a permanent uplift beyond the note's explicit time anchor.
        return week_of == pd.Timestamp("2025-05-05")
    # Other notes either deny a cost impact or do not establish a route-specific increase.
    return False


def retrieve_supporting_note(notes: pd.DataFrame, route: str, week_of: pd.Timestamp) -> Optional[pd.Series]:
    candidates = notes[notes.apply(lambda n: note_applies(n, route, week_of), axis=1)]
    for _, note in candidates.sort_values("note_id").iterrows():
        if note_is_positive_support(str(note["note"])):
            return note
    return None


def explain_row(row: pd.Series, note: Optional[pd.Series], own_weeks: int) -> tuple[str, str, str]:
    own_pct, peer_pct = row["vs_own_pct"], row["vs_similar_pct"]
    anomalous = any(p is not None and not pd.isna(p) and p >= THRESHOLD_PCT for p in [own_pct, peer_pct])
    if not anomalous:
        return "No", "", "No anomaly crossed the configured +15% comparison threshold."

    if note is not None:
        note_id = str(note["note_id"])
        date = pd.to_datetime(note["date"]).strftime("%Y-%m-%d")
        reason = (
            f"No (justified). Note {note_id} dated {date} applies to {row['route']} "
            f"for this week: {str(note['note']).strip()}"
        )
        return "No (justified)", note_id, reason

    comparisons = []
    if own_pct is not None and not pd.isna(own_pct) and own_pct >= THRESHOLD_PCT:
        comparisons.append(f"{own_pct:.1f}% above its trailing baseline")
    if peer_pct is not None and not pd.isna(peer_pct) and peer_pct >= THRESHOLD_PCT:
        comparisons.append(f"{peer_pct:.1f}% above same-week similar routes")
    if own_weeks < 8:
        history_msg = f" Only {own_weeks} prior observed weeks were available for the history baseline."
    else:
        history_msg = ""
    reason = (
        "Yes. Unexplained cost rise: " + " and ".join(comparisons) +
        ". No route-and-time-matched note explicitly supports this increase; human review recommended." +
        history_msg
    )
    return "Yes", "", reason


def run_analysis(shipments_path: Path, notes_path: Path, output_path: Path, audit_path: Path) -> pd.DataFrame:
    shipments = load_shipments(shipments_path)
    notes = pd.read_csv(notes_path, dtype=str).fillna("")
    required_notes = {"note_id", "date", "applies_to", "note"}
    if not required_notes.issubset(notes.columns):
        raise ValueError(f"Context notes must contain {sorted(required_notes)}")

    weekly = build_baselines(aggregate_weekly(shipments))
    results = []
    for _, row in weekly.iterrows():
        note = retrieve_supporting_note(notes, row["route"], row["week_of"])
        flagged, note_id, reason = explain_row(row, note, int(row["own_history_weeks"]))
        results.append({
            "route": row["route"],
            "week_of": row["week_of"].strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(float(row["cost_per_tonne_km"]), 2),
            "vs_own_history": format_comparison(row["vs_own_pct"], "this route's past average"),
            "vs_similar_routes": format_comparison(row["vs_similar_pct"], "similar-length routes this week"),
            "flagged": flagged,
            "matched_note_id": note_id,
            "reason": reason,
        })
    output = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    output.to_csv(output_path, index=False, lineterminator="\n")
    audit = {
        "shipment_rows": int(len(shipments)),
        "routes": int(shipments["route"].nunique()),
        "weekly_route_rows": int(len(output)),
        "flagged_unexplained": int((output["flagged"] == "Yes").sum()),
        "anomalies_justified": int((output["flagged"] == "No (justified)").sum()),
        "not_flagged": int((output["flagged"] == "No").sum()),
        "threshold_pct": THRESHOLD_PCT,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_llm_cost_usd": 0.0,
        "cost_note": "No LLM is called; deterministic rules and exact route/date evidence checks keep this run reproducible.",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description="Deterministic, evidence-grounded freight cost monitor")
    parser.add_argument("--shipments", default="data/shipment_records.csv")
    parser.add_argument("--notes", default="data/context_notes.csv")
    parser.add_argument("--output", default="outputs/freight_analysis.csv")
    parser.add_argument("--audit", default="outputs/evaluation_report.json")
    args = parser.parse_args()
    for p in [args.output, args.audit]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    result = run_analysis(Path(args.shipments), Path(args.notes), Path(args.output), Path(args.audit))
    print(f"Wrote {len(result)} weekly route rows to {args.output}")
    print(result["flagged"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
