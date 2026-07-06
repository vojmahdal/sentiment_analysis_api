"""
Prepare a real customer-support dataset for the /ingest endpoint.

The ingest endpoint expects a CSV (or JSON) with a `text` (or `message`) column.
This script takes one of the recommended public datasets and produces a clean
`ingest_ready.csv` with the columns: conversation_id, speaker, text.

It auto-detects the dataset by its columns, so you can point it at any of:

  * Customer Support on Twitter   (Kaggle: thoughtvector/customer-support-on-twitter)
      columns include: tweet_id, author_id, inbound, text, ...
  * Customer Support Tickets       (HF: Tobi-Bueck/customer-support-tickets)
      columns include: subject, body, queue, priority, language, ...
  * Tech Support Conversations     (Kaggle: steve1215rogg/...)
      columns include: Conversation_ID, Customer_Issue, ...
  * Generic ticket dataset         (Customer Name / Customer Email / Ticket Description)

Usage:
    python prepare_dataset.py <input.csv> [--limit 200] [--lang en]

Then upload `ingest_ready.csv` on the dashboard (Batch ingest) or via:
    curl -X POST <space-url>/ingest -F "file=@ingest_ready.csv"
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def detect_and_extract(df: pd.DataFrame, lang: str | None) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    # --- Customer Support on Twitter ---
    if "text" in cols and "author_id" in cols and "inbound" in cols:
        # keep only inbound = customer messages (the real user-generated text)
        inbound_col = cols["inbound"]
        df = df[df[inbound_col].astype(str).str.lower().isin(["true", "1"])]
        out = pd.DataFrame(
            {
                "conversation_id": df[cols.get("tweet_id", cols["text"])].astype(str),
                "speaker": "customer",
                "text": df[cols["text"]].astype(str),
            }
        )
        return out

    # --- Tobi-Bueck customer-support-tickets (subject + body) ---
    if "body" in cols:
        if lang and "language" in cols:
            df = df[df[cols["language"]].astype(str).str.lower() == lang.lower()]
        subject = df[cols["subject"]].fillna("") if "subject" in cols else ""
        body = df[cols["body"]].fillna("")
        text = (subject + ". " + body).str.strip(". ") if "subject" in cols else body
        out = pd.DataFrame(
            {
                "conversation_id": range(1, len(df) + 1),
                "speaker": "customer",
                "text": text.astype(str),
            }
        )
        return out

    # --- Tech Support Conversations ---
    if "customer_issue" in cols:
        out = pd.DataFrame(
            {
                "conversation_id": df[cols.get("conversation_id", cols["customer_issue"])].astype(str),
                "speaker": "customer",
                "text": df[cols["customer_issue"]].astype(str),
            }
        )
        return out

    # --- Generic ticket dataset (Ticket Description) ---
    if "ticket description" in cols or "ticket_description" in cols:
        desc = cols.get("ticket description", cols.get("ticket_description"))
        out = pd.DataFrame(
            {
                "conversation_id": df[cols.get("ticket id", cols.get("ticket_id", desc))].astype(str),
                "speaker": "customer",
                "text": df[desc].astype(str),
            }
        )
        return out

    # --- Fallback: first text-like column ---
    for key in ("text", "message", "content", "body", "utterance"):
        if key in cols:
            out = pd.DataFrame(
                {
                    "conversation_id": range(1, len(df) + 1),
                    "speaker": "customer",
                    "text": df[cols[key]].astype(str),
                }
            )
            return out

    raise SystemExit(
        f"Could not find a usable text column. Columns present: {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to the downloaded dataset CSV")
    ap.add_argument("--limit", type=int, default=200, help="Max rows to keep")
    ap.add_argument("--lang", default="en", help="Language filter where supported")
    ap.add_argument("--out", default="ingest_ready.csv", help="Output CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = detect_and_extract(df, args.lang)

    # clean up: drop empties, dedupe, trim, limit
    out = out[out["text"].str.strip().astype(bool)]
    out = out.drop_duplicates(subset=["text"])
    out["text"] = out["text"].str.slice(0, 1000)  # keep messages reasonable
    out = out.head(args.limit)

    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} messages to {args.out}")
    print("Preview:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
