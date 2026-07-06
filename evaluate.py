"""
Evaluation scripts for the extraction system.

Provides entity-level evaluation of NER (precision / recall / F1 via seqeval)
and recall-oriented evaluation of the anonymizer. These scripts produce the
numbers used in the "Testing" chapter of the thesis.

Usage:
    python evaluate.py ner       # evaluate NER on a small annotated sample
    python evaluate.py anonymize # evaluate PII recall (NER vs regex)

The samples here are tiny illustrative examples. For the thesis, replace them
with a real annotated dataset (e.g. a subset of CoNLL-2003 for NER, or
ai4privacy/pii-masking-200k for anonymization).
"""

from __future__ import annotations

import sys


# ---------------------------------------------------------------------------
# NER evaluation (entity level, seqeval)
# ---------------------------------------------------------------------------
def evaluate_ner():
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from processors import ner

    # Each example: (tokens, gold BIO tags). Illustrative only.
    samples = [
        (
            ["My", "name", "is", "John", "Smith", "from", "London"],
            ["O", "O", "O", "B-PER", "I-PER", "O", "B-LOC"],
        ),
        (
            ["Sarah", "works", "at", "Google", "in", "Berlin"],
            ["B-PER", "O", "O", "B-ORG", "O", "B-LOC"],
        ),
    ]

    y_true, y_pred = [], []
    for tokens, gold in samples:
        text = " ".join(tokens)
        ents = ner.extract_entities(text)

        # Build predicted BIO tags aligned to whitespace tokens.
        pred = ["O"] * len(tokens)
        # character offset of each token
        offsets = []
        pos = 0
        for tok in tokens:
            start = text.index(tok, pos)
            offsets.append((start, start + len(tok)))
            pos = start + len(tok)

        for ent in ents:
            etype = ent["type"]
            first = True
            for i, (s, e) in enumerate(offsets):
                # token overlaps the entity span
                if s >= ent["start"] and e <= ent["end"] + 1:
                    pred[i] = ("B-" if first else "I-") + etype
                    first = False

        y_true.append(gold)
        y_pred.append(pred)

    print("=== NER evaluation (entity level) ===")
    print(classification_report(y_true, y_pred))
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")


# ---------------------------------------------------------------------------
# Anonymization evaluation (PII recall: NER+Presidio vs regex only)
# ---------------------------------------------------------------------------
def evaluate_anonymize():
    from processors import anonymizer
    import re

    # (text, list of PII substrings that MUST be removed)
    samples = [
        ("My name is John Smith, email john@example.com", ["John Smith", "john@example.com"]),
        ("Call Sarah at +1 202 555 0143", ["Sarah", "+1 202 555 0143"]),
        ("I live in Berlin and work at Google", ["Berlin", "Google"]),
    ]

    def recall(anon_fn):
        found, total = 0, 0
        for text, pii_list in samples:
            anon = anon_fn(text)
            for pii in pii_list:
                total += 1
                # PII counts as removed if it no longer appears verbatim
                if pii.lower() not in anon.lower():
                    found += 1
        return found / total if total else 0.0

    # regex-only baseline (the original approach)
    def regex_only(text):
        from processors.anonymizer import _regex_anonymize
        return _regex_anonymize(text)

    print("=== Anonymization evaluation (PII recall) ===")
    print(f"Active backend:       {anonymizer.backend_name()}")
    print(f"Recall (regex only):  {recall(regex_only):.2%}")
    print(f"Recall (full system): {recall(anonymizer.anonymize_text):.2%}")
    print("\nNote: full system requires Presidio + spaCy model for name/location recall.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ner"
    if mode == "ner":
        evaluate_ner()
    elif mode == "anonymize":
        evaluate_anonymize()
    else:
        print("Usage: python evaluate.py [ner|anonymize]")
