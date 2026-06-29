"""
utils.py
--------
Small shared helpers: query logging, namespace naming.
"""

import os
import re
import csv
from datetime import datetime

LOG_PATH = os.path.join("logs", "query_log.csv")


def safe_namespace(doc_name: str) -> str:
    """Turn a filename into a Pinecone-safe namespace string."""
    name = re.sub(r"\.pdf$", "", doc_name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return name.lower()


def log_query(doc_name: str, query: str, num_results: int, answer_found: bool):
    """Append a query record to the CSV log for traceability/session history."""
    os.makedirs("logs", exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "document", "query", "num_results", "answer_found"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            doc_name,
            query,
            num_results,
            answer_found,
        ])
