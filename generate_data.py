"""Generate a deterministic, entirely synthetic telecom churn sample."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

FEATURES = (
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "late_payments",
    "contract_months",
    "fiber_service",
    "autopay",
    "streaming_services",
)
FIELDS = ("customer_id", *FEATURES, "churned")


def make_rows(count: int = 7043, seed: int = 42) -> list[dict[str, int | float | str]]:
    if count < 10:
        raise ValueError("count must be at least 10")
    rng = random.Random(seed)
    rows: list[dict[str, int | float | str]] = []
    for index in range(count):
        tenure = rng.randint(1, 72)
        contract = rng.choice((1, 1, 1, 12, 24))
        fiber = rng.randrange(2)
        autopay = rng.randrange(2)
        streaming = rng.randint(0, 3)
        tickets = min(8, int(rng.expovariate(0.7)))
        late = min(6, int(rng.expovariate(1.2)))
        charges = round(25 + fiber * 32 + streaming * 8 + rng.uniform(0, 34), 2)
        log_odds = (
            -1.3
            - 0.025 * tenure
            + 0.035 * (charges - 60)
            + 0.30 * tickets
            + 0.38 * late
            - 0.055 * contract
            - 0.55 * autopay
        )
        probability = 1 / (1 + math.exp(-log_odds))
        rows.append(
            {
                "customer_id": f"SYN-{index + 1:06d}",
                "tenure_months": tenure,
                "monthly_charges": charges,
                "support_tickets": tickets,
                "late_payments": late,
                "contract_months": contract,
                "fiber_service": fiber,
                "autopay": autopay,
                "streaming_services": streaming,
                "churned": int(rng.random() < probability),
            }
        )
    return rows


def write_csv(path: Path, count: int = 7043, seed: int = 42) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(make_rows(count, seed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/telecom_synthetic.csv"))
    parser.add_argument("--rows", type=int, default=7043)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_csv(args.output, args.rows, args.seed)
    print(f"Wrote {args.rows:,} synthetic rows to {args.output}")
