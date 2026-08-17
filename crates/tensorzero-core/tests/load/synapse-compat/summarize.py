# Modified by Delta-AI under Apache 2.0
"""Print a TSV summary of vegeta JSON reports produced by run.sh."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def ns_to_ms(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1_000_000:.2f}"


def rss_summary(path: Path) -> str:
    if not path.exists():
        return "-"
    values: list[int] = []
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].isdigit():
            values.append(int(parts[1]))
    if not values:
        return "-"
    return f"{min(values)}/{sum(values) // len(values)}/{max(values)}"


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synapse-compat-load")
    rows: list[tuple] = []
    for path in sorted(results_dir.glob("*.w*.json")):
        data = json.loads(path.read_text())
        name, worker = path.stem.rsplit(".w", 1)
        lat = data.get("latencies") or {}
        rows.append(
            (
                name,
                int(worker),
                int(data.get("requests") or 0),
                float(data.get("throughput") or 0.0),
                float(data.get("success") or 0.0),
                ns_to_ms(lat.get("50th") or lat.get("mean")),
                ns_to_ms(lat.get("95th")),
                ns_to_ms(lat.get("99th")),
                json.dumps(data.get("status_codes") or {}, sort_keys=True),
                rss_summary(path.with_suffix("").with_name(path.stem + ".rss.csv")),
            )
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    print("scenario\tworkers\treqs\tthroughput_rps\tsuccess\tp50_ms\tp95_ms\tp99_ms\tstatus_codes\trss_kb_min_avg_max")
    for row in rows:
        print(
            f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.1f}\t{row[4]:.4f}\t{row[5]}\t{row[6]}\t{row[7]}\t{row[8]}\t{row[9]}"
        )


if __name__ == "__main__":
    main()
