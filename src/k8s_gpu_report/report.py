"""Aggregate GPU allocation snapshots and write a compact CSV report.

The report measures requested ``nvidia.com/gpu`` allocation over time. It does
not measure GPU device utilization.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class NamespaceAllocationRow:
    namespace: str
    gpu_card_hours: float
    average_allocated_gpus: float
    peak_allocated_gpus: int
    snapshot_count: int


@dataclass(frozen=True)
class AllocationReport:
    period_start: str
    period_end: str
    cluster_gpu_card_hours: float
    average_cluster_allocation_ratio: float | None
    rows: tuple[NamespaceAllocationRow, ...]


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("snapshot captured_at must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("snapshot captured_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _allocation_by_namespace(snapshot: Mapping[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for pod in snapshot.get("pods") or []:
        namespace = str(pod.get("namespace") or "default").strip()
        requested = pod.get("gpu_requested")
        if not isinstance(requested, int) or requested < 0:
            raise ValueError("pod gpu_requested must be a non-negative integer")
        values[namespace] = values.get(namespace, 0) + requested
    return values


def aggregate_allocation(snapshots: Iterable[Mapping[str, Any]], period_end: datetime) -> AllocationReport:
    """Aggregate snapshots using each snapshot's actual interval to the next one.

    A snapshot applies from its ``captured_at`` up to the next snapshot. The last
    snapshot applies up to ``period_end``. This makes partial or irregular capture
    intervals explicit instead of assuming a fixed polling interval.
    """
    end = period_end.astimezone(timezone.utc) if period_end.tzinfo else None
    if end is None:
        raise ValueError("period_end must be timezone-aware")
    ordered = sorted(((_timestamp(item.get("captured_at")), item) for item in snapshots), key=lambda pair: pair[0])
    if not ordered:
        raise ValueError("at least one snapshot is required")
    if len({at for at, _ in ordered}) != len(ordered):
        raise ValueError("snapshot captured_at values must be unique")
    if ordered[-1][0] >= end:
        raise ValueError("period_end must be after the final snapshot")

    totals: dict[str, float] = {}
    peaks: dict[str, int] = {}
    sample_counts: dict[str, int] = {}
    cluster_card_hours = 0.0
    capacity_card_hours = 0.0
    namespaces: set[str] = set()

    for index, (captured_at, snapshot) in enumerate(ordered):
        next_at = ordered[index + 1][0] if index + 1 < len(ordered) else end
        hours = (next_at - captured_at).total_seconds() / 3600
        if hours <= 0:
            raise ValueError("snapshot timestamps must be strictly increasing")
        allocations = _allocation_by_namespace(snapshot)
        namespaces.update(allocations)
        total_allocated = sum(allocations.values())
        capacity = snapshot.get("total_gpu_capacity")
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError("snapshot total_gpu_capacity must be a non-negative integer")
        cluster_card_hours += total_allocated * hours
        capacity_card_hours += capacity * hours
        for namespace, requested in allocations.items():
            totals[namespace] = totals.get(namespace, 0.0) + requested * hours
            peaks[namespace] = max(peaks.get(namespace, 0), requested)
            sample_counts[namespace] = sample_counts.get(namespace, 0) + 1

    total_hours = (end - ordered[0][0]).total_seconds() / 3600
    rows = tuple(sorted((
        NamespaceAllocationRow(
            namespace=namespace,
            gpu_card_hours=totals.get(namespace, 0.0),
            average_allocated_gpus=totals.get(namespace, 0.0) / total_hours,
            peak_allocated_gpus=peaks.get(namespace, 0),
            snapshot_count=sample_counts.get(namespace, 0),
        )
        for namespace in namespaces
    ), key=lambda row: (-row.gpu_card_hours, row.namespace)))
    return AllocationReport(
        period_start=ordered[0][0].isoformat(),
        period_end=end.isoformat(),
        cluster_gpu_card_hours=cluster_card_hours,
        average_cluster_allocation_ratio=cluster_card_hours / capacity_card_hours if capacity_card_hours else None,
        rows=rows,
    )


def write_csv(report: AllocationReport, output: Path) -> None:
    """Write one namespace allocation row per CSV record."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["namespace", "gpu_card_hours", "average_allocated_gpus", "peak_allocated_gpus", "snapshot_count"])
        writer.writeheader()
        writer.writerows(asdict(row) for row in report.rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Kubernetes GPU allocation snapshots into CSV.")
    parser.add_argument("--snapshot", type=Path, action="append", required=True, help="Snapshot JSON file; repeat in chronological order")
    parser.add_argument("--period-end", required=True, help="ISO-8601 end timestamp with timezone")
    parser.add_argument("--output", type=Path, required=True, help="CSV report path")
    args = parser.parse_args()
    snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in args.snapshot]
    report = aggregate_allocation(snapshots, _timestamp(args.period_end))
    write_csv(report, args.output)


if __name__ == "__main__":
    main()
