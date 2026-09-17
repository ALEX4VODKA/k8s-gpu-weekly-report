import csv
from datetime import datetime, timezone

import pytest

from k8s_gpu_report.report import aggregate_allocation, write_csv


def snapshot(at, capacity, pods):
    return {"captured_at": at, "total_gpu_capacity": capacity, "pods": pods}


def test_aggregate_uses_actual_snapshot_intervals_and_writes_csv(tmp_path):
    snapshots = [
        snapshot("2026-01-01T00:00:00+00:00", 8, [
            {"namespace": "team-a", "gpu_requested": 3},
            {"namespace": "team-b", "gpu_requested": 2},
        ]),
        snapshot("2026-01-01T00:15:00+00:00", 8, [
            {"namespace": "team-a", "gpu_requested": 1},
            {"namespace": "team-b", "gpu_requested": 4},
        ]),
    ]
    report = aggregate_allocation(snapshots, datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc))
    assert report.cluster_gpu_card_hours == 2.5
    assert report.average_cluster_allocation_ratio == 0.625
    assert [(row.namespace, row.gpu_card_hours, row.average_allocated_gpus, row.peak_allocated_gpus) for row in report.rows] == [
        ("team-b", 1.5, 3.0, 4),
        ("team-a", 1.0, 2.0, 3),
    ]
    output = tmp_path / "allocation.csv"
    write_csv(report, output)
    with output.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["namespace"] == "team-b"
    assert rows[0]["gpu_card_hours"] == "1.5"


def test_aggregate_rejects_ambiguous_or_invalid_intervals():
    item = snapshot("2026-01-01T00:00:00+00:00", 8, [])
    with pytest.raises(ValueError, match="at least one"):
        aggregate_allocation([], datetime(2026, 1, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="unique"):
        aggregate_allocation([item, item], datetime(2026, 1, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="after the final"):
        aggregate_allocation([item], datetime(2026, 1, 1, tzinfo=timezone.utc))
