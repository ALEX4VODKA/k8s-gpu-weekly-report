from datetime import datetime, timezone

import pytest

from k8s_gpu_report.snapshot import build_snapshot, pod_gpu_request


def pod(name, node, phase, containers, namespace="team-a", init=None, overhead=None):
    return {
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"nodeName": node, "containers": containers, "initContainers": init or [], "overhead": overhead or {}},
        "status": {"phase": phase},
    }


def container(request=None, limit=None):
    resources = {"requests": {}, "limits": {}}
    if request is not None:
        resources["requests"]["nvidia.com/gpu"] = request
    if limit is not None:
        resources["limits"]["nvidia.com/gpu"] = limit
    return {"resources": resources}


def test_snapshot_uses_scheduled_nonterminal_gpu_pods_and_node_capacity():
    pods = {"items": [
        pod("trainer-a", "node-gpu-a", "Running", [container("2"), container(limit="1")]),
        pod("queued", "", "Pending", [container("4")]),
        pod("finished", "node-gpu-b", "Succeeded", [container("8")]),
        pod("cpu-only", "node-gpu-a", "Running", [container()]),
    ]}
    nodes = {"items": [
        {"metadata": {"name": "node-gpu-a"}, "status": {"capacity": {"nvidia.com/gpu": "8"}}},
        {"metadata": {"name": "node-gpu-b"}, "status": {"capacity": {"nvidia.com/gpu": "4"}}},
    ]}
    snapshot = build_snapshot(pods, nodes, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert snapshot.total_gpu_capacity == 12
    assert snapshot.total_gpu_requested == 3
    assert snapshot.allocation_ratio == 0.25
    assert [item.pod_name for item in snapshot.pods] == ["trainer-a"]
    assert snapshot.as_dict()["metric"] == "gpu_allocation"


def test_init_container_peak_and_pod_overhead_follow_scheduler_accounting():
    item = pod(
        "init-heavy", "node-gpu-a", "Running", [container("2"), container("1")],
        init=[container("6"), container("4")], overhead={"nvidia.com/gpu": "1"},
    )
    assert pod_gpu_request(item) == 7


def test_gpu_quantities_must_be_integral():
    item = pod("invalid", "node-gpu-a", "Running", [container("0.5")])
    with pytest.raises(ValueError, match="non-negative integer"):
        pod_gpu_request(item)
