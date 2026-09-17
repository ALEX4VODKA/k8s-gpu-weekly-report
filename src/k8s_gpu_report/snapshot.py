"""Parse Kubernetes Pod requests into a GPU allocation snapshot.

This module measures requested GPU allocation from ``nvidia.com/gpu`` resource
requests and node GPU capacity. It does not collect GPU device utilization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

GPU_RESOURCE = "nvidia.com/gpu"
TERMINAL_POD_PHASES = frozenset({"Succeeded", "Failed"})


@dataclass(frozen=True)
class PodGPUAllocation:
    """GPU allocation requested by a scheduled, non-terminal Pod."""

    namespace: str
    pod_name: str
    node_name: str
    phase: str
    gpu_requested: int


@dataclass(frozen=True)
class ClusterGPUSnapshot:
    """A point-in-time GPU allocation summary derived from Kubernetes objects."""

    captured_at: str
    gpu_resource: str
    node_gpu_capacity: dict[str, int]
    pods: tuple[PodGPUAllocation, ...]

    @property
    def total_gpu_capacity(self) -> int:
        return sum(self.node_gpu_capacity.values())

    @property
    def total_gpu_requested(self) -> int:
        return sum(pod.gpu_requested for pod in self.pods)

    @property
    def allocation_ratio(self) -> float | None:
        if not self.total_gpu_capacity:
            return None
        return self.total_gpu_requested / self.total_gpu_capacity

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "captured_at": self.captured_at,
            "metric": "gpu_allocation",
            "gpu_resource": self.gpu_resource,
            "node_gpu_capacity": self.node_gpu_capacity,
            "total_gpu_capacity": self.total_gpu_capacity,
            "total_gpu_requested": self.total_gpu_requested,
            "allocation_ratio": self.allocation_ratio,
            "pods": [asdict(pod) for pod in self.pods],
        }


def _quantity(value: Any) -> int:
    """Parse an integral extended-resource quantity, treating a missing value as 0."""
    if value is None:
        return 0
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid GPU quantity: {value!r}") from exc
    if parsed < 0 or parsed != parsed.to_integral_value():
        raise ValueError(f"GPU quantity must be a non-negative integer: {value!r}")
    return int(parsed)


def _resource_request(container: Mapping[str, Any]) -> int:
    resources = container.get("resources") or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    # Kubernetes extended resources conventionally use matching request and limit.
    # If only a limit is supplied, Kubernetes treats it as the effective request.
    return _quantity(requests.get(GPU_RESOURCE, limits.get(GPU_RESOURCE)))


def pod_gpu_request(pod: Mapping[str, Any]) -> int:
    """Return the scheduler-effective GPU request for one Pod specification.

    App-container requests are summed. Init containers execute sequentially, so
    their maximum request is compared with the app-container total. Pod overhead
    is then added, matching Kubernetes scheduler resource accounting.
    """
    spec = pod.get("spec") or {}
    app_total = sum(_resource_request(container) for container in spec.get("containers") or [])
    init_peak = max((_resource_request(container) for container in spec.get("initContainers") or []), default=0)
    overhead = _quantity(((spec.get("overhead") or {}).get(GPU_RESOURCE)))
    return max(app_total, init_peak) + overhead


def node_gpu_capacities(nodes: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Read advertised GPU capacity from each Kubernetes Node."""
    result: dict[str, int] = {}
    for node in nodes:
        name = str(((node.get("metadata") or {}).get("name") or "")).strip()
        if not name:
            continue
        capacity = ((node.get("status") or {}).get("capacity") or {}).get(GPU_RESOURCE)
        result[name] = _quantity(capacity)
    return result


def scheduled_gpu_pods(pods: Iterable[Mapping[str, Any]]) -> tuple[PodGPUAllocation, ...]:
    """Return scheduled non-terminal Pods with a positive GPU request."""
    allocations: list[PodGPUAllocation] = []
    for pod in pods:
        metadata = pod.get("metadata") or {}
        spec = pod.get("spec") or {}
        status = pod.get("status") or {}
        namespace = str(metadata.get("namespace") or "default").strip()
        pod_name = str(metadata.get("name") or "").strip()
        node_name = str(spec.get("nodeName") or "").strip()
        phase = str(status.get("phase") or "Unknown").strip()
        request = pod_gpu_request(pod)
        if not pod_name or not node_name or phase in TERMINAL_POD_PHASES or request == 0:
            continue
        allocations.append(PodGPUAllocation(namespace, pod_name, node_name, phase, request))
    return tuple(sorted(allocations, key=lambda item: (item.namespace, item.pod_name)))


def build_snapshot(pods_document: Mapping[str, Any], nodes_document: Mapping[str, Any], captured_at: datetime | None = None) -> ClusterGPUSnapshot:
    """Build a GPU allocation snapshot from ``kubectl get ... -o json`` documents."""
    timestamp = captured_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    return ClusterGPUSnapshot(
        captured_at=timestamp.astimezone(timezone.utc).isoformat(),
        gpu_resource=GPU_RESOURCE,
        node_gpu_capacity=node_gpu_capacities(nodes_document.get("items") or []),
        pods=scheduled_gpu_pods(pods_document.get("items") or []),
    )
