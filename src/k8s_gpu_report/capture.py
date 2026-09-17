"""Capture a generic Kubernetes GPU allocation snapshot with kubectl."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .snapshot import build_snapshot


def _kubectl_json(kubectl: str, *args: str) -> dict[str, Any]:
    result = subprocess.run([kubectl, *args, "-o", "json"], check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


def capture(kubectl: str = "kubectl") -> dict[str, Any]:
    pods = _kubectl_json(kubectl, "get", "pods", "--all-namespaces")
    nodes = _kubectl_json(kubectl, "get", "nodes")
    return build_snapshot(pods, nodes).as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Kubernetes GPU allocation from resource requests.")
    parser.add_argument("--output", type=Path, required=True, help="Snapshot JSON path")
    parser.add_argument("--kubectl", default="kubectl", help="kubectl executable")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capture(args.kubectl), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
