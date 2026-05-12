from __future__ import annotations

import os
import platform
from pathlib import Path


def effective_cpu_count() -> int:
    """Return the usable build parallelism for local compiler jobs.

    This mirrors the v1 cgroup quota behavior and prefers physical Linux cores
    when CPU topology is available, so build tools do not default to logical
    hyper-thread counts on bare metal.
    """
    logical_count = _logical_cpu_count()
    physical_count = _linux_physical_cpu_count()
    available_count = physical_count or logical_count
    quota_count = _linux_cgroup_cpu_quota_count(logical_count)
    if quota_count is not None:
        available_count = min(available_count, quota_count)
    return max(1, available_count)


def _logical_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _linux_physical_cpu_count() -> int | None:
    if platform.system() != "Linux":
        return None
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = set(range(os.cpu_count() or 1))
    cores: set[tuple[str, str]] = set()
    topology_root = Path("/sys/devices/system/cpu")
    for cpu_id in affinity:
        topology = topology_root / f"cpu{cpu_id}" / "topology"
        package_id = _read_text(topology / "physical_package_id")
        core_id = _read_text(topology / "core_id")
        if package_id is None or core_id is None:
            return None
        cores.add((package_id, core_id))
    return len(cores) or None


def _linux_cgroup_cpu_quota_count(cpu_count: int) -> int | None:
    if platform.system() in ("Windows", "Darwin"):
        return None
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip().split()
        if quota != "max":
            return max(1, min(cpu_count, int(int(quota) / int(period))))
    except Exception:
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8").strip())
        if quota > 0 and period > 0:
            return max(1, min(cpu_count, quota // period))
    except Exception:
        pass
    return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
