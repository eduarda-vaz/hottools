"""Read effective RAM headroom without assuming node RAM belongs to one job."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

_MIB = 1024**2


def _integer_file(path: Path) -> int | None:
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError):
        return None  # includes cgroup v2's literal 'max'
    return value if value >= 0 else None


def _mount_path(value: str) -> Path:
    return Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value))


def _cgroup_headrooms(proc_root: Path) -> dict[str, int]:
    """Read hard limits and current usage at this cgroup and its ancestors."""
    try:
        memberships = (proc_root / "self/cgroup").read_text().splitlines()
        mounts = (proc_root / "self/mountinfo").read_text().splitlines()
    except OSError:
        return {}
    paths = {}
    for line in memberships:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and not parts[1]:
            paths["cgroup2"] = parts[2]
        elif "memory" in parts[1].split(","):
            paths["cgroup"] = parts[2]
    found = {}
    for mount in mounts:
        before, separator, after = mount.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 5 or len(right) < 3:
            continue
        kind = right[0]
        if kind not in paths or (
            kind == "cgroup" and "memory" not in right[2].split(",")
        ):
            continue
        member = PurePosixPath(paths[kind])
        root = PurePosixPath(str(_mount_path(left[3])))
        if not member.is_absolute() or ".." in member.parts:
            continue
        if member.is_relative_to(root):
            relative = member.relative_to(root)
        elif member == PurePosixPath("/"):
            # A cgroup namespace can expose its own root as '/'.
            relative = PurePosixPath(".")
        else:
            continue
        mountpoint = _mount_path(left[4])
        current = mountpoint.joinpath(*relative.parts)
        limit_name, usage_name = (
            ("memory.max", "memory.current")
            if kind == "cgroup2"
            else ("memory.limit_in_bytes", "memory.usage_in_bytes")
        )
        while True:
            limit = _integer_file(current / limit_name)
            usage = _integer_file(current / usage_name)
            if limit is not None and usage is not None:
                found[f"{kind}:{current}"] = max(0, limit - usage)
            if current == mountpoint:
                break
            current = current.parent
    return found


def effective_memory_available(
    host_available: int,
    *,
    process_rss: int = 0,
    environ: Mapping[str, str] | None = None,
    proc_root: Path = Path("/proc"),
) -> tuple[int, dict[str, int]]:
    """Minimum of host, cgroup and known SLURM headroom, in bytes.

    A scheduler limit without cgroup accounting is only an upper bound:
    subtract this process's RSS, while other job processes remain unknown.
    Missing/malformed metadata is ignored; no scheduler value is assumed.
    """
    env = os.environ if environ is None else environ
    constraints = {"host_available": max(0, int(host_available))}
    constraints.update(_cgroup_headrooms(proc_root))

    def positive_env(name):
        try:
            value = int(env.get(name, ""))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    node_mb = positive_env("SLURM_MEM_PER_NODE")
    per_cpu_mb = positive_env("SLURM_MEM_PER_CPU")
    cpus = positive_env("SLURM_CPUS_ON_NODE") or positive_env("SLURM_CPUS_PER_TASK")
    for name, total in (
        ("slurm_node", None if node_mb is None else node_mb * _MIB),
        (
            "slurm_cpu",
            None if per_cpu_mb is None or cpus is None else per_cpu_mb * cpus * _MIB,
        ),
    ):
        if total is not None:
            constraints[name] = max(0, total - max(0, int(process_rss)))
    return min(constraints.values()), constraints
