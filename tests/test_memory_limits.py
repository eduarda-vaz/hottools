"""Synthetic cgroup mounts and scheduler metadata; no real limits are changed."""

from types import SimpleNamespace

import pytest

import hottools.memory as memory
from hottools.memory_limits import effective_memory_available

GIB = 1024**3


def cgroup(tmp_path, version=2, membership="/job/step", mount_root="/"):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    mount = tmp_path / "controller"
    mount.mkdir()
    kind = "cgroup2" if version == 2 else "cgroup"
    controllers = "" if version == 2 else "memory"
    (proc / "self/cgroup").write_text(
        f"{0 if version == 2 else 5}:{controllers}:{membership}\n"
    )
    options = "rw" if version == 2 else "rw,memory"
    (proc / "self/mountinfo").write_text(
        f"29 23 0:26 {mount_root} {mount} rw - {kind} cgroup {options}\n"
    )
    relative = (
        membership.removeprefix(mount_root).lstrip("/")
        if mount_root != "/"
        else membership.lstrip("/")
    )
    leaf = mount / relative
    leaf.mkdir(parents=True, exist_ok=True)
    return proc, mount, leaf


def set_memory(directory, limit, usage, version=2):
    directory.mkdir(parents=True, exist_ok=True)
    names = (
        ("memory.max", "memory.current")
        if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    for name, value in zip(names, (limit, usage), strict=True):
        (directory / name).write_text(str(value))


@pytest.mark.parametrize("version", [1, 2])
def test_hard_limit_minus_usage_and_ancestor_limit(tmp_path, version):
    proc, _mount, leaf = cgroup(tmp_path, version)
    set_memory(leaf, 8 * GIB, 2 * GIB, version)
    set_memory(leaf.parent, 4 * GIB, 3 * GIB, version)
    available, constraints = effective_memory_available(
        100 * GIB, environ={}, proc_root=proc
    )
    assert available == GIB
    assert len(constraints) == 3


@pytest.mark.parametrize("limit", ["max", "not-an-int", "-1"])
def test_missing_or_invalid_controller_limits_fall_back(tmp_path, limit):
    proc, _, leaf = cgroup(tmp_path)
    set_memory(leaf, limit, 100)
    assert effective_memory_available(2 * GIB, environ={}, proc_root=proc)[0] == 2 * GIB


def test_exhausted_controller_has_no_headroom(tmp_path):
    proc, _, leaf = cgroup(tmp_path)
    set_memory(leaf, GIB, GIB + 1)
    assert effective_memory_available(2 * GIB, environ={}, proc_root=proc)[0] == 0


@pytest.mark.parametrize(
    "root,membership", [("/job", "/job/step"), ("/namespace", "/")]
)
def test_bind_mount_and_cgroup_namespace_roots(tmp_path, root, membership):
    proc, _mount, leaf = cgroup(tmp_path, membership=membership, mount_root=root)
    set_memory(leaf, 4 * GIB, GIB)
    assert (
        effective_memory_available(20 * GIB, environ={}, proc_root=proc)[0] == 3 * GIB
    )


def test_host_can_be_stricter_than_controller(tmp_path):
    proc, _, leaf = cgroup(tmp_path)
    set_memory(leaf, 4 * GIB, GIB)
    assert effective_memory_available(GIB, environ={}, proc_root=proc)[0] == GIB


@pytest.mark.parametrize(
    "env,limit",
    [
        ({"SLURM_MEM_PER_NODE": "1024"}, GIB),
        ({"SLURM_MEM_PER_CPU": "1024", "SLURM_CPUS_PER_TASK": "2"}, 2 * GIB),
        (
            {
                "SLURM_MEM_PER_CPU": "1024",
                "SLURM_CPUS_ON_NODE": "4",
                "SLURM_CPUS_PER_TASK": "2",
            },
            4 * GIB,
        ),
        (
            {
                "SLURM_MEM_PER_NODE": "1024",
                "SLURM_MEM_PER_CPU": "1024",
                "SLURM_CPUS_ON_NODE": "4",
            },
            GIB,
        ),
    ],
)
def test_slurm_limits_subtract_current_process_rss(tmp_path, env, limit):
    available, _ = effective_memory_available(
        100 * GIB, process_rss=GIB // 4, environ=env, proc_root=tmp_path
    )
    assert available == limit - GIB // 4


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"SLURM_MEM_PER_NODE": "bad"},
        {"SLURM_MEM_PER_NODE": "0"},
        {"SLURM_MEM_PER_NODE": "-4"},
        {"SLURM_MEM_PER_CPU": "1024"},
    ],
)
def test_unknown_slurm_allocation_is_not_guessed(tmp_path, env):
    assert (
        effective_memory_available(3 * GIB, environ=env, proc_root=tmp_path)[0]
        == 3 * GIB
    )


def test_planner_reports_effective_constraints(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=512 * GIB)
    )
    monkeypatch.setattr(
        memory,
        "effective_memory_available",
        lambda *a, **kw: (GIB, {"test_cgroup": GIB}),
    )
    plan = memory.choose_batch_size(
        n_samples=20000, sequence_length=49152, mode="haplotypes", dtype="float32"
    )
    assert plan.batch_size < 20000
    assert plan.available_memory_bytes == GIB
    assert plan.memory_constraints == {"test_cgroup": GIB}
    assert plan.estimated_peak_batch_bytes <= plan.memory_budget_bytes
