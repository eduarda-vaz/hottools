"""
Write encoded batches to disk, as one transaction per run.

A run's outputs are planned in full before anything is written: every
data file, the metadata sidecar, and - for a per-sample directory - a
manifest. Batches are written into a staging directory beside the
output and renamed into place only once the last batch has succeeded,
so a run that fails part-way, or is refused under ``overwrite='error'``,
leaves any earlier output exactly as it was.

Under ``overwrite='replace'`` an earlier run's files that the new run
does not rewrite are removed after the commit. Only files hottools can
show are its own are ever removed: names recorded in an earlier sidecar
or manifest, and the batch files its own naming scheme produces.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config_validation import positive_integer
from .constants import OVERWRITE_POLICIES, OverwritePolicy, check_choice

__all__ = [
    "OutputPlan",
    "SampleNameCollisionWarning",
    "check_destination",
    "make_writer",
    "metadata_path_for",
    "plan_outputs",
    "previous_run_outputs",
    "safe_sample_names",
    "write_metadata_json",
]


class SampleNameCollisionWarning(RuntimeWarning):
    """Two sample IDs would share a filename and one was renamed."""


#: Target size of one HDF5 chunk. Chunks are the unit of HDF5 I/O and
#: compression; multi-hundred-MiB chunks defeat the raw chunk cache and
#: force a whole-chunk read for a single-sample slice.
DEFAULT_H5_CHUNK_BYTES = 4 * 1024**2

#: Raw chunk cache, as a multiple of the chunk size.
H5_CACHE_CHUNKS = 8

SUFFIXES = {"npy": ".npy", "npz": ".npz", "h5": ".h5", "pt": ".pt"}
DATA_SUFFIXES = frozenset(SUFFIXES.values())

#: Sidecar name inside a per-sample output directory.
PER_SAMPLE_METADATA_NAME = "hottools.metadata.json"
#: Hidden record of the files a per-sample run wrote. Written even with
#: the metadata sidecar disabled, because it is what lets a later
#: ``overwrite='replace'`` find and remove files the new run does not
#: rewrite, without guessing at files it never wrote.
MANIFEST_NAME = ".hottools-manifest.json"

#: Leaves room for a numeric suffix and the extension inside the
#: 255-byte filename limit of common filesystems.
MAX_NAME_CHARS = 200

_WINDOWS_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)

_BATCH_RE = r"\.batch\d{6}-\d{6}"


# ------------------------------------------------------------------
# per-sample filenames
# ------------------------------------------------------------------


def _safe_sample_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    # A leading dot would make a hidden file, and "." or ".." would name
    # a directory rather than a sample.
    cleaned = re.sub(r"^\.+", lambda m: "_" * len(m.group()), cleaned)
    cleaned = cleaned[:MAX_NAME_CHARS] or "sample"
    if cleaned.split(".", 1)[0].lower() in _WINDOWS_RESERVED:
        cleaned = "_" + cleaned
    return cleaned


def safe_sample_names(samples: list[str]) -> list[str]:
    """
    Filename-safe names for per-sample output, unique on every filesystem.

    Sanitising unsupported characters to ``_`` can map distinct sample
    IDs onto the same filename - ``S/2`` and ``S:2`` both become
    ``S_2`` - and on a case-insensitive filesystem (the macOS and
    Windows defaults) ``S1`` and ``s1`` are the same file too. Either
    would silently overwrite one sample's data with another's.

    Uniqueness is therefore checked case-insensitively, in three passes.
    IDs that are already valid filenames claim their names first, then
    sanitised IDs claim theirs if still free - so an ID that needs no
    change is never renamed to make room for one that did. Each sample
    still without a name takes ``<name>-<n>`` with the smallest ``n``
    that no name, natural or generated, already uses. Within a pass,
    input order decides. Renames are reported; the sample list in the
    metadata stays the authoritative mapping from row to ID.
    """
    natural = [_safe_sample_name(name) for name in samples]
    out: list[str | None] = [None] * len(samples)
    used: set[str] = set()
    for unchanged in (True, False):
        for i, name in enumerate(natural):
            if out[i] is not None or (name == samples[i]) != unchanged:
                continue
            key = name.casefold()
            if key not in used:
                used.add(key)
                out[i] = name

    renamed: list[tuple[str, str]] = []
    for i, name in enumerate(natural):
        if out[i] is not None:
            continue
        n = 2
        while True:
            suffix = f"-{n}"
            candidate = name[: MAX_NAME_CHARS - len(suffix)] + suffix
            if candidate.casefold() not in used:
                break
            n += 1
        used.add(candidate.casefold())
        out[i] = candidate
        renamed.append((samples[i], candidate))

    if renamed:
        shown = ", ".join(f"{orig!r} -> {new}" for orig, new in renamed[:5])
        if len(renamed) > 5:
            shown += f", ... ({len(renamed)} total)"
        warnings.warn(
            "Sample ID(s) would share a filename with another sample "
            "(after replacing unsupported characters, or differing only in "
            f"case) and were given a numeric suffix: {shown}. The sample "
            "list in the metadata remains authoritative.",
            SampleNameCollisionWarning,
            stacklevel=3,
        )
    return [name for name in out if name is not None]


# ------------------------------------------------------------------
# output planning
# ------------------------------------------------------------------


def check_destination(path: Path, overwrite: OverwritePolicy) -> Path:
    """Refuse to clobber an existing file unless told otherwise."""
    check_choice("overwrite", overwrite, OVERWRITE_POLICIES)
    if overwrite == "error" and Path(path).exists():
        raise FileExistsError(
            f"{path} already exists; pass --overwrite replace to replace it."
        )
    return Path(path)


def _suffix(fmt: str) -> str:
    return SUFFIXES[fmt]


def metadata_path_for(
    output_path: str | Path,
    per_sample: bool,
    batched: bool = False,
) -> Path:
    """
    Where the sidecar JSON goes.

    ``<output>.metadata.json`` next to a single output file,
    ``<dir>/hottools.metadata.json`` inside a per-sample directory, and
    ``<stem>.metadata.json`` when the requested output file was split
    into per-batch files and therefore does not itself exist.
    """
    out = Path(output_path)
    if per_sample:
        return out / PER_SAMPLE_METADATA_NAME
    if batched:
        return out.with_suffix(".metadata.json")
    return out.with_suffix(out.suffix + ".metadata.json")


def batch_path_for(output_path: Path, fmt: str, start: int, end: int) -> Path:
    """Name of the file holding samples ``[start, end)`` of a split run."""
    stem = Path(output_path).with_suffix("")
    return Path(f"{stem}.batch{start:06d}-{end - 1:06d}{_suffix(fmt)}")


def _is_batched(
    fmt: str, per_sample: bool, n_samples: int, batch_size: int | None
) -> bool:
    return (
        not per_sample
        and fmt in {"npz", "pt"}
        and batch_size is not None
        and batch_size < n_samples
    )


@dataclass(slots=True)
class OutputPlan:
    """Every file one run will write, and what an earlier run left."""

    output_path: Path
    fmt: str
    per_sample: bool
    #: Data files in the order they will be written.
    data_files: list[Path]
    metadata_path: Path | None
    manifest_path: Path | None
    #: Files an earlier run wrote into this output, found from its
    #: sidecar or manifest and from hottools' own batch-file names.
    previous: list[Path] = field(default_factory=list)

    @property
    def final_paths(self) -> list[Path]:
        extra = [p for p in (self.metadata_path, self.manifest_path) if p]
        return [*self.data_files, *extra]

    @property
    def conflicts(self) -> list[Path]:
        """Existing files this run would replace or leave inconsistent."""
        found = {str(p): p for p in self.previous}
        for path in self.final_paths:
            if path.exists():
                found.setdefault(str(path), path)
        return sorted(found.values(), key=str)

    @property
    def stale(self) -> list[Path]:
        """Earlier files the new run will not rewrite.

        Compared case-insensitively: on a case-insensitive filesystem an
        earlier ``S1.npy`` *is* the new ``s1.npy``, and removing it as
        stale would remove the new data.
        """
        keep = {(str(p.parent), p.name.casefold()) for p in self.final_paths}
        return [
            p for p in self.previous if (str(p.parent), p.name.casefold()) not in keep
        ]


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_sidecar(data: dict | None) -> bool:
    return bool(data) and "output_files" in data and "mode" in data


def _recorded_names(data: dict | None, key: str) -> list[str]:
    """Plain data-file basenames listed under ``key``.

    Only a bare filename with a data suffix is accepted. A recorded path
    is never followed elsewhere: the earlier run may have been started
    from another directory, and a sidecar is not trusted to point
    outside its own output.
    """
    if not data:
        return []
    names = []
    for entry in data.get(key) or []:
        name = Path(str(entry)).name
        if name and Path(name).suffix in DATA_SUFFIXES:
            names.append(name)
    return names


def previous_run_outputs(
    output_path: str | Path, fmt: str, per_sample: bool
) -> list[Path]:
    """
    Files an earlier hottools run left in this output location.

    Needs neither the sample list nor the batch size, so it can run
    before any VCF work: under ``overwrite='error'`` a non-empty answer
    already decides that the run must stop.
    """
    out = Path(output_path)
    fmt = fmt.lower()
    found: dict[str, Path] = {}

    def add(path: Path) -> None:
        if path.is_file():
            found.setdefault(str(path), path)

    if per_sample:
        if out.exists() and not out.is_dir():
            raise NotADirectoryError(
                f"{out} exists and is not a directory; per-sample output "
                "needs a directory"
            )
        if not out.is_dir():
            return []
        manifest = _read_json(out / MANIFEST_NAME)
        sidecar = _read_json(out / PER_SAMPLE_METADATA_NAME)
        for name in _recorded_names(manifest, "files"):
            add(out / name)
        if _is_sidecar(sidecar):
            for name in _recorded_names(sidecar, "output_files"):
                add(out / name)
            add(out / PER_SAMPLE_METADATA_NAME)
        if manifest is not None:
            add(out / MANIFEST_NAME)
        return sorted(found.values(), key=str)

    add(out)
    single = metadata_path_for(out, per_sample=False, batched=False)
    batched = metadata_path_for(out, per_sample=False, batched=True)
    sidecars = []
    data = _read_json(single)
    if _is_sidecar(data):
        sidecars.append((single, data))
    data = _read_json(batched)
    # <stem>.metadata.json is shared by every output with this stem, so
    # it belongs to this output only if it describes the same format.
    if (
        _is_sidecar(data)
        and not data.get("per_sample")
        and data.get("output_format") == fmt
    ):
        sidecars.append((batched, data))
    for path, data in sidecars:
        add(path)
        for name in _recorded_names(data, "output_files"):
            add(out.parent / name)
    if fmt in {"npz", "pt"} and out.parent.is_dir():
        pattern = re.compile(
            re.escape(out.with_suffix("").name)
            + _BATCH_RE
            + re.escape(_suffix(fmt))
            + "$"
        )
        for path in out.parent.iterdir():
            if pattern.match(path.name):
                add(path)
    return sorted(found.values(), key=str)


def plan_outputs(
    *,
    output_path: str | Path,
    fmt: str,
    per_sample: bool,
    samples: list[str],
    batch_size: int | None,
    metadata_json: bool = True,
    safe_names: list[str] | None = None,
) -> OutputPlan:
    """The complete set of files a run will write, before it writes any."""
    out = Path(output_path)
    fmt = fmt.lower()
    if fmt not in SUFFIXES:
        raise ValueError("format must be one of: npy, npz, h5, pt")
    n_samples = len(samples)
    if per_sample:
        names = (
            safe_names if safe_names is not None else safe_sample_names(list(samples))
        )
        data_files = [out / f"{name}{_suffix(fmt)}" for name in names]
        manifest = out / MANIFEST_NAME
        batched = False
    else:
        batched = _is_batched(fmt, per_sample, n_samples, batch_size)
        if batched:
            data_files = [
                batch_path_for(out, fmt, s, min(s + batch_size, n_samples))
                for s in range(0, n_samples, batch_size)
            ]
        else:
            data_files = [out]
        manifest = None
    meta = (
        metadata_path_for(out, per_sample=per_sample, batched=batched)
        if metadata_json
        else None
    )
    return OutputPlan(
        output_path=out,
        fmt=fmt,
        per_sample=per_sample,
        data_files=data_files,
        metadata_path=meta,
        manifest_path=manifest,
        previous=previous_run_outputs(out, fmt, per_sample),
    )


def check_output_plan(plan: OutputPlan, overwrite: OverwritePolicy) -> None:
    """Under ``overwrite='error'``, refuse before writing anything."""
    check_choice("overwrite", overwrite, OVERWRITE_POLICIES)
    if overwrite != "error":
        return
    conflicts = plan.conflicts
    if conflicts:
        shown = ", ".join(str(p) for p in conflicts[:5])
        more = f" and {len(conflicts) - 5} more" if len(conflicts) > 5 else ""
        raise FileExistsError(
            f"Output already exists: {shown}{more}. Nothing was written; "
            "pass --overwrite replace to replace the earlier run, or "
            "choose another output path."
        )


@contextlib.contextmanager
def _publication_lock(plan: OutputPlan):
    """Fail fast if another process is publishing this output namespace.

    Atomic directory creation works across processes/nodes without a persistent
    lock inode. Never reap an existing lock: its owner may run on another node.
    """
    # Replacing a file symlink changes resolve() mid-commit. Resolve only
    # its parent; directory outputs instead lock their resolved directory.
    target = (plan.output_path.resolve() if plan.per_sample else
              plan.output_path.parent.resolve() / plan.output_path.name)
    namespace = str(target.with_suffix("")).casefold()
    digest = hashlib.sha256(namespace.encode()).hexdigest()[:24]
    lock = target.parent / f".hottools-{digest}.publish-lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Another writer may be publishing {plan.output_path}; lock: {lock}. "
            "After a hard kill, remove this directory only after confirming "
            "no writer still owns this output."
        ) from exc
    try:
        yield
    finally:
        lock.rmdir()


class _Staging:
    """A directory beside the output that holds a run until it commits."""

    def __init__(self, plan: OutputPlan) -> None:
        self.plan = plan
        anchor = plan.output_path.parent
        anchor.mkdir(parents=True, exist_ok=True)
        # Same parent directory, so the final rename never crosses a
        # filesystem boundary and each file appears atomically.
        self.dir = Path(
            tempfile.mkdtemp(
                prefix=f".{plan.output_path.name}.",
                suffix=".hottools-partial",
                dir=anchor,
            )
        )
        self.pairs: list[tuple[Path, Path]] = []
        self.done = False
        self.preserve_recovery = False

    def path_for(self, final: Path) -> Path:
        staged = self.dir / Path(final).name
        self.pairs.append((staged, Path(final)))
        return staged

    def commit(self, remove: list[Path], overwrite: OverwritePolicy) -> None:
        """Publish files, rolling back ordinary commit failures.

        Backups stay on the same filesystem. Stale files are moved into
        the backup directory rather than deleted until publication succeeds.
        A failed rollback preserves recovery files instead of deleting them.
        This is exception-safe publication, not a multi-file atomic rename.
        """
        backup_dir = self.dir / ".rollback"
        backup_dir.mkdir()
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        retired: list[tuple[Path, Path]] = []
        created_output = False
        try:
            for index, (_, final) in enumerate(self.pairs):
                if final.exists() or final.is_symlink():
                    backup = backup_dir / f"original-{index}"
                    # Hard links preserve the original inode without a third
                    # full data copy; fall back on filesystems without them.
                    try:
                        os.link(final, backup, follow_symlinks=False)
                    except OSError:
                        shutil.copy2(final, backup, follow_symlinks=False)
                    backups[final] = backup
            # Keep an on-disk filename map for manual recovery if a later
            # restore fails or the process terminates during publication.
            (backup_dir / "journal.json").write_text(
                json.dumps(
                    {
                        "originals": [
                            {"final": str(final), "backup": backup.name}
                            for final, backup in backups.items()
                        ],
                        "new_files": [
                            str(final)
                            for _, final in self.pairs
                            if final not in backups
                        ],
                        "stale": [
                            {"final": str(path), "backup": f"stale-{index}"}
                            for index, path in enumerate(remove)
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if self.plan.per_sample and not self.plan.output_path.exists():
                self.plan.output_path.mkdir(parents=True)
                created_output = True
            for staged, final in self.pairs:
                if overwrite == "error":
                    # link() publishes the complete inode atomically and fails
                    # with EEXIST even if a non-Hottools writer races the lock.
                    os.link(staged, final)
                    installed.append(final)
                    staged.unlink()
                else:
                    os.replace(staged, final)
                    installed.append(final)
            for index, path in enumerate(remove):
                if path.exists() or path.is_symlink():
                    backup = backup_dir / f"stale-{index}"
                    os.replace(path, backup)
                    retired.append((path, backup))
        except BaseException as exc:
            failures = []
            for final, backup in reversed(retired):
                try:
                    os.replace(backup, final)
                except OSError as error:
                    failures.append(error)
            for final in reversed(installed):
                try:
                    if final in backups:
                        os.replace(backups[final], final)
                    else:
                        final.unlink(missing_ok=True)
                except OSError as error:
                    failures.append(error)
            if created_output:
                try:
                    self.plan.output_path.rmdir()
                except OSError as error:
                    failures.append(error)
            if failures:
                self.preserve_recovery = True
                raise RuntimeError(
                    f"Commit failed and rollback could not restore all files; "
                    f"recovery files preserved at {self.dir}: {failures[0]}"
                ) from exc
            self.abort()
            raise
        shutil.rmtree(self.dir, ignore_errors=True)
        self.done = True

    def abort(self) -> None:
        if not self.preserve_recovery:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.done = True


def auto_h5_chunks(
    full_shape: tuple[int, ...],
    itemsize: int,
    sequence_axis: int,
    target_bytes: int = DEFAULT_H5_CHUNK_BYTES,
    chunk_samples: int | None = None,
    chunk_length: int | None = None,
) -> tuple[int, ...]:
    """
    Pick an HDF5 chunk shape of roughly ``target_bytes``.

    Chunking only along the sample axis produces chunks the size of a
    whole sample tensor, which for a 500 kb window is hundreds of MiB.
    The sequence axis is split as well so that chunks stay cache-sized,
    while keeping whole samples together when they already fit.
    """
    positive_integer("chunk_samples", chunk_samples, optional=True)
    positive_integer("chunk_length", chunk_length, optional=True)
    if sequence_axis <= 0 or sequence_axis >= len(full_shape):
        raise ValueError("sequence_axis must index a non-sample axis")

    n_samples = full_shape[0]
    length = full_shape[sequence_axis]
    # Bytes for one sample, one base: every axis except sample and
    # sequence (haplotype and channel axes).
    other = 1
    for axis, size in enumerate(full_shape):
        if axis not in (0, sequence_axis):
            other *= size
    base_bytes = max(1, other * itemsize)

    if chunk_samples is not None and chunk_length is not None:
        s_chunk = max(1, min(n_samples, int(chunk_samples)))
        l_chunk = max(1, min(length, int(chunk_length)))
    else:
        row_bytes = length * base_bytes
        if row_bytes <= target_bytes:
            l_chunk = length
            s_chunk = max(1, min(n_samples, target_bytes // max(1, row_bytes)))
        else:
            s_chunk = 1
            l_chunk = max(1, min(length, target_bytes // base_bytes))
        if chunk_samples is not None:
            s_chunk = max(1, min(n_samples, int(chunk_samples)))
        if chunk_length is not None:
            l_chunk = max(1, min(length, int(chunk_length)))

    chunks = list(full_shape)
    chunks[0] = s_chunk
    chunks[sequence_axis] = l_chunk
    return tuple(chunks)


class BatchWriter(ABC):
    """
    Base writer. Data goes to a staging directory; :meth:`commit` moves it
    into place. Leaving the ``with`` block on an exception discards the
    staging directory, so earlier output survives a failed run intact.
    """

    def __init__(
        self,
        *,
        output_path: str | Path,
        fmt: str,
        full_shape: tuple[int, ...],
        dtype: np.dtype,
        samples: list[str],
        per_sample: bool,
        metadata: dict,
        sequence_axis: int = 1,
        h5_compression: str | None = None,
        h5_chunk_samples: int | None = None,
        h5_chunk_length: int | None = None,
        overwrite: OverwritePolicy = "error",
        batch_size: int | None = None,
        metadata_json: bool = False,
        plan: OutputPlan | None = None,
    ) -> None:
        check_choice("overwrite", overwrite, OVERWRITE_POLICIES)
        self.output_path = Path(output_path)
        self.fmt = fmt
        self.full_shape = full_shape
        self.dtype = np.dtype(dtype)
        self.samples = list(samples)
        self.per_sample = per_sample
        self.metadata = metadata
        self.sequence_axis = sequence_axis
        self.h5_compression = h5_compression
        self.h5_chunk_samples = h5_chunk_samples
        self.h5_chunk_length = h5_chunk_length
        self.overwrite = overwrite
        #: Metadata entries HDF5 can store as attributes, filtered once
        #: rather than re-scanned for every per-sample file.
        self.scalar_metadata = {
            k: v
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool, np.number))
        }
        self.plan = plan or plan_outputs(
            output_path=self.output_path,
            fmt=fmt,
            per_sample=per_sample,
            samples=self.samples,
            batch_size=batch_size,
            metadata_json=metadata_json,
        )
        # Refused here, before the staging directory exists, so a
        # conflicting run creates nothing at all.
        check_output_plan(self.plan, overwrite)
        self.safe_names = (
            [p.with_suffix("").name for p in self.plan.data_files] if per_sample else []
        )
        #: Final paths of every data file written so far, in order.
        self.written_paths: list[Path] = []
        #: Earlier files removed by the commit (``overwrite='replace'``).
        self.removed_paths: list[Path] = []
        self._staging = _Staging(self.plan)

    def _stage(self, final: Path) -> Path:
        """Staging path for ``final``; ``final`` is recorded as written.

        Every file must be the next one the plan expects. A batch that
        disagrees with the planned batch size is refused here, at the
        first write, rather than discovered at commit.
        """
        index = len(self.written_paths)
        expected = self.plan.data_files
        if index >= len(expected) or str(expected[index]) != str(final):
            want = expected[index] if index < len(expected) else "nothing"
            raise ValueError(
                f"write produced {final}, but the output plan expects "
                f"{want} next; pass make_writer the batch_size used for "
                "write() calls"
            )
        self.written_paths.append(Path(final))
        return self._staging.path_for(final)

    @abstractmethod
    def write(self, batch: np.ndarray, start: int, end: int) -> None: ...

    def close(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release file handles. Backends without any are free to skip this."""

    def commit(self, metadata: dict | None = None) -> list[Path]:
        """
        Move the staged run into place; return the earlier files removed.

        ``metadata`` is written to the planned sidecar path as part of the
        same commit, so the sidecar never describes a run that did not
        finish.
        """
        if self._staging.done:
            raise RuntimeError("this writer has already committed or aborted")
        self.close()
        planned = [str(p) for p in self.plan.data_files]
        written = [str(p) for p in self.written_paths]
        if written != planned:
            self.abort()
            raise RuntimeError(
                "the batches written do not match the output plan: "
                f"planned {len(planned)} file(s), wrote {len(written)}"
            )
        with _publication_lock(self.plan):
            self.plan.previous = previous_run_outputs(
                self.plan.output_path, self.plan.fmt, self.plan.per_sample
            )
            check_output_plan(self.plan, self.overwrite)
            remove = self.plan.stale if self.overwrite == "replace" else []
            if metadata is not None and "stale_outputs_removed" in metadata:
                metadata["stale_outputs_removed"] = [str(path) for path in remove]
            if self.plan.manifest_path is not None:
                manifest = {
                    "tool": "hottools",
                    "files": [p.name for p in self.plan.data_files],
                }
                self._staging.path_for(self.plan.manifest_path).write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
            if metadata is not None and self.plan.metadata_path is not None:
                staged = self._staging.path_for(self.plan.metadata_path)
                write_metadata_json(staged, metadata, overwrite="error")
            self._staging.commit(remove, self.overwrite)
            self.removed_paths = list(remove)
            return self.removed_paths

    def abort(self) -> None:
        """Discard everything staged; earlier output is left untouched."""
        self.close()
        if not self._staging.done:
            self._staging.abort()

    @contextlib.contextmanager
    def _abort_on_error(self):
        """Discard the staging directory if opening the output fails.

        A constructor that raises never reaches a ``with`` block, so
        ``__exit__`` would not run and the staging directory would be
        left behind.
        """
        try:
            yield
        except BaseException:
            self.abort()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        elif not self._staging.done:
            try:
                self.commit()
            except BaseException:
                self.abort()
                raise


class NpyWriter(BatchWriter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._arr = None
        if not self.per_sample:
            from numpy.lib.format import open_memmap

            with self._abort_on_error():
                self._arr = open_memmap(
                    self._stage(self.plan.data_files[0]),
                    mode="w+",
                    dtype=self.dtype,
                    shape=self.full_shape,
                )

    def write(self, batch, start, end):
        if self.per_sample:
            for local_i, final in enumerate(self.plan.data_files[start:end]):
                np.save(self._stage(final), batch[local_i])
        else:
            self._arr[start:end] = batch
            # memmap.flush() msyncs the whole mapping rather than the
            # slice just written. Dropping it made no reproducible
            # difference, so it is kept for durability of long runs.
            self._arr.flush()

    def close(self):
        if self._arr is not None:
            self._arr.flush()
        self._arr = None


class H5Writer(BatchWriter):
    def __init__(self, **kwargs):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "HDF5 output requires h5py. Install hottools[hdf5]."
            ) from exc
        super().__init__(**kwargs)
        self.h5py = h5py
        self.chunks = None
        self._file = None
        self._ds = None

        if not self.per_sample:
            with self._abort_on_error():
                self._open_combined(h5py)

    def _open_combined(self, h5py) -> None:
        """Create the single HDF5 file, its datasets and attributes."""
        self.chunks = auto_h5_chunks(
            self.full_shape,
            self.dtype.itemsize,
            self.sequence_axis,
            chunk_samples=self.h5_chunk_samples,
            chunk_length=self.h5_chunk_length,
        )
        chunk_bytes = int(np.prod(self.chunks)) * self.dtype.itemsize
        self._file = h5py.File(
            self._stage(self.plan.data_files[0]),
            "w",
            rdcc_nbytes=max(1024**2, H5_CACHE_CHUNKS * chunk_bytes),
            rdcc_nslots=10007,
        )
        self._ds = self._file.create_dataset(
            "sequences",
            shape=self.full_shape,
            dtype=self.dtype,
            chunks=self.chunks,
            compression=self.h5_compression,
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._file.create_dataset(
            "samples",
            data=np.asarray(self.samples, dtype=object),
            dtype=string_dtype,
        )
        self._write_attrs(self._file)

    def _write_attrs(self, handle) -> None:
        for k, v in self.scalar_metadata.items():
            handle.attrs[k] = v

    def write(self, batch, start, end):
        if self.per_sample:
            for local_i, (sample, final) in enumerate(
                zip(
                    self.samples[start:end],
                    self.plan.data_files[start:end],
                    strict=True,
                )
            ):
                with self.h5py.File(self._stage(final), "w") as f:
                    f.create_dataset(
                        "sequence",
                        data=batch[local_i],
                        compression=self.h5_compression,
                    )
                    f.attrs["sample"] = sample
                    self._write_attrs(f)
        else:
            self._ds[start:end] = batch
            self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


class ChunkFileWriter(BatchWriter):
    """
    Writer for formats that are not naturally appendable (.npz, .pt).

    - per_sample=True: one file per sample
    - per_sample=False:
        * one batch total -> requested output file
        * multiple batches -> <stem>.batch000000-000127.<suffix>
    """

    def __init__(self, **kwargs):
        if kwargs.get("fmt") == "pt":
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                # Refused before staging, not after the first batch.
                raise RuntimeError(
                    "PyTorch output requires torch. Install hottools[torch]."
                ) from exc
        super().__init__(**kwargs)

    def _save_array(self, path: Path, arr: np.ndarray):
        if self.fmt == "npz":
            np.savez_compressed(path, sequences=arr)
        else:
            import torch

            torch.save(torch.from_numpy(np.ascontiguousarray(arr)), path)

    def write(self, batch, start, end):
        if self.per_sample:
            for local_i, final in enumerate(self.plan.data_files[start:end]):
                self._save_array(self._stage(final), batch[local_i])
            return

        if start == 0 and end == self.full_shape[0]:
            final = self.output_path
        else:
            final = batch_path_for(self.output_path, self.fmt, start, end)
        self._save_array(self._stage(final), batch)


def make_writer(
    *,
    output_path: str | Path,
    fmt: str,
    full_shape: tuple[int, ...],
    dtype: np.dtype,
    samples: list[str],
    per_sample: bool,
    metadata: dict,
    sequence_axis: int = 1,
    h5_compression: str | None = None,
    h5_chunk_samples: int | None = None,
    h5_chunk_length: int | None = None,
    overwrite: OverwritePolicy = "error",
    batch_size: int | None = None,
    metadata_json: bool = False,
    plan: OutputPlan | None = None,
) -> BatchWriter:
    """
    Open a transactional writer.

    ``batch_size`` is the number of samples per :meth:`BatchWriter.write`
    call; it fixes the names of split ``.npz``/``.pt`` files in advance,
    so the whole run can be checked against ``overwrite`` before any
    write. When omitted, each batch is assumed to cover every sample.
    """
    fmt = fmt.lower()
    if fmt not in SUFFIXES:
        raise ValueError("format must be one of: npy, npz, h5, pt")
    if batch_size is None:
        batch_size = full_shape[0]
    kwargs = {
        "output_path": output_path,
        "fmt": fmt,
        "full_shape": full_shape,
        "dtype": dtype,
        "samples": samples,
        "per_sample": per_sample,
        "metadata": metadata,
        "sequence_axis": sequence_axis,
        "h5_compression": h5_compression,
        "h5_chunk_samples": h5_chunk_samples,
        "h5_chunk_length": h5_chunk_length,
        "overwrite": overwrite,
        "batch_size": batch_size,
        "metadata_json": metadata_json,
        "plan": plan,
    }
    if fmt == "npy":
        return NpyWriter(**kwargs)
    if fmt == "h5":
        return H5Writer(**kwargs)
    return ChunkFileWriter(**kwargs)


def write_metadata_json(
    path: str | Path,
    metadata: dict,
    overwrite: OverwritePolicy = "error",
) -> Path:
    """
    Write ``metadata`` as JSON at ``path``, honouring ``overwrite``.

    The default is ``error``, the same as the command line's: a sidecar
    is never silently replaced.
    """
    out = check_destination(Path(path), overwrite)
    out.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in metadata.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, Path):
            v = str(v)
        elif isinstance(v, (list, tuple)):
            v = [str(x) if isinstance(x, Path) else x for x in v]
        serializable[k] = v

    # Exclusive creation closes the check/write race for standalone callers.
    payload = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    with out.open("x" if overwrite == "error" else "w", encoding="utf-8") as handle:
        handle.write(payload)
    return out
