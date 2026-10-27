# dsbf/engine/cache.py
"""Keep the results of completed checks, so an unchanged check is not run again."""

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dsbf.data import Dataset
from dsbf.engine.artifacts import decode, encode
from dsbf.engine.plan import PlanEntry
from dsbf.engine.reference import Reference
from dsbf.engine.tips import library_digest
from dsbf.model import ArtifactRef, CheckResult, CheckStatus, DatasetProfile
from dsbf.model.artifact import payload_extension
from dsbf.paths import dsbf_home

logger = logging.getLogger(__name__)

# Bump when what goes into a key changes, so older entries stop matching.
KEY_SCHEME = 2
_ENTRY_FORMAT = 1
# Only these results are a function of the inputs. An error, timeout, skip or missing
# extra depends on the machine, the clock and the limits; a "not applicable" result is
# decided from the profile without calling the check, so recomputing it costs nothing.
_STORED = (CheckStatus.OK,)
_TEMP_GRACE = 3600.0  # seconds before a leftover ``.tmp`` file counts as an entry


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """What a result was computed from, and the digest of it that names the entry."""

    check_id: str
    digest: str
    components: Mapping[str, Any]  # kept with the entry, so a person can see why


class ResultCache(Protocol):
    """Where results are kept between runs; a cache that fails has missed."""

    def get(self, key: CacheKey) -> CheckResult | None:
        """The result stored under ``key``, or None."""

    def put(self, key: CacheKey, result: CheckResult) -> None:
        """Store ``result`` under ``key``."""


@dataclass(frozen=True)
class CachePruned:
    """What removing entries from a ``DiskCache`` took away and what is left."""

    removed: int
    freed_bytes: int
    remaining: int
    remaining_bytes: int


class DiskCache:
    """One JSON file per result, ``<root>/<check id>/<digest>.json``, and one file
    beside it per artifact, ``<digest>.<name>.parquet`` or ``.json``.

    A file is written whole and renamed into place, so a crash or a second process
    writing the same entry never leaves a half-written one. A file that cannot be
    read, is of another format or holds another key counts as absent.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def default(cls) -> "DiskCache":
        """The cache under the DSBF home, which is read when this is called."""
        return cls(dsbf_home() / "cache" / "results")

    def path(self, key: CacheKey) -> Path:
        return self.root / key.check_id / f"{key.digest}.json"

    def _artifact_path(self, key: CacheKey, artifact: ArtifactRef) -> Path:
        name = artifact.ref.partition("/")[2]
        extension = payload_extension(artifact.media_type)
        return self.path(key).with_name(f"{key.digest}.{name}.{extension}")

    def get(self, key: CacheKey) -> CheckResult | None:
        try:
            text = self.path(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            entry = json.loads(text)
            if entry["format"] != _ENTRY_FORMAT or entry["digest"] != key.digest:
                return None
            result = CheckResult.model_validate(entry["result"])
        except (ValueError, KeyError, TypeError) as error:  # incl. ValidationError
            logger.debug("ignoring cache entry %s: %s", self.path(key), error)
            return None
        payloads = {}
        try:
            for artifact in result.artifacts:
                path = self._artifact_path(key, artifact)
                payloads[artifact.ref] = decode(path.read_bytes(), artifact.media_type)
                os.utime(path)
        except (OSError, ValueError) as error:  # a missing payload is a miss
            logger.debug("ignoring cache entry %s: %s", self.path(key), error)
            return None
        try:
            os.utime(self.path(key))  # a hit is a use: eviction removes the unused
        except OSError:
            pass
        return result.with_payloads(payloads)

    def put(self, key: CacheKey, result: CheckResult) -> None:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        for artifact in result.artifacts:  # before the entry, which names them
            self._write(
                self._artifact_path(key, artifact),
                encode(result.payloads[artifact.ref])[0],
            )
        text = json.dumps(
            {
                "format": _ENTRY_FORMAT,
                "digest": key.digest,
                "components": key.components,
                "result": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        self._write(target, text.encode("utf-8"))

    @staticmethod
    def _write(target: Path, data: bytes) -> None:
        handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(data)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def clear(self) -> CachePruned:
        """Remove every entry, and the folders they leave empty."""
        return self.prune(max_bytes=0)

    def prune(
        self, *, max_bytes: int | None = None, max_age_seconds: float | None = None
    ) -> CachePruned:
        """Remove entries older than ``max_age_seconds``, then the least recently used
        until at most ``max_bytes`` remain; ``None`` leaves that limit off.

        An entry that cannot be removed stays and counts as remaining.
        """
        entries = []
        for path in self.root.glob("*/*"):
            try:
                info = path.stat()
            except OSError:
                continue
            if path.suffix == ".tmp" and time.time() - info.st_mtime < _TEMP_GRACE:
                continue  # another process is writing it
            entries.append((info.st_mtime, info.st_size, path))
        entries.sort(key=lambda item: item[0])
        total = sum(size for _, size, _ in entries)
        now = time.time()
        removed = freed = 0
        for modified, size, path in entries:
            too_old = max_age_seconds is not None and now - modified > max_age_seconds
            too_big = max_bytes is not None and total > max_bytes
            if not (too_old or too_big):
                continue
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            freed += size
            removed += 1
        for folder in self.root.glob("*"):
            try:
                folder.rmdir()  # only if empty
            except OSError:
                pass
        return CachePruned(removed, freed, len(entries) - removed, total)


def result_digest(result: CheckResult) -> str:
    """A digest of everything in ``result`` but its timing, which differs run to run."""
    return _digest(result.model_dump(mode="json", exclude={"timing"}))


class CacheSession:
    """The cache as one run uses it: keys, lookups and stores, none of which can fail.

    A key covers the data (its fingerprint and sampling), the profile, the check (id,
    version, parameters) and the digest of every upstream result the check may read.
    Because an upstream result is in the key, changing one check's parameters changes
    the keys of its dependents, and only of those whose upstream result changed.
    A cache that raises is logged once and treated as missing from then on.
    """

    def __init__(
        self,
        cache: ResultCache,
        dataset: Dataset,
        profile: DatasetProfile,
        reference: Reference | None = None,
    ) -> None:
        self._cache: ResultCache | None = cache
        self._scope = {
            "scheme": KEY_SCHEME,
            "fingerprint": dataset.fingerprint,
            "sampling": (
                dataset.sampling.model_dump(mode="json") if dataset.sampling else None
            ),
            "profile": _digest(profile.model_dump(mode="json")),
            "tips": library_digest(),
        }
        self._reference = (
            None
            if reference is None
            else _digest(
                {
                    "profile": reference.profile.model_dump(mode="json"),
                    "n_rows": reference.n_rows,
                    "fingerprint": reference.fingerprint,
                    "sampling": (
                        reference.sampling.model_dump(mode="json")
                        if reference.sampling
                        else None
                    ),
                }
            )
        )
        self._keys: dict[str, CacheKey] = {}
        self._digests: dict[str, str] = {}
        self.failure: str | None = None

    def lookup(
        self, entry: PlanEntry, results: Mapping[str, CheckResult]
    ) -> CheckResult | None:
        """The stored result for ``entry`` given the results before it, if any.

        ``results`` must hold every upstream result of ``entry`` that exists.
        """
        key = self._key(entry, results)
        self._keys[entry.id] = key
        return self._guarded(lambda cache: cache.get(key))

    def store(self, entry: PlanEntry, result: CheckResult) -> None:
        """Keep ``result`` of ``entry``, which ``lookup`` missed, if it is reusable."""
        key = self._keys.get(entry.id)
        if key is None or result.status not in _STORED:
            return
        self._guarded(lambda cache: cache.put(key, result))

    def _key(self, entry: PlanEntry, results: Mapping[str, CheckResult]) -> CacheKey:
        upstream = []
        for name in entry.check.spec.upstream:
            found = results.get(name)
            upstream.append([name, None if found is None else self._digest_of(found)])
        components = {
            **self._scope,
            "check": entry.id,
            "version": entry.check.version,
            "params": entry.params.model_dump(mode="json"),
            "upstream": upstream,
        }
        if entry.check.spec.requires.reference:
            components["reference"] = self._reference
        return CacheKey(entry.id, _digest(components), components)

    def _digest_of(self, result: CheckResult) -> str:
        digest = self._digests.get(result.check_id)
        if digest is None:
            digest = self._digests[result.check_id] = result_digest(result)
        return digest

    def _guarded[T](self, use: Callable[[ResultCache], T]) -> T | None:
        if self._cache is None:
            return None
        try:
            return use(self._cache)
        except Exception as error:
            logger.warning(
                "the result cache failed (%s: %s); this run continues without it",
                type(error).__name__,
                error,
            )
            self.failure = f"{type(error).__name__}: {error}"
            self._cache = None
            return None
