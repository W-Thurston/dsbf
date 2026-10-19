# dsbf/engine/cache.py
"""Keep the results of completed checks, so an unchanged check is not run again."""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dsbf.data import Dataset
from dsbf.engine.plan import PlanEntry
from dsbf.model import CheckResult, CheckStatus, DatasetProfile
from dsbf.paths import dsbf_home

logger = logging.getLogger(__name__)

# Bump when what goes into a key changes, so older entries stop matching.
KEY_SCHEME = 1
_ENTRY_FORMAT = 1
# Only these results are a function of the inputs. An error, timeout, skip or missing
# extra depends on the machine, the clock and the limits; a "not applicable" result is
# decided from the profile without calling the check, so recomputing it costs nothing.
_STORED = (CheckStatus.OK,)


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


class DiskCache:
    """One JSON file per result, ``<root>/<check id>/<digest>.json``.

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

    def get(self, key: CacheKey) -> CheckResult | None:
        try:
            text = self.path(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            entry = json.loads(text)
            if entry["format"] != _ENTRY_FORMAT or entry["digest"] != key.digest:
                return None
            return CheckResult.model_validate(entry["result"])
        except (ValueError, KeyError, TypeError) as error:  # incl. ValidationError
            logger.debug("ignoring cache entry %s: %s", self.path(key), error)
            return None

    def put(self, key: CacheKey, result: CheckResult) -> None:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            {
                "format": _ENTRY_FORMAT,
                "digest": key.digest,
                "components": key.components,
                "result": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(text)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


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
        self, cache: ResultCache, dataset: Dataset, profile: DatasetProfile
    ) -> None:
        self._cache: ResultCache | None = cache
        self._scope = {
            "scheme": KEY_SCHEME,
            "fingerprint": dataset.fingerprint,
            "sampling": (
                dataset.sampling.model_dump(mode="json") if dataset.sampling else None
            ),
            "profile": _digest(profile.model_dump(mode="json")),
        }
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
