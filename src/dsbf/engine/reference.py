# dsbf/engine/reference.py
"""The dataset a run is compared with: its stored profile and where it came from."""

from dataclasses import dataclass

from dsbf.data import Dataset
from dsbf.model import DatasetProfile, Run, SamplingInfo


@dataclass(frozen=True)
class Reference:
    """What a check that compares a dataset with a reference reads.

    It holds the reference's profile and provenance, never its values, so it is
    small to send to a worker and a ``Run`` makes one as readily as a table does.
    ``n_rows`` counts the rows of the source before any sampling; the profile counts
    those it describes.
    """

    name: str
    kind: str  # "data" for a table that was profiled, "run" for a stored Run
    profile: DatasetProfile
    n_rows: int
    fingerprint: str
    sampling: SamplingInfo | None = None

    @classmethod
    def of_data(cls, dataset: Dataset, profile: DatasetProfile) -> "Reference":
        """The reference made from ``dataset`` and its ``profile``.

        A sample has the fingerprint of the full data it came from.
        """
        sampling = dataset.sampling
        return cls(
            name=dataset.name,
            kind="data",
            profile=profile,
            n_rows=sampling.original_rows if sampling else dataset.n_rows,
            fingerprint=dataset.fingerprint,
            sampling=sampling,
        )

    @classmethod
    def of_run(cls, run: Run) -> "Reference":
        """The reference made from the profile a ``Run`` stored."""
        manifest = run.dataset
        return cls(
            name=manifest.name,
            kind="run",
            profile=run.profile,
            n_rows=manifest.n_rows,
            fingerprint=manifest.fingerprint,
            sampling=manifest.sampling,
        )
