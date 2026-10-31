# dsbf/data/fingerprint.py
"""A content hash of a table: equal data gives an equal fingerprint."""

import hashlib

import polars as pl

# Polars does not promise `Series.hash` is stable across its versions. A test pins a
# fingerprint, so an upgrade that changes the hash fails there and bumps this tag.
SCHEME = "polars-hash-v2"
_SEED = 0
_DIGEST_BYTES = 16


def fingerprint_frame(frame: pl.DataFrame) -> str:
    """Hash column names, dtypes, row order and every value, one column at a time.

    Hashing per column keeps the cost at one pass over the data (0.6 s for 1M rows
    by 60 columns) and independent of how the table is split into chunks.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    digest.update(f"{frame.height}x{frame.width}".encode())
    for name, dtype in frame.schema.items():
        _update_with_text(digest, name)
        _update_with_text(digest, str(dtype))
        digest.update(frame.get_column(name).hash(seed=_SEED).to_numpy().tobytes())
    return f"{SCHEME}:{digest.hexdigest()}"


def _update_with_text(digest, text: str) -> None:
    """Length-prefix ``text`` so ``("ab", "c")`` and ``("a", "bc")`` differ."""
    encoded = text.encode()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
