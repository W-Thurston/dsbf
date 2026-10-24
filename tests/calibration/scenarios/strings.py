"""Text columns with encoded values, mixed types, stray characters or mixed formats."""

from __future__ import annotations

import base64
import string
import uuid

import numpy as np
import pandas as pd

from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, plant

WORDS = (
    "river stone cloud maple harbor violet ember falcon meadow quartz lantern "
    "orchard summit thunder willow copper marble saffron tundra velvet walnut"
).split()
CITIES = ["Paris", "Berlin", "Madrid", "Vienna", "Prague", "Dublin", "Athens"]


def _hex(rng: np.random.Generator, n: int, digits: int) -> list[str]:
    return [rng.bytes(digits // 2 + 1).hex()[:digits] for _ in range(n)]


@scenario("encoded_planted")
def encoded_planted(seed: int) -> Planted:
    """Columns of UUIDs, digests, base64 and random keys beside plain words."""
    rng = np.random.default_rng(seed)
    n = 1000
    alphabet = np.array(list(string.ascii_letters + string.digits))
    frame = pd.DataFrame(
        {
            "uuid": [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(n)],
            "sha1": _hex(rng, n, 40),
            "sha256": _hex(rng, n, 64),
            "short_hex": _hex(rng, n, 8),
            "base64": [base64.b64encode(rng.bytes(15)).decode() for _ in range(n)],
            "token": ["".join(rng.choice(alphabet, 24)) for _ in range(n)],
            "word": rng.choice(WORDS, n),
            "two_words": [
                f"{a} {b}" for a, b in zip(rng.choice(WORDS, n), rng.choice(WORDS, n))
            ],
            "city": rng.choice(CITIES, n),
            "email": [
                f"{rng.choice(WORDS)}{rng.integers(10, 999)}@{rng.choice(WORDS)}.org"
                for _ in range(n)
            ],
            "product_code": [f"SKU-{rng.integers(10000, 99999)}-A" for _ in range(n)],
            "timestamp": pd.to_datetime(
                pd.Timestamp("2024-01-01")
                + pd.to_timedelta(rng.integers(0, 10**7, n), "s")
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mac": [
                ":".join(rng.bytes(6).hex()[i : i + 2] for i in range(0, 12, 2))
                for _ in range(n)
            ],
        }
    )
    return Planted(
        frame,
        plants=tuple(
            plant("encoded_columns", c)
            for c in ("uuid", "sha1", "sha256", "short_hex", "base64", "token")
        ),
        checks=("encoded_columns",),
    )


@scenario("mixed_types_planted")
def mixed_types_planted(seed: int) -> Planted:
    """Strings that are half numbers and half words, beside tidy text columns."""
    rng = np.random.default_rng(seed)
    n = 1000
    numbers = rng.integers(0, 500, n).astype(str)
    mixed = np.where(rng.random(n) < 0.5, numbers, rng.choice(WORDS, n))
    dates = np.where(
        rng.random(n) < 0.7,
        pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D"),
        pd.Series(rng.choice(WORDS, n)),
    ).astype(str)
    null_tokens = np.where(
        rng.random(n) < 0.2,
        rng.choice(["N/A", "null", "-", "?"], n),
        rng.choice(CITIES, n),
    )
    frame = pd.DataFrame(
        {
            "mixed": mixed,
            "dates_and_words": dates,
            "words": rng.choice(WORDS, n),
            "city_with_null_tokens": null_tokens,
            "numbers_as_text": numbers,
        }
    )
    return Planted(
        frame,
        plants=(
            plant("mixed_types", "mixed"),
            plant("mixed_types", "dates_and_words"),
        ),
        checks=("mixed_types",),
    )


@scenario("string_anomalies_planted")
def string_anomalies_planted(seed: int) -> Planted:
    """City columns with case variants, padding, hidden characters and a long value."""
    rng = np.random.default_rng(seed)
    n = 1000
    city = rng.choice(CITIES, n)
    cased = np.where(rng.random(n) < 0.3, np.char.upper(city), city)
    cased = np.where(rng.random(n) < 0.2, np.char.lower(cased), cased)
    padded = np.where(rng.random(n) < 0.15, np.char.add(" ", city), city)
    hidden = np.where(rng.random(n) < 0.1, np.char.add(city, "​"), city)
    long_values = city.astype(object)
    long_values[rng.choice(n, 5, replace=False)] = "x" * 300
    frame = pd.DataFrame(
        {
            "clean": city,
            "cased": cased,
            "padded": padded,
            "hidden": hidden,
            "long_values": long_values,
        }
    )
    return Planted(
        frame,
        plants=(
            plant("string_anomalies", "cased", codes=["mixed_case"]),
            plant("string_anomalies", "padded", codes=["edge_whitespace"]),
            plant("string_anomalies", "hidden", codes=["invisible_characters"]),
            plant("string_anomalies", "long_values", codes=["length_outliers"]),
        ),
        checks=("string_anomalies",),
    )


@scenario("format_violations_planted")
def format_violations_planted(seed: int) -> Planted:
    """A phone column with 5% malformed values; dates in one format and in two."""
    rng = np.random.default_rng(seed)
    n = 1000
    phone = np.array(
        [f"{rng.integers(100, 999)}-{rng.integers(1000, 9999)}" for _ in range(n)]
    )
    phone[rng.random(n) < 0.05] = "call me"
    good = rng.choice(["2024-03-05", "2023-11-30", "2022-07-14", "2024-12-01"], n)
    stamps = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D")
    one_format = stamps.strftime("%Y-%m-%d")
    two_formats = np.where(
        rng.random(n) < 0.3, stamps.strftime("%d/%m/%Y"), stamps.strftime("%Y-%m-%d")
    )
    frame = pd.DataFrame(
        {
            "phone": phone,
            "iso_dates": one_format,
            "mixed_dates": two_formats,
            "few_dates": good,
        }
    )
    return Planted(
        frame,
        plants=(
            plant("format_violations", "phone", codes=["format_violations"]),
            plant("format_violations", "mixed_dates", codes=["mixed_date_formats"]),
        ),
        params={
            "format_violations": {
                "rules": [{"column": "phone", "pattern": r"\d{3}-\d{4}"}]
            }
        },
        checks=("format_violations",),
    )
