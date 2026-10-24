"""Rows repeated exactly, and records written twice with a small difference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dsbf.model import CheckResult
from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, Probe, plant
from tests.generators.dsbf_test_dataset_generator import DSBFTestDataGenerator

FIRST = (
    "Alice Bruno Carla Dmitri Elena Farid Greta Hugo Ines Jamal Katya Liam Mona "
    "Nils Olga Pavel Quinn Rosa Sven Tara Uma Viktor Wanda Xavier Yara Zoltan "
    "Amara Boris Chloe Dario Esme Felix Gwen Hana Igor Jules Kira Leon Mira Noor "
    "Omar Petra Raul Sana Tomas Ursula Vera Willem Yusuf Zara"
).split()
LAST = (
    "Anders Berg Castillo Duarte Eriksen Fontaine Garcia Haas Ivanov Jensen Kovacs "
    "Larsen Moreau Novak Okafor Petrov Quist Rossi Schmidt Tanaka Ueda Varga Weber "
    "Xu Yilmaz Zhang Abbott Baptiste Cohen Dubois Evans Fischer Gomez Hartmann "
    "Ibrahim Jansen Keller Lopez Mensah Nowak Ortiz Patel Rahman Silva Torres"
).split()
CITIES = (
    "Lisbon Oslo Pune Quito Riga Seoul Turin Utrecht Vilnius Wroclaw Zagreb Accra "
    "Bergen Cork Doha Essen Faro Genoa Hanoi Izmir Jaipur Kyoto Lagos Malmo Nantes"
).split()
STREETS = "Oak Pine Cedar Elm Birch Maple Willow Ash Spruce Hazel Alder Rowan".split()


def records(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """``n`` distinct customer records: name, email, street, city."""
    first = rng.choice(FIRST, n)
    last = rng.choice(LAST, n)
    serial = rng.integers(10, 9999, n)
    return pd.DataFrame(
        {
            "name": [f"{f} {s}" for f, s in zip(first, last, strict=True)],
            "email": [
                f"{f.lower()}.{s.lower()}{k}@example.org"
                for f, s, k in zip(first, last, serial, strict=True)
            ],
            "street": [
                f"{k} {st} Street"
                for k, st in zip(rng.integers(1, 400, n), rng.choice(STREETS, n))
            ],
            "city": rng.choice(CITIES, n),
            "balance": rng.normal(1000, 300, n).round(2),
        }
    )


@scenario("duplicate_rows_planted")
def duplicate_rows_planted(seed: int) -> Planted:
    """800 distinct records and 3% exact repeats of them."""
    rng = np.random.default_rng(seed)
    base = records(rng, 800)
    repeats = base.iloc[rng.choice(800, 24, replace=False)]
    frame = pd.concat([base, repeats], ignore_index=True)
    return Planted(
        frame,
        plants=(plant("duplicate_rows"),),
        checks=("duplicate_rows",),
    )


@scenario("rows_distinct")
def rows_distinct(seed: int) -> Planted:
    """800 distinct customer records with names, streets and cities that overlap."""
    rng = np.random.default_rng(seed)
    return Planted(
        records(rng, 800),
        allowed=(plant("id_columns", "email"),),
        checks=("duplicate_rows", "fuzzy_duplicates", "id_columns"),
    )


def _typo_pairs(result: CheckResult, planted: Planted) -> float | None:
    if result.output is None:
        return None
    truth = planted.meta["pairs"]
    return min(result.output["pairs_flagged"], truth) / truth


def _extra_pairs(result: CheckResult, planted: Planted) -> float | None:
    if result.output is None:
        return None
    flagged, truth = result.output["pairs_flagged"], planted.meta["pairs"]
    return max(0, flagged - truth) / max(1, result.output["records"])


FUZZY_PROBES = (
    Probe("fuzzy_duplicates", "pair_recall", _typo_pairs),
    Probe("fuzzy_duplicates", "extra_pairs_per_record", _extra_pairs, False),
)


def _with_copies(
    rng: np.random.Generator, base: pd.DataFrame, copies: int, columns_edited: int
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """``base`` followed by ``copies`` of its records, each with edited text."""
    kinds = ("case", "padding", "typo", "transposition", "dropped_character")
    sources = rng.choice(len(base), copies, replace=False)
    rows = base.iloc[sources].copy().reset_index(drop=True)
    text = ["name", "email", "street", "city"]
    for k in range(copies):
        for column in rng.choice(text, columns_edited, replace=False):
            kind = kinds[(k + len(column)) % len(kinds)]
            value = str(rows.at[k, column])
            edited = DSBFTestDataGenerator._edit(value, kind, rng)
            rows.at[k, column] = edited if edited != value else value + " "
    return pd.concat([base, rows], ignore_index=True), [
        (int(a), len(base) + k) for k, a in enumerate(sources)
    ]


@scenario("fuzzy_one_edit")
def fuzzy_one_edit(seed: int) -> Planted:
    """800 records and 40 copies, each with one text column edited."""
    rng = np.random.default_rng(seed)
    frame, truth = _with_copies(rng, records(rng, 800), 40, 1)
    return Planted(
        frame,
        plants=(plant("fuzzy_duplicates"),),
        checks=("fuzzy_duplicates",),
        probes=FUZZY_PROBES,
        meta={"pairs": 40, "truth": truth},
    )


@scenario("fuzzy_two_edits")
def fuzzy_two_edits(seed: int) -> Planted:
    """800 records and 40 copies, each with two text columns edited."""
    rng = np.random.default_rng(seed)
    frame, truth = _with_copies(rng, records(rng, 800), 40, 2)
    return Planted(
        frame,
        plants=(plant("fuzzy_duplicates"),),
        checks=("fuzzy_duplicates",),
        probes=FUZZY_PROBES,
        meta={"pairs": 40, "truth": truth},
    )


@scenario("fuzzy_look_alikes")
def fuzzy_look_alikes(seed: int) -> Planted:
    """Distinct products whose codes and names differ by a digit or one word."""
    rng = np.random.default_rng(seed)
    n = 600
    series = rng.choice(np.arange(100, 900), n, replace=False)
    frame = pd.DataFrame(
        {
            "code": [f"SKU-{k:04d}-{rng.choice(list('ABC'))}" for k in series],
            "name": [
                f"{rng.choice(['Steel', 'Copper', 'Oak', 'Pine'])} "
                f"{rng.choice(['bolt', 'plate', 'hinge', 'rail'])} {k % 50} mm"
                for k in series
            ],
            "supplier": rng.choice([f"Supplier {i}" for i in range(40)], n),
        }
    )
    frame = frame.drop_duplicates().reset_index(drop=True)
    return Planted(frame, checks=("fuzzy_duplicates",))


@scenario("fuzzy_blocked", seeds=10)
def fuzzy_blocked(seed: int) -> Planted:
    """2 300 records, so only pairs that share a rare token are scored, 40 copied."""
    rng = np.random.default_rng(seed)
    frame, truth = _with_copies(rng, records(rng, 2300), 40, 2)
    return Planted(
        frame,
        plants=(plant("fuzzy_duplicates"),),
        checks=("fuzzy_duplicates",),
        probes=FUZZY_PROBES,
        meta={"pairs": 40, "truth": truth},
    )


def _two_sources(seed: int, params: dict) -> Planted:
    rng = np.random.default_rng(seed)
    base = records(rng, 760)
    sources = rng.choice(len(base), 40, replace=False)
    first = base.copy()
    first["key"] = [f"A-{i:05d}" for i in range(len(first))]
    second = base.iloc[sources].copy().reset_index(drop=True)
    second["key"] = [f"B-{rng.integers(10**4, 10**5)}" for _ in range(len(second))]
    second["name"] = [
        DSBFTestDataGenerator._edit(str(v), "typo", rng) for v in second["name"]
    ]
    frame = pd.concat([first, second], ignore_index=True)
    return Planted(
        frame,
        plants=(plant("fuzzy_duplicates"),),
        checks=("fuzzy_duplicates",),
        params=params,
        probes=FUZZY_PROBES,
        meta={
            "pairs": 40,
            "truth": [(int(a), len(first) + k) for k, a in enumerate(sources)],
        },
    )


@scenario("fuzzy_two_sources")
def fuzzy_two_sources(seed: int) -> Planted:
    """The same 40 people in two systems with different keys, which are left out."""
    return _two_sources(seed, {"fuzzy_duplicates": {"exclude_columns": ["key"]}})


@scenario("fuzzy_two_sources_unaided")
def fuzzy_two_sources_unaided(seed: int) -> Planted:
    """The same 40 people in two systems, each with its own key, none left out."""
    return _two_sources(seed, {})
