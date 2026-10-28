# dsbf/checks/mutual_information.py
"""How much each feature tells about the target, in any shape of relationship."""

import numpy as np
from pydantic import Field

from dsbf.checks.pairs import Undefined
from dsbf.checks.target import (
    Kind,
    Target,
    Variable,
    class_variable,
    numeric_variable,
    read,
    resolve_target,
    sample_rows,
)
from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    Action,
    CheckParams,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Guidance,
    Severity,
    Subject,
)

_CONFIDENT_ROWS = 200
# Class features are measured against the null each on its own; with this many
# estimates in all they get fewer draws each (never under `_MIN_NULL_DRAWS`).
_CLASS_NULL_BUDGET = 4_000
_MIN_NULL_DRAWS = 12
# Above this a feature is, to within estimation error, the target.
_LEAK_LIKE = 0.95


class MutualInformationParams(CheckParams):
    target: str | None = Field(
        None,
        description="The column to predict. `dsbf.profile(target=...)` sets it; "
        "without one the check is not applicable.",
    )
    n_neighbors: int = Field(
        3,
        ge=1,
        description="Neighbours of the estimator for a continuous feature or "
        "target; more smooth the estimate and bias it low.",
    )
    min_normalised: float = Field(
        0.05,
        gt=0,
        lt=1,
        description="Normalised mutual information from which a feature is "
        "reported, when it also exceeds the noise floor.",
    )
    min_n: int = Field(
        30, ge=5, description="Complete rows a feature needs to be measured."
    )
    max_levels: int = Field(
        50,
        ge=2,
        description="Most levels a class column or target may have; the estimate "
        "over more is inflated by the levels' own noise.",
    )
    max_rows: int = Field(
        10_000,
        ge=1000,
        description="Most rows read; more are sampled with `seed`. The estimator "
        "searches neighbours, so its cost grows faster than the rows.",
    )
    null_draws: int = Field(
        40,
        ge=_MIN_NULL_DRAWS,
        description="Estimates against a shuffled target that set a noise floor, the "
        "highest of them: a truly independent feature passes it about once in "
        "`null_draws + 1`. A class feature gets that many of its own, since its "
        "estimate is biased up by its number of levels; the continuous features "
        "share one floor.",
    )
    null_features: int = Field(
        6,
        ge=1,
        description="Continuous features measured against shuffled targets to set "
        "their shared noise floor.",
    )
    null_shuffles: int = Field(
        2,
        ge=1,
        description="Shuffled copies of the target each of those features is "
        "measured against, or more when that leaves fewer than `null_draws` "
        "estimates.",
    )
    seed: int = Field(42, description="Seed of the row sample, shuffles and estimator.")
    max_findings: int = Field(
        20, ge=1, description="Most findings; the most informative first."
    )


class FeatureInformation(Contract):
    feature: str
    discrete: bool = Field(
        description="A class column, estimated from counts; else continuous."
    )
    mutual_information: float = Field(
        ge=0, description="Estimated mutual information with the target, in nats."
    )
    normalised: float = Field(
        ge=0,
        le=1,
        description="The share of the target's uncertainty the feature removes: for a "
        "class target its mutual information over the target's entropy; for a numeric "
        "one 1 - exp(-2 MI), which is r² for a bivariate normal pair.",
    )
    n_used: int = Field(
        ge=0,
        description="Rows the estimate used: those where the feature and target exist.",
    )
    noise_floor: float = Field(
        ge=0,
        description="The highest mutual information, in nats, this feature shows with "
        "a shuffled copy of the target.",
    )
    notable: bool = Field(
        description="At least `min_normalised` and above the noise floor."
    )


class MutualInformationOutput(CheckOutputModel):
    target: str
    target_kind: Kind
    target_classes: int | None = Field(
        description="Classes of a class target; None for a numeric one."
    )
    normalisation: str = Field(
        description="`target_entropy` for a class target, `information_r2` for a "
        "numeric one."
    )
    features: list[FeatureInformation] = Field(
        description="Every feature that was measured, most informative first."
    )
    noise_floor: dict[str, float] = Field(
        description="The highest noise floor, in nats, among the features of a kind "
        "(`discrete`, `continuous`): what a feature of that kind shows with a "
        "shuffled copy of the target."
    )
    features_tested: int = Field(ge=0)
    features_notable: int = Field(ge=0)
    untested: dict[str, int] = Field(
        description="Features not measured, by reason: too_few_rows or no_variation."
    )
    skipped_columns: dict[str, str] = Field(
        description="Columns left out, with the reason."
    )
    rows_used: int = Field(ge=0, description="Rows read, after any sampling.")
    rows_total: int = Field(ge=0)


def _copied_rows(columns: list[np.ndarray], rows: int) -> np.ndarray:
    """Which rows repeat an earlier row in every one of these continuous columns.

    The neighbour estimator reads a repeated point as a zero distance, which looks
    like dependence: 250 copied rows among 5 250 lift the estimate for an unrelated
    column from 0 to 0.057 nats. A column counts as continuous when nine in ten of
    its values are distinct, so a rounded or coded column, whose rows repeat by
    chance, does not make a row a copy. Fewer than two such columns leaves no row
    that can be told from a tie.
    """
    continuous = [c for c in columns if np.unique(c[~np.isnan(c)]).size >= 0.9 * rows]
    copied = np.zeros(rows, dtype=bool)
    if len(continuous) < 2:
        return copied
    key = np.column_stack([np.where(np.isnan(c), np.inf, c) for c in continuous])
    _, first = np.unique(key, axis=0, return_index=True)
    copied[:] = True
    copied[first] = False
    return copied


def _untie(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A continuous column with its ties spread, or the column when it has none.

    The neighbour estimator reads a repeated value as a zero distance, which looks
    like dependence when the column is only rounded (ages in whole years). Each value
    moves by up to half the median gap between distinct values, which gives rounded
    values back the spread they lost.
    """
    distinct = np.unique(values)
    if distinct.size == values.size or distinct.size < 2:
        return values
    gap = float(np.median(np.diff(distinct)))
    return values + rng.uniform(-0.5, 0.5, values.size) * gap


def _estimate(
    feature: Variable,
    target: Variable,
    both: np.ndarray,
    params: MutualInformationParams,
) -> tuple[float, np.ndarray]:
    """Mutual information in nats (never below 0), and the target values it used."""
    # Imported here: the module is part of the core pack, which imports without the
    # `ml` extra, and the plan marks this check unavailable when scikit-learn is not.
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    discrete = feature.kind == "class"
    x = (feature.codes if discrete else feature.values)[both].astype(np.float64)
    y = (target.codes if target.kind == "class" else target.values)[both]
    y = y.astype(np.float64)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise Undefined("no_variation")
    rng = np.random.default_rng(params.seed)
    if not discrete:
        x = _untie(x, rng)
    if target.kind == "numeric":
        y = _untie(y, rng)
    options = {
        "discrete_features": discrete,
        "n_neighbors": params.n_neighbors,
        "random_state": params.seed,
    }
    column = x.reshape(-1, 1)
    if target.kind == "class":
        estimate = mutual_info_classif(column, y.astype(np.int64), **options)
    else:
        estimate = mutual_info_regression(column, y, **options)
    return max(float(estimate[0]), 0.0), y


def _normalise(information: float, target: Variable, used: np.ndarray) -> float:
    if target.kind == "class":
        counts = np.bincount(used.astype(np.int64))
        shares = counts[counts > 0] / counts.sum()
        entropy = float(-np.sum(shares * np.log(shares)))
        return min(information / entropy, 1.0) if entropy > 0 else 0.0
    return float(1.0 - np.exp(-2.0 * information))


def _noise_floor(
    ctx: CheckContext,
    measured: list[tuple[str, Variable, np.ndarray]],
    target: Target,
    params: MutualInformationParams,
) -> tuple[dict[str, float], dict[str, float]]:
    """The highest mutual information that features show with a shuffled target.

    The estimator is not zero for independent data, its spread grows as rows fall,
    and a class feature's bias grows with its levels. So each class feature is
    measured against its own shuffled copies of the target (shuffled among the rows
    that have one), and a few continuous features, which share a bias, are measured
    against enough copies to give ``null_draws`` estimates. Returns the floor of each
    kind (the highest) and the floor of each feature.
    """
    floor = {"discrete": 0.0, "continuous": 0.0}
    by_feature: dict[str, float] = {}
    have = np.flatnonzero(target.present)
    shuffles: list[Target] = []

    def shuffle(number: int) -> Target:
        while len(shuffles) <= number:
            order = np.arange(target.present.size)
            seed = params.seed + len(shuffles) + 1
            order[have] = np.random.default_rng(seed).permutation(have)
            shuffles.append(target.take(order))
        return shuffles[number]

    def highest(feature: Variable, both: np.ndarray, draws: int) -> float:
        top = 0.0
        for number in range(draws):
            ctx.checkpoint()
            try:
                null, _ = _estimate(feature, shuffle(number), both, params)
            except Undefined:
                continue
            top = max(top, null)
        return top

    classes = [(n, f, b) for n, f, b in measured if f.kind == "class"]
    if classes:
        draws = max(
            _MIN_NULL_DRAWS, min(params.null_draws, _CLASS_NULL_BUDGET // len(classes))
        )
        for name, feature, both in classes:
            by_feature[name] = highest(feature, both, draws)
        floor["discrete"] = max(by_feature.values())
    pool = [(n, f, b) for n, f, b in measured if f.kind != "class"]
    if pool:
        chosen = pool[: params.null_features]
        draws = max(params.null_shuffles, -(-params.null_draws // len(chosen)))
        pooled = max(highest(f, b, draws) for _, f, b in chosen)
        floor["continuous"] = pooled
        for name, _, _ in pool:
            by_feature[name] = pooled
    return floor, by_feature


@check(
    id="mutual_information",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric", "categorical", "boolean"),
    dimensions=[Dimension.RELATIONSHIP],
    params=MutualInformationParams,
    output=MutualInformationOutput,
    requires=Requires(profile=["counts"], params=["target"]),
    cost=Cost.MODERATE,
    extras=["ml"],
    codes=["informative_feature"],
    packs=["core"],
    tags=["target", "mutual-information", "feature-selection"],
)
def mutual_information(
    ctx: CheckContext, params: MutualInformationParams
) -> CheckOutput:
    """Rank the features by what they tell about the target, linear or not.

    Each feature is measured on the rows where it and the target exist, so no value
    is imputed. A shuffled copy of the target gives the noise floor of the estimator,
    and a feature is notable only above it.
    """
    rows = sample_rows(ctx, params.max_rows, params.seed)
    resolved = resolve_target(ctx, params.target, rows, params.max_levels)
    if isinstance(resolved, CheckOutput):
        return resolved
    target: Target = resolved
    total_rows = ctx.profile.n_rows
    rows_used = total_rows if rows is None else int(rows.size)
    measured: list[tuple[str, Variable, np.ndarray]] = []
    untested: dict[str, int] = {}
    skipped: dict[str, str] = {}
    candidates: dict[str, Variable] = {}
    for name in ctx.columns:
        if name == target.name:
            continue
        column = ctx.profile.column(name)
        raw = read(ctx, name, rows)
        if column.semantic_type == "numeric":
            feature = numeric_variable(raw)
        else:
            distinct = column.counts.distinct
            if distinct < 2:
                skipped[name] = "fewer than two levels"
                continue
            if distinct > params.max_levels:
                skipped[name] = f"more than {params.max_levels} levels"
                continue
            feature = class_variable(raw)
        candidates[name] = feature
    numeric = [f.values for f in candidates.values() if f.kind == "numeric"]
    if target.kind == "numeric":
        numeric.append(target.values)
    copied = _copied_rows(numeric, target.present.size)
    if copied.any():
        ctx.note(
            "repeated_rows_counted_once",
            f"{int(np.count_nonzero(copied)):,} rows repeat an earlier row in every "
            "continuous column and are left out, since the neighbour estimator reads "
            "a copy as dependence.",
            hint="`duplicate_rows` and `fuzzy_duplicates` report such rows.",
        )
    for name, feature in candidates.items():
        both = feature.present & target.present & ~copied
        if int(np.count_nonzero(both)) < params.min_n:
            untested["too_few_rows"] = untested.get("too_few_rows", 0) + 1
            continue
        measured.append((name, feature, both))

    scores: list[tuple[str, bool, float, float, int]] = []
    kept: list[tuple[str, Variable, np.ndarray]] = []
    for name, feature, both in measured:
        ctx.checkpoint()
        try:
            information, used = _estimate(feature, target, both, params)
        except Undefined as why:
            untested[str(why)] = untested.get(str(why), 0) + 1
            continue
        kept.append((name, feature, both))
        scores.append(
            (
                name,
                feature.kind == "class",
                information,
                _normalise(information, target, used),
                int(used.size),
            )
        )
    floor, feature_floor = _noise_floor(ctx, kept, target, params)

    features = sorted(
        (
            FeatureInformation(
                feature=name,
                discrete=discrete,
                mutual_information=information,
                normalised=normalised,
                n_used=n_used,
                noise_floor=feature_floor[name],
                notable=normalised >= params.min_normalised
                and information > feature_floor[name],
            )
            for name, discrete, information, normalised, n_used in scores
        ),
        key=lambda f: -f.normalised,
    )
    notable = [f for f in features if f.notable]
    shown = notable[: params.max_findings]
    if len(notable) > len(shown):
        ctx.note(
            "findings_capped",
            f"{len(notable):,} features are notable; the findings cover the "
            f"{len(shown):,} most informative, all of them are in the output.",
        )
    for feature in shown:
        _report(ctx, feature, target, total_rows, floor, params)
    return ctx.output(
        MutualInformationOutput(
            target=target.name,
            target_kind=target.kind,
            target_classes=target.n_levels if target.kind == "class" else None,
            normalisation=(
                "target_entropy" if target.kind == "class" else "information_r2"
            ),
            features=features,
            noise_floor=floor,
            features_tested=len(features),
            features_notable=len(notable),
            untested=untested,
            skipped_columns=skipped,
            rows_used=rows_used,
            rows_total=total_rows,
        )
    )


def _report(
    ctx: CheckContext,
    feature: FeatureInformation,
    target: Target,
    total_rows: int,
    floor: dict[str, float],
    params: MutualInformationParams,
) -> None:
    name = feature.feature
    meaning = (
        f"removes {feature.normalised:.1%} of the uncertainty about the target"
        if target.kind == "class"
        else f"explains {feature.normalised:.1%} of the target (an r² computed from "
        "mutual information, so it counts any shape of dependence)"
    )
    kind = "discrete" if feature.discrete else "continuous"
    leak = (
        f" At this level {name!r} nearly determines the target, which a real "
        "predictor rarely does; ask whether it is known at prediction time "
        "(`target_leakage` checks for that)."
        if feature.normalised >= _LEAK_LIKE
        else ""
    )
    ctx.finding(
        code="informative_feature",
        subject=Subject.column(name),
        severity=Severity.INFO,
        confidence=(
            Confidence.HIGH
            if feature.n_used >= _CONFIDENT_ROWS
            else Confidence.MODERATE
        ),
        title=f"Informative about {target.name!r} ({feature.normalised:.2f})",
        message=(
            f"Knowing {name!r} {meaning} (mutual information "
            f"{feature.mutual_information:.3f} nats on {feature.n_used:,} rows, "
            f"above the {floor[kind]:.3f} a shuffled target gives). Unlike a "
            "correlation, it also counts a non-linear or non-monotonic relationship, "
            "and says nothing about its shape: plot the feature against the target."
            + leak
        ),
        evidence=Evidence(
            metrics={
                "mutual_information": feature.mutual_information,
                "normalised": feature.normalised,
                "noise_floor": floor[kind],
                "n_neighbors": params.n_neighbors,
                "target": target.name,
            },
            n_used=feature.n_used,
            n_total=total_rows,
            method="knn_mutual_information"
            if not feature.discrete or target.kind == "numeric"
            else "plug_in_mutual_information",
        ),
        guidance=Guidance(
            phase="ml",
            actions=(
                Action(
                    action="keep",
                    columns=(name,),
                    condition="its value is known when the model has to predict",
                    detail=f"A candidate feature for predicting {target.name!r}.",
                ),
            ),
        ),
    )
