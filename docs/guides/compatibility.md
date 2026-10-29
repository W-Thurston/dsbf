# What stays compatible in 1.x

From `1.0.0`, the contracts DSBF publishes follow semantic versioning. Code that reads them keeps working for the whole 1.x series: a new minor release adds, and only `2.0.0` may remove or reshape.

## The contracts

| Contract | Where it is published | Version | Bump it in |
|---|---|---|---|
| The run (`run.json`, `Run`) | `schema/run.schema.json` | `schema_version` | `SCHEMA_VERSION` in `dsbf/model/run.py` |
| The Parquet tables | `schema/tables.schema.json` | `schema_version` | `TABLES_SCHEMA_VERSION` in `dsbf/sinks/tables.py` |
| A check's parameters and output | `schema/checks/<id>.schema.json` | `check_version` | the check's `version` |
| The policy file | `schema/policy.schema.json` | `version`, a whole number | `Policy.version` in `dsbf/model/policy.py` |
| The comparison of two runs (`RunDiff`) | `schema/diff.schema.json` | `schema_version` | `DIFF_SCHEMA_VERSION` in `dsbf/model/diff.py` |
| The dashboard API | `schema/openapi.json` | `info.version` | `API_VERSION` in `dsbf/service/app.py` |

A check's version is the version of its statistics and output, so it moves on its own and is not reset at 1.0.0. The API document was already at 1.x while the dashboard was built (`1.8.0` at the release), and a version never goes down, so 1.x compatibility counts from there.

## The rules

- **Additive changes are minor.** A new optional field, a new nullable table column, a new endpoint, a new check or a new value of an open vocabulary raise the minor version.
- **Breaking changes are major.** Removing or renaming a field, column, endpoint or parameter, changing a type, making an optional field required, or removing a check needs the next major version, `2.0.0` for the run.
- **A deprecation lasts one minor release.** Something that is going away is marked deprecated in a minor release, keeps working and says what replaces it, and may be removed no earlier than the next major version.
- **A change that leaves the shape alone needs nothing.** Wording, descriptions and defaults of a field's documentation do not move a version.
- **A reader accepts its own major.** `Run` reads any `1.x` run; a run from another major is refused with the versions in the message.
- **The policy file is different.** Its `version` is a whole number, so only a breaking change moves it; an additive change needs no bump.

A run carries the version it was written with, and the settings it was made with, so a stored run can be read and rerun by any 1.x release:

```python
import dsbf
from dsbf import Run

run = dsbf.profile("data.csv", depth="quick")
run.write("out", sinks=["json"])

again = Run.read("out")
assert again.schema_version.startswith("1.")
assert again.plan.settings == {"depth": "quick"}
```

## How the rules are enforced

`dsbf schema export` writes every file above from the models, and `--check` fails when a committed file is stale. `dsbf schema check --base origin/main` compares each file with the one on the base branch and fails when the change is not announced by its version: a breaking change without a major bump, or an additive one without a minor bump. CI runs both on every pull request. When a gate fails, bump the constant named in the table, run `dsbf schema export --types frontend/src/generated/run.d.ts`, and commit the regenerated files with the change.

## Known limitations

- **`RunDiff` counts size-scaled metrics as changes.** Comparing a full run with a run of a sample of the same data reports many changed findings, because metrics such as counts grow with the number of rows (the titanic table against its first 300 rows reports 101). Compare runs of the same sample size until this is investigated; a per-metric tolerance or a list of metrics that scale with the row count is under consideration and would change what `RunDiff` reports, not its shape.
