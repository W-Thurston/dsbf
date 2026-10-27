# Write a policy

A policy is a YAML file of rules over a run's findings and the health of its checks. A rule has a
`when` that selects findings (or checks), and a `then` that says what to do with them. A policy is
data, never code, and it is report-only by default: only a rule whose `then` includes `fail` can
make `dsbf check` exit with 1.

The examples run against the customer table `data.csv` used in
[Use DSBF in your pipeline](pipeline.md).

## A first policy

```python
from pathlib import Path

Path("policy.yaml").write_text("""\
version: 1
rules:
  - name: heavy-tails
    when: { code: heavy_tails, severity: ">=warn" }
    then: fail
  - name: skewed-columns
    when: { dimension: distribution, severity: warn }
    then: warn
  - name: broken-checks
    when: { health.status: error }
    then: annotate
""")
```

Run it:

```bash
dsbf check data.csv --depth quick --policy policy.yaml --out results   # exits 1
```

or from Python:

```python
import dsbf
from dsbf.policy import load_policy

policy = load_policy("policy.yaml")
run = dsbf.profile("data.csv", depth="quick", policy=policy)
for outcome in run.policy.rules:
    print(outcome.rule.name, "matched", len(outcome.finding_ids), "findings")
print("exit code", run.policy.exit_code)
```

`Run.policy` records every rule, fired or not, with the ids of the findings it matched, so the file
`run.json` explains why a pipeline failed.

## The `when` of a rule

Every key you give must hold, and a rule needs at least one.

| Key | Selects |
| --- | --- |
| `dimension` | findings of one quality dimension, such as `leakage` |
| `code` | findings of one kind, such as `high_null_rate`; each check's codes are in the [check reference](../CHECK_REFERENCE.md) |
| `check` | findings of one check, by id |
| `column` | findings whose subject names the column (a pair matches on either) |
| `tag` | findings carrying the tag |
| `severity`, `confidence` | findings at a level; `info` < `notice` < `warn` < `error` and `low` < `moderate` < `high` |
| `metric` | findings whose evidence has these numbers, such as `{ null_share: ">0.3" }` |
| `new_since: previous_run` | findings absent from the previous run |
| `health.status` | checks that ended in a status (`error`, `timeout`, `skipped` and so on), not findings |

A bare value means equality. `severity`, `confidence` and `metric` values may begin with an operator
(`>=`, `>`, `<=`, `<`, `==`, `!=`), which YAML needs in quotes. `health.status` selects checks and
combines only with `check`. A metric rule reads the metric by name from the finding's evidence and
never cross-reads another check's bounds.

Test a `when` before you commit it. `Findings.filter` takes the same keys:

```python
rows = run.findings.filter(code="heavy_tails", severity=">=warn")
print([finding.subject.columns for finding in rows])
```

## The `then` of a rule

| Action | Effect |
| --- | --- |
| `annotate` | The default. Records the match in `Run.policy`. |
| `warn` | Records the match like `annotate`; it marks a rule as a warning for the person reading the policy and never changes the exit code. |
| `fail` | Makes the exit code of `dsbf check` 1 (`Run.policy.failed`). |
| `notify` | Records a message for a target written `scheme:destination`, such as `{notify: "slack:#data-quality"}`. DSBF sends nothing itself; a pipeline step reads `Run.policy` and does. |

A rule may list several actions: `then: [warn, { notify: "slack:#data-quality" }]`.

## Mistakes are named

A policy that does not validate raises `PolicyError` (`dsbf check` exits 2) and lists every problem
under the name of its rule:

```python
from dsbf.policy import PolicyError

try:
    load_policy({"rules": [{"name": "too-loose", "when": {}, "then": "fail"}]})
except PolicyError as error:
    print(error)
```

`schema/policy.schema.json` is the JSON Schema of the file, for editor completion.
