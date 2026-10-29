# Run DSBF on your own machine

> **Preview.** The run store, the API and the dashboard are on `main` and ship with `1.0.0`. The
> dashboard shows each page's data, mostly as tables and cards; a chart-led redesign is planned
> (Phase 6b in [`ROADMAP.md`](../../ROADMAP.md)) before that release. Until then, install from a
> checkout or a wheel built from `main` (`.github/scripts/build_dashboard.sh && uv build`).

Mode 2 of DSBF: you profile data from the command line, the runs pile up in a local store, and
`dsbf dashboard` shows them in a browser. Nothing leaves your machine: the dashboard listens on
`127.0.0.1`, `localhost` or `::1` only, and has no login because it has no other audience. The
examples use the customer table `data.csv` from [Use DSBF in your pipeline](pipeline.md).

## Install

The dashboard and its store need the `server` extra. A wheel built with
`.github/scripts/build_dashboard.sh` carries the built dashboard, so Node is not needed to run it:

```console
pip install "dsbf[server] @ file:///path/to/dsbf-<version>-py3-none-any.whl"
```

## Keep runs

`--sink store` writes a run to the store (and a snapshot of the data as profiled, so a finding
can be traced to its rows later). The store lives under `DSBF_HOME` (by default your user data
directory); a run's name for its dataset is `--dataset`, and a file profiled again joins the
same dataset by its path.

```bash
dsbf check data.csv --sink store --dataset customers
dsbf check data.csv --sink store --dataset customers --depth deep
dsbf runs list
```

`dsbf runs show`, `diff`, `delete` and `gc` browse, compare and clean the store; see the
[CLI reference](../CLI_REFERENCE.md).

## Open the dashboard

```console
dsbf dashboard
```

It serves on `http://127.0.0.1:8765/` and opens your browser (`--no-open` and `--port` change
that). The pages are:

| Page | Shows |
| --- | --- |
| Datasets and a dataset's page | its runs, how the source moved between them, and the history of a finding |
| Run overview | health (with the reason for every check that did not run), the policy verdict and the top findings |
| Findings | every finding, filtered and paged |
| Columns | one column's profile, with charts |
| Relationships | associations between columns |
| Quality and ML readiness | five quality and five readiness dimensions and the ML gate |
| Time series | stationarity, ACF and PACF and the STL decomposition |
| Compare | two runs: new, resolved and changed findings |
| Settings | a dataset's saved settings, in a form, with a rerun |

A rerun from the dashboard profiles the dataset's file again with its saved settings, so the
file has to still be where the first run read it.

## If the dashboard is missing

A build made from a source checkout without the frontend serves the API only, and its `/` page
says the dashboard is not installed. Build it with `pnpm --dir frontend build` (Node 24), or
install a release from PyPI. The API is documented at `/v2/docs` while the server runs.
