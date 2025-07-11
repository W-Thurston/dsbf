# DSBF – Data Scientist’s Best Friend

**DSBF** is a fast, modular, and extensible profiling engine for tabular datasets. It analyzes raw CSVs, Pandas, or Polars DataFrames and produces actionable data health reports to support EDA, model diagnostics, and data quality initiatives.

---

## Key Features

- Supports both **Pandas** and **Polars** backends
- Generates structured reports with **task-level results**, **ML impact scores**, and **reliability warnings**
- Saves all outputs (plots, logs, JSON) to a **timestamped directory**
- Plugin-ready with a registry and safety checks
- Infers data maturity stage (raw, cleaned, modeling) to tailor analysis
- Supports CLI, Python API, and Jupyter notebook workflows
- Built-in task filtering by profiling depth, runtime estimate, domain, etc.
- DAG-based execution engine with visual output (optional)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/W-Thurston/dsbf.git
cd dsbf

# Install dependencies
poetry install  # or pip install -e .
````

You’ll also need:

* Python 3.8+
* Optional: Graphviz (for DAG visualizations)

---

## Usage Options: CLI, API, and Jupyter

DSBF can be run via **command line**, **Python API**, or inside a **Jupyter notebook**. All interfaces use the same profiling engine under the hood.

### 1. Command-Line Interface (CLI)

The CLI is the easiest way to run a full profiling report from the terminal.

#### Basic Usage

```bash
dsbf profile data.csv
```

#### Options

| Flag / Command       | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `--depth`            | Set profiling depth: `basic`, `standard`, `full`          |
| `--verbosity`        | Set log level: `quiet`, `warn`, `stage`, `info2`, `debug` |
| `--config path.yaml` | Use a custom config file                                  |
| `--output path/`     | Set output directory (default: `dsbf/outputs/TIMESTAMP`)  |

#### Example

```bash
dsbf profile data.csv --depth full --verbosity debug
```

---

### 2. Quickstart Command

For rapid experimentation with built-in datasets like Titanic:

```bash
dsbf quickstart titanic
```

This loads the dataset using `seaborn.load_dataset("titanic")` and runs a full profiling pipeline.

---

### 3. Python API

You can run DSBF programmatically:

```python
from dsbf.api import profile_file

results = profile_file("data.csv", depth="standard")
```

Or run with a custom config:

```python
from dsbf.api import profile_file
from dsbf.config import load_config

cfg = load_config("custom_config.yaml")
results = profile_file("data.csv", config=cfg)
```

---

### 4. Jupyter / Notebook Usage

For interactive use:

```python
from dsbf.api import ProfileEngine
import pandas as pd

df = pd.read_csv("data.csv")

engine = ProfileEngine()
engine.df = df  # Use in-memory DataFrame
results = engine.run()

# View summary
results[0].summary
```

This avoids writing to disk unless you explicitly export results.

---

## Output Artifacts

Each run creates a timestamped folder under `dsbf/outputs/`, containing:

| File                   | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `report.json`          | Main profiling results from EDA tasks                      |
| `metadata_report.json` | System metadata, timing, and diagnostic results            |
| `run.log`              | Full log of task execution                                 |
| `figs/`                | All generated plots (static + interactive)                 |
| `dag.png`              | Optional visualization of the execution graph (if enabled) |

---

---

## Configuration

DSBF uses a YAML config file to control engine behavior, profiling depth, task selection, and output settings. See `default_config.yaml` for reference.

Examples:

```yaml
metadata:
  dataset_path: "data.csv"
  profiling_depth: "standard"
  message_verbosity: "info"

engine:
  backend: "polars"
  reference_dataset_path: null

task_groups:
  - core

tasks:
  detect_outliers:
    sensitivity: 0.01
```

---

## Plugin System

You can extend DSBF by adding custom tasks in a directory and loading them via config:

```yaml
task_groups:
  - core
  - ./custom_plugins/healthcare/
```

Tasks must use the `@register_task` decorator and subclass `BaseTask`. Plugins are validated at runtime, and any files that register no tasks will generate a warning in `metadata_report.json`.

---

## Advanced Topics

* **Execution DAG**: Tasks run in topological order with dependency resolution and failure recovery. See `graph.py` for details.
* **Stage Inference**: DSBF auto-detects whether data is raw, cleaned, or modeling-ready to prioritize appropriate tasks.
* **ML Impact Scores**: TaskResults can include an `ml_impact_score` and ranked recommendations to guide downstream decisions.
* **Reliability Warnings**: Warnings are structured into high/medium/low tiers with suggested next steps.

---

## Roadmap

Planned features:

* [ ] Interactive Panel-based report viewer
* [ ] Data drift detection & monitoring
* [ ] Pipeline integration (e.g., Airflow, Prefect)
* [ ] GitHub Actions for automated profiling
* [ ] More core tasks and built-in domains (healthcare, NLP, etc.)

---

## Contributing

We welcome contributions! Please:

1. Fork the repo
2. Create a feature branch
3. Write clear commits and add tests
4. Submit a pull request

All contributions must include docstrings, adhere to linting standards, and pass existing test suites (`pytest`).

---

## License

MIT License. See `LICENSE` for details.

---
