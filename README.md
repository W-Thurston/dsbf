
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

### Output Structure

All runs create a timestamped directory under `dsbf/outputs/`, containing:

* `report.json` – structured task output
* `run.log` – complete logs
* `figs/` – generated plots
* `metadata_report.json` – run metadata
