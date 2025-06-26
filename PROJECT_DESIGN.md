# Project Design: DSBF

## 🎯 Purpose

DSBF exists to bridge the gap between black-box AutoML tools and rigorous, transparent diagnostics. Its goal is to:

* Teach beginners *why* model assumptions matter.
* Give mid-level DSs a toolkit to catch red flags early.
* Let experts plug in and extend new modules with confidence.

## 📐 Architecture Overview

```
📦 DSBF Root
├── run_dsbf.py               # Entry point with YAML config
├── dsbf/
│   ├── checks/               # Assumption modules (e.g. linearity.py)
│   ├── core/                 # Shared utils (e.g. model_wrapper, config)
│   ├── report/               # Report builder
│   └── engine/               # Polars/Pandas backend adapters
├── tests/                    # Unit tests per module
├── docs/                     # Markdown docs + design notes
└── config/                   # Example YAMLs
```

## 📦 Core Principles

* **Modularity**: Every assumption check is its own unit.
* **Transparency**: Outputs are human-readable by default.
* **Reproducibility**: Runs are YAML-configurable with versioned outputs.
* **Extensibility**: Add your own check with a single import and a few lines.

## 🔮 Stretch Goals

* DAG-based execution flow
* Time series support
* Data drift monitoring
* Dash + Streamlit frontend

## ✅ Current Status (as of v0.2.3)

| Feature                                | Status           |
| -------------------------------------- | ---------------- |
| Linearity, Homoscedasticity, Normality | ✅ Complete       |
| Multicollinearity, Influence           | ✅ Complete       |
| Outliers, Independence, Missingness    | 🚧 In Progress   |
| Unified `model_wrapper` abstraction    | ✅ Complete       |
| Report generation                      | 🚧 MVP sketching |
| CI, Testing framework                  | 🚧 Initial setup |
| Plugin support                         | ⬜ Not started    |

## 📈 Future Roadmap

See GitHub Projects: [https://github.com/W-Thurston/dsbf/projects](https://github.com/W-Thurston/dsbf/projects)

---
