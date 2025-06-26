---

## ✅ README.md

# Data Scientist's Best Friend (DSBF)

**A transparent, modular AutoML diagnostics pipeline built for humans.**

DSBF is your one-stop-shop for rigorously checking linear model assumptions, profiling datasets, and generating clear, human-friendly reports. Designed for transparency, reproducibility, and extensibility — it’s a white-box alternative to tools like ydata-profiling, with the added bonus of helping you *learn* along the way.

![DSBF Report Example](report_examples/sample.png) <!-- Optional visual -->

---

### 🚀 Features

* 📊 **Assumption Checking**: Linearity, Homoscedasticity, Normality, Multicollinearity, Independence, Outliers, Influence, Missingness
* ⚙️ **Modular Profiling Pipeline**: Each check is a plug-and-play module
* 🧠 **Verbosity Control**: Choose the depth of explanation and diagnostics shown
* 🔍 **Polars + Pandas Backend Support**
* 🧪 **Built-In Testing** and CI for every module
* 🧰 **YAML-Configurable Runs** for reproducibility
* 📄 **Markdown/HTML Report Output** with sample visualizations

---

### 📦 Installation

```bash
# Coming soon via PyPI
pip install dsbf
```

---

### 🧠 Who It’s For

| Role          | Use Case                                           |
| ------------- | -------------------------------------------------- |
| Beginner      | Learn what each diagnostic means                   |
| Mid-Level DS  | Catch modeling mistakes before they cost you       |
| Senior/Expert | Use and extend a modular, testable profiling stack |

---

### 📂 Quickstart

```bash
# Step 1: Clone repo
https://github.com/W-Thurston/dsbf.git

# Step 2: Run on toy dataset
python run_dsbf.py --config config/example.yaml
```

See [docs/](./docs/) for usage examples, output previews, and developer notes.

---

### 📈 Roadmap

* [x] Core assumption modules
* [x] Unified model wrapper abstraction
* [ ] Visual report generator
* [ ] Plugin architecture
* [ ] CLI-free dashboard preview mode
* [ ] Monitoring support
* [ ] Explainability extensions

---

### 🤝 Contributing

We love contributions — from fixing typos to writing new modules. See [`CONTRIBUTING.md`](./CONTRIBUTING.md).

---

### 📄 License

MIT License. See [`LICENSE`](./LICENSE).

---
