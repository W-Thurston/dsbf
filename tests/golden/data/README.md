# Vendored golden-run data

`titanic.csv` is the Titanic passenger table from the
[seaborn-data](https://github.com/mwaskom/seaborn-data) repository
(`titanic.csv`, last changed upstream in commit `a29a014` on 2014-03-21). It is a copy of what
`seaborn.load_dataset("titanic")` downloads, kept here so the golden baselines don't need the
network and can't change when upstream does.

- 891 rows, 15 columns.
- SHA-256: `81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2`.
- seaborn-data has no LICENSE file and says that some of its datasets are modified from their
  sources. The passenger list is widely redistributed (R, Kaggle, scikit-learn's OpenML copy),
  but this repository can't confirm a license for this particular copy. It is used only as a
  small test fixture.

The eight archetype frames and `kitchen5k` are not committed: `tests/generators` rebuilds them
from fixed seeds on every run.
