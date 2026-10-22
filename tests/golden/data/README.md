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

`SMSSpamCollection` is the UCI [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
(v.1, from `sms+spam+collection.zip`), byte for byte: 5,574 messages, one per line, a label
(`ham` or `spam`) and the text separated by a tab, with no header. The golden datasets
write it as a CSV with the header `label,text` (`tests/golden/datasets.py`). It is the
suite's only free text: 187 messages have leading or trailing whitespace, 35 an invisible
character, 483 a non-ASCII character, and 403 repeat an earlier one.

- SHA-256: `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), stated on the dataset
  page, which allows redistribution with attribution.
- Attribution: Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection \[Dataset\]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5CC84. The messages come from the
  Grumbletext site, Caroline Tagg's PhD thesis, the NUS SMS Corpus and the SMS Spam Corpus
  v.0.1 Big, as the dataset's readme lists.
