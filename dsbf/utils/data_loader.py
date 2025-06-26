# dsbf/utils/data_loader.py
"""
Data Loader utility.

Provides flexible dataset loading for local files, sklearn built-ins,
seaborn demos, and OpenML, with backend-agnostic support for pandas and polars.
"""
import inspect

import pandas as pd
import seaborn as sns
from sklearn import datasets as sklearn_datasets
from sklearn.datasets import fetch_openml


def load_dataset(
    name: str = "iris",
    source: str = "sklearn",
    as_frame: bool = True,
    backend: str = "pandas",
):
    """
    Load a standard dataset for testing or demonstration.

    Parameters:
        name (str): The name of the dataset.
        source (str): One of 'sklearn', 'seaborn', or 'openml'.
        as_frame (bool): Whether to return as a pandas DataFrame.
        backend (str): pandas or polars

    Returns:
        pd.DataFrame: The requested dataset.
    """
    if source == "sklearn":
        if hasattr(sklearn_datasets, f"load_{name}"):
            loader = getattr(sklearn_datasets, f"load_{name}")
            data = loader(as_frame=True)
            df = (
                data.frame
                if hasattr(data, "frame")
                else pd.DataFrame(data.data, columns=data.feature_names)
            )
        else:
            raise ValueError(f"Scikit-learn dataset 'load_{name}' not found.")

    elif source == "seaborn":
        try:
            df = sns.load_dataset(name)
        except Exception:
            raise ValueError(f"Seaborn dataset '{name}' not found.")

    elif source == "openml":
        try:
            df = fetch_openml(name, version=1, as_frame=True).frame
        except Exception:
            raise ValueError(f"OpenML dataset '{name}' not found or failed to load.")

    else:
        raise ValueError(f"Unsupported source: {source}")

    if backend == "polars":
        try:
            import polars as pl

            return pl.from_pandas(df)
        except ImportError:
            print("[DataLoader] Polars not installed. Falling back to pandas.")
            return df

    return df


def list_available_datasets(source: str = "sklearn"):
    if source == "sklearn":
        return sorted(
            name.replace("load_", "")
            for name, func in inspect.getmembers(sklearn_datasets, inspect.isfunction)
            if name.startswith("load_")
        )
    elif source == "seaborn":
        return sns.get_dataset_names()
    elif source == "openml":
        return ["adult", "titanic", "bank-marketing", "mnist_784"]
    else:
        raise ValueError(f"Unsupported source: {source}")
