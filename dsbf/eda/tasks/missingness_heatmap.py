# dsbf/eda/tasks/missingness_heatmap.py

import os

import matplotlib.pyplot as plt

from dsbf.utils.backend import is_polars


def missingness_heatmap(df, output_dir="outputs"):
    try:
        import missingno as msno

        if is_polars(df):
            df = df.to_pandas()
        fig_path = os.path.join(output_dir, "missingness_heatmap.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        msno.heatmap(df)
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        return {"image": fig_path}
    except Exception as e:
        return {"error": f"missingness_heatmap failed: {e}"}
