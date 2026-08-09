# scripts/export_latex.py
import os
import pandas as pd


def generate_latex_table(
    metrics_dict: dict, output_filepath: str = "latex/metrics_table.tex"
):
    """Exports model evaluation metrics directly into a LaTeX table format."""
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "Model Architecture"
    df.reset_index(inplace=True)

    # Format column names to uppercase for academic presentation
    df.columns = [str(col).upper() for col in df.columns]

    latex_string = df.to_latex(
        index=False,
        caption="Comparative Performance Evaluation of Monsoon Prediction Models",
        label="tab:model_performance",
        float_format="%.4f",
        column_format="|l|c|c|c|",
        escape=True,
    )

    # Automatically create the destination directory if it doesn't exist
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(latex_string)

    print(f"Successfully exported dynamic LaTeX table to {output_filepath}")


# Example Execution
if __name__ == "__main__":
    sample_metrics = {
        "XGBoost Baseline": {"MAE": 2.14, "RMSE": 3.82, "R2": 0.842},
        "CNN-LSTM Hybrid": {"MAE": 1.45, "RMSE": 2.11, "R2": 0.915},
        "Temporal Fusion Transformer": {"MAE": 1.12, "RMSE": 1.78, "R2": 0.948},
    }
    generate_latex_table(sample_metrics)