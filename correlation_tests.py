"""Script to determine the correlation between the human determined similarity matrix and all the other similarity matrices."""

from glob import glob
import pandas as pd
import numpy as np
import mantel
from sanitize_ml_labels import sanitize_ml_labels


def execute_array_correlation_tests(
    human_df: pd.DataFrame,
    model_df: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Compute correlation metrics (Spearman, Pearson, Kendall's Tau) between two similarity matrices,
    excluding diagonal elements. Returns a DataFrame with results.

    Args:
        human_df (pd.DataFrame): Human similarity matrix.
        model_df (pd.DataFrame): Model similarity matrix.
        name (str): Name of the model.

    Returns:
        pd.DataFrame: Correlation results.
    """
    from scipy.stats import spearmanr, pearsonr, kendalltau
    import numpy as np

    # We convert the DataFrames to numpy arrays removing the
    # diagonal elements, maintaining both the upper and lower triangular parts
    # as the human matrix is not necessarily symmetric
    human_array = np.array(
        [
            human_df.values[i, j]
            for i in range(len(human_df))
            for j in range(len(human_df))
            if i != j
        ]
    )
    model_array = np.array(
        [
            model_df.values[i, j]
            for i in range(len(model_df))
            for j in range(len(model_df))
            if i != j
        ]
    )

    # Calculate the correlation coefficients and p-values
    spearman_corr, spearman_p = spearmanr(human_array, model_array)
    pearson_corr, pearson_p = pearsonr(human_array, model_array)
    kendall_corr, kendall_p = kendalltau(human_array, model_array)

    # Create a DataFrame to hold the results
    return pd.DataFrame(
        {
            "Metric": ["Spearman", "Pearson", "Kendall's Tau"],
            "Correlation": [spearman_corr, pearson_corr, kendall_corr],
            "P-value": [spearman_p, pearson_p, kendall_p],
            "Model": name,
            "Number of samples": len(human_array),
        }
    )


def execute_matrix_correlation_tests(
    human_df: pd.DataFrame,
    model_df: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Compute Mantel test correlations (Spearman, Pearson) between symmetric distance matrices
    derived from human and model similarity matrices. Returns a DataFrame with results.

    Args:
        human_df (pd.DataFrame): Human similarity matrix.
        model_df (pd.DataFrame): Model similarity matrix.
        name (str): Name of the model.

    Returns:
        pd.DataFrame: Mantel test results.
    """

    # The human matrix is not necessarily symmetric, so we make it symmetric
    symmetric_human_df = (human_df + human_df.T) / 2

    # Due to float errors, also the model matrix might not be perfectly symmetric
    model_df = (model_df + model_df.T) / 2

    # We ensure that the diagonals are exactly 1.0
    np.fill_diagonal(symmetric_human_df.values, 1.0)
    np.fill_diagonal(model_df.values, 1.0)

    # We convert the two similarity matrices into distance matrices
    symmetric_human_df_distance = 1.0 - symmetric_human_df
    model_df_distance = 1.0 - model_df

    test_results = []

    for method_name in ["spearman", "pearson"]:
        mantel_test_result = mantel.test(
            symmetric_human_df_distance, model_df_distance, method=method_name
        )

        test_results.append(
            {
                "Metric": f"Mantel {sanitize_ml_labels(method_name)}",
                "Correlation": mantel_test_result[0],
                "P-value": mantel_test_result[1],
                "Model": name,
                "Number of samples": symmetric_human_df.shape[0],
            }
        )

    return pd.DataFrame(test_results)


def execute_correlation_tests(
    human_df: pd.DataFrame, model_df: pd.DataFrame, name: str
) -> pd.DataFrame:
    """
    Run all correlation tests (array-based and matrix-based) between human and model similarity matrices.
    Returns a concatenated DataFrame of all results.

    Args:
        human_df (pd.DataFrame): Human similarity matrix.
        model_df (pd.DataFrame): Model similarity matrix.
        name (str): Name of the model.

    Returns:
        pd.DataFrame: All correlation test results.
    """
    labelled_model_df = model_df.loc[human_df.index, human_df.columns]
    return pd.concat(
        [
            execute_array_correlation_tests(human_df, labelled_model_df, name),
            execute_matrix_correlation_tests(human_df, labelled_model_df, name),
        ],
        axis=0,
    )


if __name__ == "__main__":
    # Main script to run correlation tests between human and model similarity matrices.
    human_determined_file_name: str = "scores/expert_human.csv"
    human_determined: pd.DataFrame = (
        pd.read_csv(human_determined_file_name, index_col=0) / 10.0
    )
    test_results = []
    for file_name in glob("scores/*.csv"):
        if file_name == human_determined_file_name:
            continue
        df: pd.DataFrame = pd.read_csv(file_name, index_col=0)
        name: str = sanitize_ml_labels(file_name.split("/")[-1].split(".")[0])
        test_results.append(execute_correlation_tests(human_determined, df, name))

    test_results_df: pd.DataFrame = pd.concat(test_results, axis=0)

    # We sort the dataframe by Correlation descending and then by P-value ascending
    test_results_df = test_results_df.sort_values(
        by=["Correlation", "P-value"], ascending=[False, True]
    )

    test_results_df.to_csv("correlation_tests_results.csv", index=False)

    # Print the results to the console
    print(test_results_df)
