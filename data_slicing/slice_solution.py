import pandas as pd


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    # Load with proper column names for iris dataset
    if "iris" in file_path.lower():
        column_names = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "class",
        ]
        return pd.read_csv(file_path, header=None, names=column_names)
    else:
        return pd.read_csv(file_path)


def slice_iris(df, feature: str):
    """Function for calculating descriptive stats on slices of the Iris dataset."""
    for cls in df["class"].unique():
        df_temp = df[df["class"] == cls]
        mean = df_temp[feature].mean()
        stddev = df_temp[feature].std()
        print(f"Class: {cls}")
        print(f"{feature} mean: {mean:.4f}")
        print(f"{feature} stddev: {stddev:.4f}")
    print()


if __name__ == "__main__":
    df = load_dataframe("./data/iris.data")
    slice_iris(df, "sepal_length")
    slice_iris(df, "sepal_width")
    slice_iris(df, "petal_length")
    slice_iris(df, "petal_width")
