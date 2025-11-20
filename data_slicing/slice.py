import pandas as pd


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)


def slice_dataframe(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Slices the given DataFrame from the start index to the end index.

    Parameters:
    df (pd.DataFrame): The DataFrame to slice.
    start (int): The starting index for the slice.
    end (int): The ending index for the slice.

    Returns:
    pd.DataFrame: The sliced DataFrame.
    """
    return df.iloc[start:end]


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides descriptive statistics of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to describe.

    Returns:
    pd.DataFrame: A DataFrame containing descriptive statistics.
    """
    return df.describe()


def identify_categorical_variables(df: pd.DataFrame) -> dict:
    """
    Identifies categorical and numeric variables in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    dict: Dictionary with 'categorical' and 'numeric' column lists.
    """
    # Get data types
    print("Data types:")
    print(df.dtypes)
    print("\n" + "=" * 50 + "\n")

    # Identify categorical columns (object/string type or low cardinality)
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            categorical_cols.append(col)
        elif df[col].dtype in ["int64", "float64"]:
            # Check if numeric column has low cardinality (might be categorical)
            unique_values = df[col].nunique()
            if unique_values < 10:  # Threshold for potential categorical
                print(
                    f"Column '{col}' is numeric but has only {unique_values} unique values:"
                )
                print(f"Values: {sorted(df[col].unique())}")
                print("Consider if this should be treated as categorical.\n")
            numeric_cols.append(col)

    print(f"Categorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    print("\nSample of categorical data:")
    for col in categorical_cols:
        print(f"{col}: {df[col].unique()[:5]}")  # Show first 5 unique values

    return {"categorical": categorical_cols, "numeric": numeric_cols}


def descriptive_stats_by_category(df: pd.DataFrame, categorical_col: str) -> dict:
    """
    Outputs descriptive statistics for each numeric feature grouped by categorical variable.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    categorical_col (str): The name of the categorical column to group by.

    Returns:
    dict: Dictionary containing descriptive stats for each category.
    """
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    print(
        f"Descriptive statistics for numeric features grouped by '{categorical_col}':"
    )
    print("=" * 70)

    # Group by categorical variable and calculate stats
    grouped_stats = {}

    for category in df[categorical_col].unique():
        print(f"\nCategory: {category}")
        print("-" * 40)

        # Filter data for this category
        category_data = df[df[categorical_col] == category][numeric_cols]

        # Calculate descriptive statistics
        stats = category_data.describe()
        grouped_stats[category] = stats

        print(stats)
        print()

    return grouped_stats


if __name__ == "__main__":
    # Example usage
    df = load_dataframe("data/iris.data")
    # sliced = slice_dataframe(df, 2, 5)
    # print(sliced)

    print("\nFirst few rows:")
    print(df.head())
    print("\n" + "=" * 50 + "\n")

    # Identify variable types
    var_types = identify_categorical_variables(df)
    print("Variable types:\n", var_types)
    print("\n" + "=" * 70 + "\n")

    # Get descriptive stats by category (this is what the exercise asks for!)
    if var_types["categorical"]:
        categorical_col = var_types["categorical"][0]  # Use first categorical column
        grouped_stats = descriptive_stats_by_category(df, categorical_col)

    print("\n" + "=" * 50 + "\n")
    description = describe_dataframe(df)
    print("Descriptive statistics:\n", description)
