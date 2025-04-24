import pandas as pd

def binary_labeler(data, data_cols, one_var, zero_var, suffix='_bin'):
    df = data.copy()
    for col in data_cols:
        new_col = col + suffix
        df[new_col] = df[col].apply(lambda x: 1 if x == one_var else (0 if x == zero_var else x))
    return df


def checkpoint(df, existing_copy=None):
    """
    Prompts the user to confirm saving a copy of the DataFrame.
    Returns the copied DataFrame if confirmed, else returns the previous copy (if any).

    Args:
        df (pd.DataFrame): The DataFrame to potentially save
        existing_copy (pd.DataFrame or None): An existing backup copy to retain if skipped

    Returns:
        pd.DataFrame: Either a new copy (if saved), or the previous copy (if skipped)
    """
    print(f"\nDo you want to save a copy of the DataFrame?")
    choice = input("Press 'y' to confirm, or any other key to skip: ").strip().lower()

    if choice == 'y':
        print("DF Saved.")
        return df.copy()
    else:
        print("Skipped.")
        return existing_copy



def count_values(df,value:str):
    
    """
    Returns printout of whatever character you're looking for across the dataset
    """
    
    total_rows = len(df)
    val_counts = []

    for col in df.columns:
        count = (df[col] == value).sum()
        if count > 0:
            percent = (count / total_rows) * 100
            val_counts.append((col, count, percent))

    # Sort by count descending
    val_counts.sort(key=lambda x: x[1], reverse=True)

    # Print results
    print(f"{'Column':<30} {'? Count':>15} {'% of Rows':>12}")
    print("-" * 60)
    for col, count, percent in val_counts:
        print(f"{col:<30} {count:>15,} {percent:>11.2f}%")
        
def group_rare_categories(df, column, threshold=100, new_label='Other'):
    """
    Replace categories in a column that appear fewer than `threshold` times with `new_label`.
    
    Args:
        df (pd.DataFrame): DataFrame to process
        column (str): Name of the column to transform
        threshold (int): Frequency threshold below which values are replaced
        new_label (str): Label to replace rare categories with

    Returns:
        pd.DataFrame: Updated DataFrame with rare values grouped
    """
    value_counts = df[column].value_counts()
    rare_values = value_counts[value_counts < threshold].index
    df[column] = df[column].apply(lambda x: new_label if x in rare_values else x)
    return df

def count_nulls(df):
    total_rows = len(df)
    null_counts = []

    for col in df.columns:
        count = df[col].isna().sum()
        if count > 0:
            percent = (count / total_rows) * 100
            null_counts.append((col, count, percent))

    # Sort descending
    null_counts.sort(key=lambda x: x[1], reverse=True)

    # Print nicely
    print(f"{'Column':<30} {'NULL Count':>15} {'% of Rows':>12}")
    print("-" * 60)
    for col, count, percent in null_counts:
        print(f"{col:<30} {count:>15,} {percent:>11.2f}%")

def collapse_rare_categories(df, col, min_count=100):
    freq = df[col].value_counts()
    rare_vals = freq[freq < min_count].index
    df[col] = df[col].where(~df[col].isin(rare_vals), other='Other')
    return df