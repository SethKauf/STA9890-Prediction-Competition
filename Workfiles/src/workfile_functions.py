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

def get_columns_by_dtype(df, dtypes):
    """
    Returns a list of column names from df where the dtype matches any in dtypes.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        dtypes (list): A list of dtype strings to filter by (e.g., ['object'], ['int64', 'float64']).
    
    Returns:
        list: Column names with matching dtypes.
    """
    return df.select_dtypes(include=dtypes).columns.tolist()

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

# Read in csv files from site
def read_gh(URL, f):
    return pd.read_csv(URL + f)

def clean_building_dfs(df):
    tfcols = ['elevator','has_cooling','has_heat']
    for col in tfcols:
        df[col] = df[col].astype('Int64')

    quality_map = {
    'F':0,
    'E':1,
    'D':2,
    'C':3,
    'B':4,
    'A':5,
    'X':6
    }

    df['quality'] = df['quality'].map(quality_map)

    quality_description_map = {
        'Poor':0,
        'Very Low':1,
        'Low':2,
        'Average':3,
        'Good':4,
        'Excellent':5,
        'Superior':6
    }

    df['quality_description'] = df['quality_description'].map(quality_description_map)

    bc_map = {
        'Unsound':0,
        'Very Poor':1,
        'Poor':2,
        'Fair':3,
        'Average':4,
        'Good':5,
        'Very Good':6,
        'Excellent':7
    }

    df['building_condition'] = df['building_condition'].map(bc_map)

    grade_map = {
        'E-':1.3,
        'E':1.6,
        'E+':1.9,
        'D-':2.3,
        'D':2.6,
        'D+':2.9,
        'C-':3.3,
        'C':3.6,
        'C+':3.9,
        'B-':4.3,
        'B':4.6,
        'B+':4.9,
        'A-':5.3,
        'A':5.6,
        'A+':5.9,
        'X-':6.3,
        'X':6.6,
        'X+':6.9
    }

    df['grade'] = df['grade'].map(grade_map)

    pc_map = {
        'Unsound':0,
        'Very Poor':1,
        'Poor':2,
        'Fair':3,
        'Average':4,
        'Good':5,
        'Very Good':6,
        'Excellent':7
    }
    
    df['physical_condition'] = df['physical_condition'].map(pc_map)

    df.loc[df['foundation_type'] == 'Basement and Basement', 'foundation_type'] = 'Basement'

    df.loc[(df['foundation_type']=='Basement and Slab')|(df['foundation_type']=='Crawl Space and Slab')|(df['foundation_type']=='Basement and Crawl Space')|
           (df['foundation_type']=='Basement and Pier and Beam')|(df['foundation_type']=='Pier and Beam')|
           (df['foundation_type']=='Pier and Beam and Pier and Beam')|(df['foundation_type']=='Pier and Beam and Slab'),'foundation_type'] = 'Mixed'

    dummies = pd.get_dummies(df['foundation_type'], prefix='foundation').astype('Int64')
    df = pd.concat([df, dummies], axis=1)
        
    df = group_rare_categories(df, 'exterior_walls', threshold=1000, new_label='Other')

    # Keywords to extract
    keywords = [
        'Brick Veneer',
        'Brick Masonry',
        'Concrete Block',
        'Vinyl',
        'Stucco',
        'Stone',
        'Other'
    ]

    # Create 1/0 dummy columns based on substring presence
    for key in keywords:
        col_name = key.replace(' ', '_').lower()  # e.g., 'Brick Veneer' → 'brick_veneer'
        df[col_name] = df['exterior_walls'].fillna('').str.contains(key).astype('int8')

    df = df.drop(['exterior_walls','foundation_type'],axis=1)

    return df