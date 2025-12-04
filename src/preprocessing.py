from __future__ import annotations
from typing import Tuple, List
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Dynamic motion data + static dimensions
NUMERIC_FEATURES: List[str] = [
    "sog",      
    "cog",      
    "heading",  
    "width",    
    "length",   
    "draught"   
]

# Categorical feature
CATEGORICAL_FEATURES: List[str] = [
    "shiptype" 
]

# Target variable to predict
TARGET_COLUMN: str = "navigationalstatus"


def load_raw_ais(path: str) -> pd.DataFrame:
    """
    Reads the raw AIS data from CSV and drops unnecessary identifier columns.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame without ID columns.

    Notes:
        - We drop `mmsi` because it uniquely identifies a vessel (and encodes country/type).
        - `Unnamed: 0` is dropped as it is an artifact of CSV saving.
    """
    df = pd.read_csv(path)
    
    cols_to_drop = ["Unnamed: 0", "mmsi"]

    # Drop columns that actually exist in the dataframe
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)
    return df


def filter_physical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-based sanity checks to remove physically impossible or 
    highly unlikely data points.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame with outliers removed.

    Rules applied:
        - `sog` (Speed): [0, 60] knots. (Most ships < 25, high-speed < 50)
        - `cog` (Course) & `heading`: [0, 360] degrees.
        - `width`: (0, 80] meters. (Largest ships ~60-70m wide)
        - `length`: (0, 400] meters. (Largest ships ~400m long)
        - `draught`: (0, 25] meters. (Deepest draughts ~24m)
    """
    df_out = df.copy()
    
    # Speed Over Ground (0-60 knots)
    if "sog" in df_out.columns:
        df_out = df_out[
            (df_out["sog"] >= 0) & 
            (df_out["sog"] <= 60)
        ]
        
    # Course Over Ground (0-360 degrees)
    if "cog" in df_out.columns:
        df_out = df_out[
            (df_out["cog"] >= 0) & 
            (df_out["cog"] <= 360)
        ]
        
    # Heading (0-360 degrees)
    if "heading" in df_out.columns:
        df_out = df_out[
            (df_out["heading"] >= 0) & 
            (df_out["heading"] <= 360)
        ]
        
    # Width (0-80 meters)
    if "width" in df_out.columns:
        df_out = df_out[
            (df_out["width"] > 0) & 
            (df_out["width"] <= 80)
        ]
        
    # Length (0-400 meters)
    if "length" in df_out.columns:
        df_out = df_out[
            (df_out["length"] > 0) & 
            (df_out["length"] <= 400)
        ]
        
    # Draught (0-25 meters)
    if "draught" in df_out.columns:
        df_out = df_out[
            (df_out["draught"] > 0) & 
            (df_out["draught"] <= 25)
        ]
    return df_out


def clean_types_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes data types and handles essential missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cleaned types and no missing targets.
    """
    df_clean = df.copy()
    
    # Coerce numeric columns
    for col in NUMERIC_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            
    # Clean categorical columns (string conversion + strip)
    if "shiptype" in df_clean.columns:
        df_clean["shiptype"] = df_clean["shiptype"].astype("string").str.strip()
    if TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype("string").str.strip()
        
    # Drop missing critical values (target or shiptype)
    # We cannot impute the target, and `shiptype` is a key feature.
    cols_to_check = [c for c in [TARGET_COLUMN, "shiptype"] if c in df_clean.columns]
    if cols_to_check:
        df_clean = df_clean.dropna(subset=cols_to_check)
    return df_clean


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the data into the feature matrix X and target vector y.

    Args:
        df (pd.DataFrame): Cleaned and filtered DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
            X: Contains numeric and categorical features.
            y: Contains the target labels.
    """
    # Select only columns that exist in the dataframe
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[features].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def drop_rare_classes(
    X: pd.DataFrame,
    y: pd.Series,
    min_samples: int = 50,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Removes classes from the dataset that have fewer than `min_samples` samples.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        min_samples (int): Minimum number of samples required to keep a class.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Filtered (X, y) with rare classes removed.

    Notes:
        - This step makes the classification problem more learnable by focusing
          on classes with sufficient representation.
        - It also avoids instability in stratified train/validation/test splits
          when classes have only 1–2 samples (which would cause sklearn to fail).
    """
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < min_samples].index.tolist()

    if rare_classes:
        print(f"[drop_rare_classes] Dropping {len(rare_classes)} rare classes with < {min_samples} samples: {rare_classes}")

    mask = ~y.isin(rare_classes)
    X_filtered = X[mask].copy()
    y_filtered = y[mask].copy()
    return X_filtered, y_filtered


def train_val_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits data into Stratified Train, Validation, and Test sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Fraction of data for the test set.
        val_size (float): Fraction of data for the validation set.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple containing:
        (X_train, X_val, X_test, y_train, y_val, y_test)

    Note:
        We use stratified splitting to ensure the class distribution of
        `navigationalstatus` is preserved across all three splits,
        crucial for handling class imbalance. However, if any class has
        fewer than 2 samples in the subset being split, stratification
        is not possible and will be disabled for that step.
    """

    def _can_stratify(target: pd.Series) -> bool:
        """Return True if every class has at least 2 samples."""
        class_counts = Counter(target)
        return min(class_counts.values()) >= 2

    # First split: Separate train from (val + test)
    # The size of temp is (val_size + test_size)
    temp_size = val_size + test_size
    stratify_first = y if _can_stratify(y) else None
    if stratify_first is None:
        print(
            "[train_val_test_split] Warning: disabling stratification for first split "
            "because at least one class has < 2 samples."
        )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        stratify=stratify_first,
        random_state=random_state,
    )
    
    # Second split: Separate Val from Test
    # We need to calculate the proportion of `test` relative to `temp`
    # relative_test_size = test_size / (val_size + test_size)
    # Example: if `test=0.2`, `val=0.1` -> `temp=0.3`.
    # `relative_test_size = 0.2 / 0.3 = 0.66...`
    relative_test_size = test_size / temp_size
    stratify_second = y_temp if _can_stratify(y_temp) else None
    if stratify_second is None:
        print(
            "[train_val_test_split] Warning: disabling stratification for second split "
            "because at least one class has < 2 samples in the temp set."
        )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=stratify_second,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """
    Creates an sklearn ColumnTransformer for preprocessing.

    Returns:
        ColumnTransformer: Transformer that:
            - Standardizes numeric features (StandardScaler).
            - One-hot encodes categorical features (OneHotEncoder).
            - Ignores unknown categories during transform.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        # Drop columns not specified in transformers (if any)
        # though we likely already filtered `X` to only relevant columns.
        remainder="drop" 
    )
    
    return preprocessor


def prepare_ais_data(path: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, 
    pd.Series, pd.Series, pd.Series, 
    ColumnTransformer
]:
    """
    End-to-end data preparation helper.

    1. Load raw CSV.
    2. Drop `mmsi` and `Unnamed: 0` columns.
    3. Apply domain-based outlier filters.
    4. Clean types and basic missing values.
    5. Build X (features) and y (target).
    6. Drop ultra-rare classes (< 50 samples).
    7. Stratified split into train/val/test.
    8. Build preprocessing ColumnTransformer.

    Args:
        path (str): Path to the AIS data CSV.

    Returns:
        Tuple containing:
        (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    # 1. Load & 2. Drop IDs
    df = load_raw_ais(path)
    
    # 3. Filter outliers
    df = filter_physical_outliers(df)
    
    # 4. Clean types & missing
    df = clean_types_and_missing(df)
    
    # 5. Features & Target
    X, y = get_features_and_target(df)
    
    # 6. Drop rare classes
    X, y = drop_rare_classes(X, y, min_samples=50)
    
    # 7. Splits
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    
    # 8. Preprocessor
    preprocessor = build_preprocessor()
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

print("Compilation complete")