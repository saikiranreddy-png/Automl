# preprocess.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def make_ohe():
    # OneHotEncoder arg name changed across sklearn versions; handle both.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X: pd.DataFrame):
    """
    Returns (preprocessor, num_cols, cat_cols).
    preprocessor: ColumnTransformer that handles numeric and categorical columns,
    imputes missing values, scales numeric, and one-hot encodes categorical strings.
    """
    num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","string","bool"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe())
    ])

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", num_pipeline, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipeline, cat_cols))

    if len(transformers) == 0:
        # No numeric or categorical columns: pass through (unlikely)
        preprocessor = ColumnTransformer([], remainder="passthrough")
    else:
        preprocessor = ColumnTransformer(transformers, remainder="drop")

    return preprocessor, num_cols, cat_cols
