import pandas as pd


def get_categorical_columns(X: pd.DataFrame):
    return [
        cname
        for cname in X.columns
        if X[cname].nunique() < 10 and X[cname].dtype == "object"
    ]


def get_numeric_columns(X: pd.DataFrame):
    return [cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]]
