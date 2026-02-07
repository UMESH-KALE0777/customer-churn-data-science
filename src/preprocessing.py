import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def prepare_features(df):
    X = df.drop(columns=["churn_label"])
    y = df["churn_label"]

    leakage_cols = [
        "customer_id",
        "customer_status",
        "churn_score",
        "churn_category",
        "churn_reason"
    ]
    X = X.drop(columns=leakage_cols)

    X = pd.get_dummies(X, drop_first=True)
    return X, y
