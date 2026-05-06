import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COLUMN, RANDOM_STATE


def create_target(df):
    df = df.copy()
    df["target"] = (df[TARGET_COLUMN] >= 10).astype(int)
    return df


def preprocess_data(df):
    df = create_target(df)

    # ❗ уникнення leakage
    df = df.drop(columns=["G3"])

    # encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test