import pandas as pd
from src.config import DATA_FILE


def load_data():
    df = pd.read_csv(DATA_FILE, sep=";")
    print(f"[INFO] Data loaded: {df.shape}")
    return df