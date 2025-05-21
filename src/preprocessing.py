import numpy as np
import pandas as pd
import re
import unicodedata


def entoak(file_path: str):
    """Loads english to twi dictionary

    Args:
        file_path (str): Path to dictionary file

    Returns:
        df: pd.DataFrame
    """
    df = pd.read_csv(file_path)
    df = df.drop(columns=["pos"])
    return df


fp = "../data/raw/twi_dict.csv"
df = entoak(file_path=fp)

print(df.head())

entoak()
