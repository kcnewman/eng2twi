import numpy as np
import pandas as pd
import re
import unicodedata


def load_dict(file_path: str, drop_pos: bool = True):
    """Loads english to twi dictionary

    Args:
        file_path (str): Path to dictionary file
        drop_pos: Drop part of speech column

    Returns:
        df: pd.DataFrame
    """
    df = pd.read_csv(file_path)
    if bool:
        df = df.drop(columns=["pos"])
    return df
