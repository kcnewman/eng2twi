import numpy as np
import pandas as pd
import re
import unicodedata


def load_dictionary(pos: bool = True):
    """Loads English to Twi dictionary from disk

    Args:
        pos (bool, optional): Drop part of speech column in dataset. Defaults to True.

    Returns:
        pd.DataFrame
    """
    file_dir = "../data/raw/twi_dict.csv"
    df = pd.read_csv(file_dir)
    if pos:
        df = df.drop("pos", axis=1)
    else:
        pass
    return df
