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


def clean_text(text: str, lang: str = "ak"):
    """Preprocess text

    Args:
        text (str): corpus
        lang (str, optional): language of corpus. Defaults to 'ak'.

    Returns:
        str: processed corpus
    """
    if lang == "ak":

        def strip_accents(text):
            return "".join(
                t
                for t in unicodedata.normalize("NFD", text)
                if unicodedata.category(t) != "Mn" or t in "ɛɔƐƆŋŊ"
            )

        chars = "a-zA-ZɛɔƐƆŋŊ'"
        text = strip_accents(text)
        text = text.lower()
        text = re.sub(rf"[^{chars}\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    else:
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s]", "", text)
        return text
