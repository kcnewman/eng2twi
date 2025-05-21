import numpy as np
import pandas as pd
import re
import pickle
import unicodedata
import os


def load_dictionary(file_dir: str, pos: bool = True):
    """Loads English to Twi dictionary from disk

    Args:
        pos (bool, optional): Drop part of speech column in dataset. Defaults to True.

    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(file_dir)
    df = df.dropna(subset=["english", "twi"])
    if pos:
        df = df.drop("pos", axis=1)
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


def save_matrix(matrix, name, directory="../models"):
    path = os.path.join(directory, f"{name}.npy")
    np.save(path, matrix)
    print(f"Saved matrix to: {path}")


def load_matrix(name, directory="../models"):
    path = os.path.join(directory, f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}. Run save_matrix()")
    matrix = np.load(path)
    print(f"Loaded matrix from: {path}")
    return matrix


def save_dict(obj, name, directory="../models"):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{name}.pkl"), "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved dict to: models/{name}.pkl")


def load_dict(name, directory="../models"):
    path = os.path.join(directory, f"{name}.pkl")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded dict from: {path}")
    return obj
