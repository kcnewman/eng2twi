import tqdm
import numpy as np
from sklearn.preprocessing import normalize


def load_embeddings(file_path: str, fasttext: bool = False):
    """Load Fasttext.vec or Glove.txt word embeddings

    Args:
        file_path (str): Path to embedding file
        fasttext (bool, optional): Placeholder for fasttext file type. Defaults to False.

    Returns:
        Dict: Word embeddings
    """
    embeddings = {}
    if fasttext:
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split()
            vocab, dim = int(header[0]), int(header[1])

            for i in tqdm.tqdm(range(vocab)):
                line = f.readline().strip().split()
                word = line[0]
                vector = np.array(line[1:], dtype="float32")
                embeddings[word] = vector
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm.tqdm(f)):
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype="float32")
                embeddings[word] = vector
    print(f"Loaded embeddings from {file_path}...")
    return embeddings


def phrase_embeddings(phrase, embeddings):
    words = phrase.split()
    vecs = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else None


def build_matrices(dictionary, en_embeddings, twi_embeddings):
    X = []
    Y = []

    for en_word, twi_phrase in dictionary.items():
        if en_word not in en_embeddings:
            continue
        twi_vec = phrase_embeddings(twi_phrase, twi_embeddings)
        if twi_vec is not None:
            X.append(en_embeddings[en_word])
            Y.append(twi_vec)
    X = normalize(np.vstack(X), axis=1)
    Y = normalize(np.vstack(Y), axis=1)
    return X, Y
