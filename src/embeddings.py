import tqdm
import numpy as np
import os
import pickle
from sklearn.preprocessing import normalize


def load_embeddings(file_path: str, fasttext: bool = False, from_pickle: bool = False):
    """
    Load word embeddings from a .vec/.txt file or a .pkl file.

    Args:
        file_path (str): Path to the embedding file.
        fasttext (bool, optional): If True, parse as FastText format. Defaults to False.
        from_pickle (bool, optional): If True, load from pickle file. Defaults to False.

    Returns:
        dict: Dictionary of word embeddings.
    """
    if from_pickle:
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Loaded embeddings from pickle: {file_path}")
        return embeddings

    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        if fasttext:
            header = f.readline().strip().split()
            vocab, dim = int(header[0]), int(header[1])
            for _ in tqdm.tqdm(range(vocab)):
                line = f.readline().strip().split()
                word = line[0]
                vector = np.array(line[1:], dtype="float32")
                embeddings[word] = vector
        else:
            for line in tqdm.tqdm(f):
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype="float32")
                embeddings[word] = vector

    print(f"Loaded embeddings from text: {file_path}")
    return embeddings


def save_embedding(en_embeddings, tw_embeddings, directory="../models"):
    with open(os.path.join(directory, "english_vecs.pkl"), "wb") as f:
        pickle.dump(en_embeddings, f)
    print(f"Saved English embeddings to {directory}/english_vecs.pkl")

    with open(os.path.join(directory, "twi_vecs.pkl"), "wb") as f:
        pickle.dump(tw_embeddings, f)
    print(f"Saved Twi embeddings to {directory}/twi_vecs.pkl")


def phrase_embeddings(phrase, embeddings):
    words = phrase.split()
    vecs = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else None


def build_matrices(dictionary, en_embeddings, twi_embeddings):
    dictionary = dict(dictionary)
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


def compute_loss(X, Y, R):
    m = X.shape[0]
    d = X @ R - Y
    dsqr = np.square(d)
    sum_dsqr = np.sum(dsqr)
    loss = sum_dsqr / m
    return loss


def align_embeddings(
    X, Y, steps=100, lr=0.003, verbose=True, compute_loss=compute_loss, tol=1e-4, pat=20
):
    np.random.seed(42)
    R = np.random.rand(X.shape[1], X.shape[1])
    m = X.shape[0]
    best_loss = float("inf")
    wait = 0
    loss_history = []
    iter_history = []
    for i in range(steps):
        loss = compute_loss(X, Y, R)
        if verbose and i % 100 == 0:
            print(f"loss at iteration {i} is: {loss:.4f}")
            loss_history.append(loss)
            iter_history.append(i)
        if loss + tol < best_loss:
            best_loss = loss
            wait = 0
        else:
            wait += 1
        if wait >= pat:
            if verbose:
                print(
                    f"Early stopping at iteration {i}, loss did not improve for {pat} steps."
                )
                print(f"Loss at early stopping: {loss:.4f}")
            break
        gradient = (2 / m) * (X.T @ (X @ R - Y))
        R -= lr * gradient
    return R, loss_history, iter_history
