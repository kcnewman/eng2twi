from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def test_vocabulary(X, Y, R):
    """
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    """
    pred = X @ R
    sim_matrix = cosine_similarity(pred, Y)
    pred_idx = np.argmax(sim_matrix, axis=1)
    corrects = np.sum(pred_idx == np.arange(len(Y)))
    accuracy = corrects / len(Y)
    return accuracy
