import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embeddings(embeddings, labels, title="Embedding Space", save_path=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = reduced[i, 0], reduced[i, 1]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, label, fontsize=9)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
