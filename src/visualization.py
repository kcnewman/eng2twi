from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# def plot_embeddings(
#     embeddings, labels, title="Embedding Space", save_path="../results/figures"
# ):
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(embeddings)

#     plt.figure(figsize=(10, 8))
#     for i, label in enumerate(labels):
#         x, y = reduced[i, 0], reduced[i, 1]
#         plt.scatter(x, y)
#         plt.text(x + 0.01, y + 0.01, label, fontsize=9)
#     plt.title(title)
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()


def plot_embeddings(
    embeddings, labels, title, save_path="../results/figures", annotate_limit=100
):
    reduced = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=10)

    # Annotate only a few points
    for i in range(min(annotate_limit, len(labels))):
        x, y = reduced[i]
        plt.text(x + 0.01, y + 0.01, labels[i], fontsize=8)

    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
