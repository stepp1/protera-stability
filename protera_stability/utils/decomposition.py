import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def dim_reduction(
    X,
    y=None,
    strategy="PCA",
    n_components=2,
    prefix=None,
    plot_viz=True,
    save_viz=False,
):
    valid_strats = ("PCA", "UMAP", "TSNE")
    if strategy not in valid_strats:
        raise ValueError(f"{strategy} is not a valid dimensionality reduction strategy")

    if strategy == valid_strats[0]:
        reducer = PCA(n_components=n_components)
    elif strategy == valid_strats[1]:
        raise NotImplementedError
        # reducer = sklearn.decomposition.UMAP(n_components=2)
    elif strategy == valid_strats[2]:
        reducer = TSNE(n_components=2)

    X_hat = reducer.fit_transform(X)

    if plot_viz:
        f, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(X_hat[:, 0], X_hat[:, 1], c=y)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        cb = plt.colorbar(scatter, spacing="proportional")
        cb.set_label(prefix)
        plt.show()

    if save_viz:
        fname = f"{strategy}.png"
        print(f"Saved as {fname}")
        plt.savefig(fname, dpi=300)

    return X_hat
