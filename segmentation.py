import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt

from matplotlib.colors import LogNorm
from ruptures.base import BaseCost
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(lines):
    line_str = [' '.join(line) for line in lines]
    vectorizer = TfidfVectorizer(analyzer="word", token_pattern = '[0-9]+[a-zAZ]*[.0-9]*[a-zAZ]*')
    vectorized_text = vectorizer.fit_transform(line_str)
    return vectorized_text, vectorizer


def save_glyphs(vectorized_text, vectorizer, breakpoints):
    feature_names = vectorizer.get_feature_names_out()
    X = vectorized_text.toarray()
    distinctive_glyphs = []
    for bkpt in breakpoints:
        indices = [0] + bkpt
        for i in range(len(indices) - 1):
            best_features = np.argmax(X[indices[i]:indices[i+1], :], axis=1)
            distinctive_glyphs.append(feature_names[list(set(best_features))].tolist())
    return distinctive_glyphs


class CosineCost(BaseCost):
    """Cost derived from the cosine similarity."""

    # The 2 following attributes must be specified for compatibility.
    model = "custom_cosine"
    min_size = 2

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        self.gram = cosine_similarity(signal, dense_output=False)
        return self

    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment
        Returns:
            segment cost
        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub_gram = self.gram[start:end, start:end]
        val = sub_gram.diagonal().sum()
        val -= sub_gram.sum() / (end - start)
        return val


def draw_square_on_ax(start, end, ax, linewidth=1.2, color="black"):
    """Draw a square on the given ax object."""
    ax.vlines(
        x=[start - 0.5, end - 0.5],
        ymin=start - 0.5,
        ymax=end - 0.5,
        linewidth=linewidth,
        color=color
    )
    ax.hlines(
        y=[start - 0.5, end - 0.5],
        xmin=start - 0.5,
        xmax=end - 0.5,
        linewidth=linewidth,
        color=color
    )
    return ax


def plot_breakpoints(vectorized_text, n_bkps, save_path=None):
    """Plot the breakpoints of the given text."""
    X = vectorized_text.toarray()
    algo = rpt.Dynp(custom_cost=CosineCost(), min_size=1, jump=2).fit(X)

    num_cols = min(len(n_bkps), 2)
    num_rows = (len(n_bkps) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols)

    if len(n_bkps) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    res = []
    for idx, n in enumerate(n_bkps):
        predicted_bkps = algo.predict(n_bkps=n)
        res.append(predicted_bkps)

        title_fontsize = 16
        ax = axes[idx]
        ax.imshow(algo.cost.gram, cmap="viridis", norm=LogNorm(), interpolation="none")
        for start, end in rpt.utils.pairwise([0] + predicted_bkps):
            draw_square_on_ax(start=start, end=end, ax=ax, color="white")
        ax.set_title(f"n={n}", fontsize=title_fontsize)
        ax.set_xticks([i for i in range(X.shape[0])])
        ax.set_xticklabels([i + 1 for i in range(X.shape[0])])
        ax.set_yticks([i for i in range(X.shape[0])])
        ax.set_yticklabels([i + 1 for i in range(X.shape[0])])
        ax.set_xlabel("line")
        ax.set_ylabel("line")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    return res
