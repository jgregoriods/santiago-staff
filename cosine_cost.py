import matplotlib.pyplot as plt
import ruptures as rpt

from matplotlib.colors import LogNorm
from ruptures.base import BaseCost
from sklearn.metrics.pairwise import cosine_similarity


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


def plot_breakpoints(vectorized_text, n_bkps):
    """Plot the breakpoints of the given text."""
    algo = rpt.Dynp(custom_cost=CosineCost(), min_size=2, jump=1).fit(vectorized_text)
    predicted_bkps = algo.predict(n_bkps=n_bkps)
    print(predicted_bkps)

    fig, ax = plt.subplots(figsize=(4,4))

    # plot config
    # title_fontsize = 10
    # label_fontsize = 7

    # plot gram matrix
    ax.imshow(algo.cost.gram.toarray(), cmap="viridis", interpolation='none')
    # add text segmentation
    for start, end in rpt.utils.pairwise([0] + predicted_bkps):
        draw_square_on_ax(start=start, end=end, ax=ax)
    # add labels and title
    #ax.set_title(title, fontsize=title_fontsize)
    return fig, ax