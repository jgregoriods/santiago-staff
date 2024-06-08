import matplotlib.pyplot as plt
import numpy as np
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


def calculate_aic(n, mse, k):
    return n * np.log(mse) + 2 * k


def find_optimnal_bkps(vectorized_text, n_bkps_range):
    X = vectorized_text.toarray()

    n, d = X.shape

    best_aic = float("inf")
    best_n_bkps = None

    for n_bkps in n_bkps_range:
        try:
            algo = rpt.Dynp(custom_cost=CosineCost(), min_size=1, jump=1).fit(X)
            result = algo.predict(n_bkps=n_bkps)
            n_segments = len(result) - 1

            total_mse = 0
            for i in range(n_segments):
                start = result[i]
                end = result[i+1]
                segment = X[start:end]
                segment_mse = np.mean(segment, axis=0)
                total_mse += np.sum((segment - segment_mse) ** 2)

            mse = total_mse / n

            aic = calculate_aic(n, mse, n_segments)

            if aic < best_aic:
                best_aic = aic
                best_n_bkps = n_bkps
        except:
            print(f"Could not run the algorithm for n_bkps={n_bkps}")

    return best_n_bkps


def plot_breakpoints(vectorized_text, n_bkps):
    """Plot the breakpoints of the given text."""
    X = vectorized_text.toarray()
    algo = rpt.Dynp(custom_cost=CosineCost(), min_size=1, jump=1).fit(X)
    predicted_bkps = algo.predict(n_bkps=n_bkps)

    fig, ax = plt.subplots(figsize=(4,4))

    # plot config
    # title_fontsize = 10
    # label_fontsize = 7

    # plot gram matrix
    ax.imshow(algo.cost.gram, cmap="viridis", norm=LogNorm(), interpolation="none")
    # add text segmentation
    for start, end in rpt.utils.pairwise([0] + predicted_bkps):
        draw_square_on_ax(start=start, end=end, ax=ax)
    # add labels and title
    #ax.set_title(title, fontsize=title_fontsize)
    plt.show()

    return predicted_bkps
