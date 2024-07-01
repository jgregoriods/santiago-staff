import numpy as np
from scipy.stats import norm


def glyph_indices(glyph, text):
    # glyph is a list of lists
    res = []
    for i in range(len(text) - len(glyph) + 1):
        if glyph == text[i:i + len(glyph)]:
            res.append(i)
    return res

def nearest_neighbor_analysis_1d(points, length, alpha=0.05):
    n_points = len(points)
    if n_points < 2:
        return np.nan, "random"

    nearest_distances = np.diff(points)
    observed_mean_distance = np.mean(nearest_distances)
    expected_mean_distance = length / (n_points + 1)
    nnr = observed_mean_distance / expected_mean_distance

    sd = np.std(nearest_distances, ddof=1)
    standard_error = sd / np.sqrt(n_points)
    z_score = (observed_mean_distance - expected_mean_distance) / standard_error

    critical_value = norm.ppf(1 - alpha / 2)
    if abs(z_score) > critical_value:
        result = "clustered" if z_score < 0 else "dispersed"
    else:
        result = "random"

    return z_score, result


def glyph_bound(glyph, text):
    start = end = None
    for i, line in enumerate(text):
        for j in range(len(line) - len(glyph) + 1):
            if glyph == line[j:j + len(glyph)]:
                if start is None:
                    start = i
                end = i
    return (start, end)


def analyze_glyphs(encoded_lines, min_count=4):
    text = []
    for line in encoded_lines:
        text.extend(line)
    unique_glyphs = {glyph for glyph in text if glyph != '?' and text.count(glyph) >= min_count}
    unique_glyphs = [[glyph] for glyph in unique_glyphs]
    clustered = []
    dispersed = []

    for glyph in unique_glyphs:
        points = glyph_indices(glyph, text)
        _, result = nearest_neighbor_analysis_1d(points, len(text))
        if result == "clustered":
            clustered.append(glyph)
        elif result == "dispersed":
            dispersed.append(glyph)

    clustered_sorted = sorted(clustered, key=lambda x: glyph_bound(x, encoded_lines))
    # clustered_formatted = [[glyph] for glyph in clustered_sorted]

    return clustered_sorted, dispersed

