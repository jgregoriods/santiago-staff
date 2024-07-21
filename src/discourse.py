import matplotlib.pyplot as plt


def plot_discourse(glyphs, encoded_lines, bkpt=None, figsize=(8, 6), save_path=None):
    x_coords = []
    y_coords = []
    y = 0

    text = []
    for line in encoded_lines:
        text.extend(line)

    for glyph in glyphs:
        for i in range(len(text) - len(glyph) + 1):
            if text[i:i+len(glyph)] == glyph:
                x_coords.append(i)
                y_coords.append(y)
        y -= 1

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_coords, y_coords, color='black', s=10)
    ax.set_xlim(0, len(text))
    ax.set_yticks(range(0, y, -1))
    ax.set_yticklabels([' '.join(glyph) for glyph in glyphs])
    ax.set_xlabel('Position in Text')
    ax.set_ylabel('Glyph')

    if bkpt is not None:
        break_x = sum([len(line) for line in encoded_lines[:bkpt]])
        ax.axvspan(break_x, len(text), color='gray', alpha=0.1)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

