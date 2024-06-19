import matplotlib.pyplot as plt


def plot_discourse(gl, x):
    xcoords = []
    ycoords = []
    y = 0
    for glyph in gl:
        for i in range(len(x) - len(glyph) + 1):
            if x[i:i+len(glyph)] == glyph:
                xcoords.append(i)
                ycoords.append(y)
        y -= 1
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xcoords, ycoords, color='black', s=10)
    ax.set_xlim(0, len(x))
    ax.set_yticks(list(range(0,y,-1)), gl)


def glyph_bound(glyph, text):
    start = end = None
    for i in range(len(text)):
        if glyph in text[i]:
            if start is None:
                start = i
            end = i
    return (start, end)
