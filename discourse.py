import matplotlib.pyplot as plt


def plot_discourse(gl, x):
    xcoords = []
    ycoords = []
    y = 0
    for glyph in gl:
        for i in range(len(x)):
            if x[i] == glyph:
                xcoords.append(i)
                ycoords.append(y)
        y -= 1
    fig, ax = plt.subplots()
    ax.scatter(xcoords, ycoords, marker = "|", color="black")
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