from pathlib import Path

import matplotlib.pyplot as plt


def save_plot(x, y, fname: str, title: str = ""):
    Path("plots").mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(fname)
    plt.close()
