#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
import scienceplots

from matplotlib.pyplot import cm

# Налаштування для підтримки спеціальних символів через LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Налаштування стилю
plt.style.use('science')

colors = cm.tab10(np.linspace(0, 1, 10))

# Data
N = 200
x = np.linspace(0, 4 * np.pi, N)
sin = np.sin(x) + np.random.rand(*x.shape) * np.sin(x) * 0.2 + 10
cos = np.cos(x) + np.random.rand(*x.shape) * np.cos(x) * 0.2 + 10
std_sin = np.random.rand(*x.shape) * 0.2 + 0.4
std_cos = np.random.rand(*x.shape) * 0.2 + 0.4


def fl(p, gamma):
    return -((1 - p) ** gamma) * np.log(p)


for gamma in [0, 0.5, 1, 2, 5]:
    x = np.linspace(0, 1, 250)
    plt.plot(x, fl(x, gamma), label=r"$\gamma = {}$".format(gamma))

def make_content():
    # plt.plot(x, sin, label="sin", color=colors[0])
    # plt.fill_between(x, sin - std_sin, sin + std_sin, alpha=0.3, color=colors[0])
    # plt.plot(x, cos, label="cos", color=colors[1])
    # plt.fill_between(x, cos - std_cos, cos + std_cos, alpha=0.3, color=colors[1])
    # plt.xlabel("Some interesting $X$ label")
    # plt.ylabel("Some interesting $Y$ label")

    for gamma, marker in zip([0, 0.5, 1, 2, 5], ["+", "x", "d", ".", "*"]):
        x = np.linspace(0, 1, 250)
        plt.plot(x, fl(x, gamma), label=r"$\gamma = {}$".format(gamma), marker=marker, markersize=4, markevery=10)

    # plt.text(x=0.05, y=4.15, s=r"FL$(p_c^t,\gamma)=-(1-p_c^t)^\gamma \log(p_c^t)$", fontsize=8)

    start = np.array([0.6, 1.0])
    end = np.array([1.0, 1.0])
    line = np.array([start, end])
    lw = 0.5
    plt.plot(*line.T, color="black", lw=lw)
    diff = [0.0, 0.1]
    plt.plot(*np.array([start - diff, start + diff]).T, color="black", lw=lw)
    plt.plot(*np.array([end - diff, end + diff]).T, color="black", lw=lw)
    plt.text(x=0.8, y=1.15, s="easy objects", ha="center")


    plt.xlabel("Probability of ground-truth class ($p_c^t$)")
    plt.ylabel("Focal Loss (FL)")
    plt.xlim(0, 1)
    plt.ylim(0, 5.5)
    plt.tight_layout()



def make_base():
    plt.figure()
    make_content()
    plt.legend()
    plt.savefig("base.png", dpi=240)


def make_sciplots():
    plt.figure()
    make_content()
    plt.legend()
    plt.savefig("sciplots.png", dpi=240)


def make_legend():
    plt.figure()
    make_content()
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)
    plt.savefig("legend.png", dpi=240)


def make_linewidths():

    fig = plt.figure()
    ax = plt.gca()
    make_content()
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)
    width = 0.5
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)
    ax.spines["top"].set_linewidth(width)
    ax.tick_params(width=width)
    plt.savefig("linewidths.png", dpi=240)

def make_pgf():
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            # "font.size": 20,
        }
    )

    fig = plt.figure()
    ax = plt.gca()
    make_content()
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)
    width = 0.5
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)
    ax.spines["top"].set_linewidth(width)
    ax.tick_params(width=width)
    plt.savefig("pgf.pgf")

def make_final():
    textwidth = 3.31314
    aspect_ratio = 6/8
    scale = 1.0
    figwidth = textwidth * scale
    figheight = figwidth * aspect_ratio

    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            # "font.size": 20,
        }
    )

    fig = plt.figure(figsize=(figwidth, figheight))
    ax = plt.gca()
    make_content()
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)
    width = 0.5
    ax.spines["left"].set_linewidth(width)
    ax.spines["bottom"].set_linewidth(width)
    ax.spines["right"].set_linewidth(width)
    ax.spines["top"].set_linewidth(width)
    ax.tick_params(width=width)
    plt.savefig("final.pgf")

make_base()
make_sciplots()
make_legend()
# make_linewidths()
make_pgf()
make_final()
