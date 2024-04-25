import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plot_markers = ["o", "v", "^", "8", "s", "p", "P", "*", "h", "X", "D", "d"]


def plot_experiment(experiment):
    xlabels = []
    ylabels = []
    heatmap_data = []
    for run in experiment.runs:
        if not (run.n in ylabels):
            ylabels.insert(0, run.n)
        if not (run.m in xlabels):
            xlabels.append(run.m)
    i = 0
    for _ in ylabels:
        yrow = []
        for _ in xlabels:
            yrow.append(experiment.runs[i].training_loss_values[-1])
            i = i + 1
        heatmap_data.insert(0, yrow)
    print(heatmap_data)
    print(xlabels)
    print(ylabels)
    plot_heatmap(experiment, heatmap_data, xlabels, ylabels)
    for row in heatmap_data:

    plot_lineplot(experiment)

    plt.close("all")
    return


def plot_heatmap(experiment, data, xlabels, ylabels):
    plt.figure(0)

    plt.imshow(data, cmap="bone", interpolation="nearest")
    plt.xlabel("Hidden Dimension Size")
    plt.ylabel("Dataset Size")
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    plt.colorbar()
    plt.savefig(
        experiment.path + "/plots/" + "heatmap" + ".pdf",
        bbox_inches="tight",
    )
    return


def plot_lineplot(experiment, xdata, ydata):
    # mpl.rcParams["axes.spines.right"] = False
    # mpl.rcParams["axes.spines.top"] = False
    plt.figure(1)
    plt.xlabel("Dataset Size")
    plt.ylabel("Number of Parameters")
    plt.plot(xdata, ydata, linewidth=2, markevery=1, marker="o")
    plt.savefig(
        experiment.path + "/plots/" + "lineplot" + ".pdf",
        bbox_inches="tight",
    )
    return


"""
def plot(experiment, run, run_data, plot_settings, j):
    plt.figure(j)
    plt.plot(
        list(range(0, len(run_data))),
        run_data,
        linewidth=plot_settings["linewidth"],
        markevery=plot_settings["markevery"],
        label=plot_settings["label"],
        marker=plot_settings["marker"],
    )
    plt.xlabel(experiment.plot_metrics.x_metric)
    plt.ylabel(plot_settings["y_metric"])
    plt.xlim(left=0, right=len(run_data))
    plt.title(setup_title(experiment, run))

    plt.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc="lower center",
        borderaxespad=0,
        ncol=plot_settings["ncols"],
    )
    plt.savefig(
        experiment.path
        + "/plots/"
        + experiment.dataset.name
        + "_"
        + experiment.model_type
        + "_"
        + plot_settings["y_metric"]
        + ".pdf",
        bbox_inches="tight",
    )
    return
"""
