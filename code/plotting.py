import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
import numpy as np

fontsize_axis = 7
fontsize_ticks = 6

default_height_to_width_ratio = (5.0**0.5 - 1.0) / 2.0


def plot_experiment(experiment, path):
    dataset_sizes = []
    unique_beginnings = []
    m_vals = []
    heatmap_data = []
    eps = 0.01

    for run in experiment.runs:
        if not (run.n in dataset_sizes):
            dataset_sizes.append(run.n)
            unique_beginnings.append(run.unique_beginnings)
        if not (run.m in m_vals):
            m_vals.append(run.m)

    i = 0
    min_model_num_params = []
    min_model_ms = []
    for _ in unique_beginnings:
        yrow = []
        min_m = 1e8
        for _ in m_vals:
            training_loss = experiment.runs[i].training_loss_values[-1]
            yrow.append((training_loss - experiment.runs[i].emp_loss))
            if training_loss - experiment.runs[i].emp_loss < (
                eps * experiment.runs[i].emp_loss
            ) and (experiment.runs[i].m < min_m):
                min_model_ms.append(experiment.runs[i].m)
                min_model_num_params.append(experiment.runs[i].model_num_params)
                min_m = experiment.runs[i].m
            i = i + 1
        heatmap_data.append(yrow)

    # Reverses order of lists
    unique_beginnings_rev = unique_beginnings[::-1]
    heatmap_data_rev = heatmap_data[::-1]

    nrows = 1
    ncols = 2
    _, ax = plot_settings(nrows=nrows, ncols=ncols)

    plot_heatmap(
        axis=ax[0],
        data=np.array(heatmap_data_rev),
        xlabel="Hidden Dimension Size",
        ylabel="Unique Contexts",
        xticks=m_vals,
        yticks=unique_beginnings_rev,
    )
    plot_lineplot(
        axis=ax[1],
        xdata=unique_beginnings,
        ydata=min_model_num_params,
        xlabel="Unique Contexts",
        ylabel="Number of Parameters",
    )

    plt.savefig(
        os.path.join(
            path, "plots", "plot-subset-" + str(experiment.train_subset) + ".pdf"
        ),
        bbox_inches="tight",
    )
    return


def plot_heatmap(axis, data, xlabel, ylabel, xticks, yticks):
    plot = axis.imshow(
        data,
        cmap="bone",
        norm=colors.LogNorm(vmin=data.min(), vmax=data.max()),
        aspect=0.75,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(len(xticks)), xticks)
    axis.set_yticks(range(len(yticks)), yticks)
    axis.tick_params(axis="x", rotation=90)
    cbar = plt.colorbar(plot, format="%1.3g")
    cbar.ax.tick_params()


def plot_lineplot(axis, xdata, ydata, xlabel, ylabel):
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(ticks=np.arange(0, 3500, 500))
    axis.tick_params(axis="x", rotation=90)
    if ylabel == "Number of Parameters":
        axis.ticklabel_format(style="sci", axis="y", scilimits=(4, 4))
    axis.plot(xdata, ydata)


def plot_settings(
    nrows=1, ncols=1, width=6.0, height_to_width_ratio=default_height_to_width_ratio
):
    subplot_width = width / ncols
    subplot_height = height_to_width_ratio * subplot_width
    height = subplot_height * nrows
    figsize = (width, height)

    plt.rcParams.update(
        {
            "axes.labelsize": fontsize_axis,
            "figure.figsize": figsize,
            "figure.constrained_layout.use": False,
            "figure.autolayout": False,
            "lines.linewidth": 2,
            "lines.marker": "o",
            "xtick.labelsize": fontsize_ticks,
            "ytick.labelsize": fontsize_ticks,
            "figure.dpi": 250,
        }
    )

    return plt.subplots(nrows, ncols, constrained_layout=True)
