import matplotlib.pyplot as plt


def plot_experiment(experiment, path):
    dataset_sizes = []
    m_vals = []
    heatmap_data = []
    training_threshold = 0.2

    for run in experiment.runs:
        if not (run.n in dataset_sizes):
            dataset_sizes.append(run.n)
        if not (run.m in m_vals):
            m_vals.append(run.m)
    i = 0
    min_model_num_params = []
    min_model_ms = []
    # print(experiment.runs)
    for dataset_size in dataset_sizes:
        yrow = []
        min_m = 1e8
        for m in m_vals:
            print("Run:", i + 1)
            print("Dataset Size:", dataset_size)
            print("Hidden Size:", m)
            print("Actual Dataset Size", experiment.runs[i].n)
            print("Actual Hidden Size", experiment.runs[i].m)
            print("Final Training Loss:", experiment.runs[i].training_loss_values[-1])
            print("Num Params:", experiment.runs[i].model_num_params)
            yrow.append(
                experiment.runs[i].training_loss_values[-1]
                - experiment.runs[i].emp_loss
            )
            if (
                experiment.runs[i].training_loss_values[-1]
                - experiment.runs[i].emp_loss
                < training_threshold
                and experiment.runs[i].m < min_m
            ):
                min_model_ms.append(experiment.runs[i].m)
                min_model_num_params.append(experiment.runs[i].model_num_params)
                min_m = experiment.runs[i].m
            i = i + 1
        heatmap_data.append(yrow)
    print("Heatmap Data")
    print(heatmap_data)
    print("Min Model Params")
    print(min_model_num_params)
    print("Min Model m's")
    print(min_model_ms)
    print("M vals")
    print(m_vals)
    print("Dataset Sizes")
    print(dataset_sizes)

    # Reverses order of lists
    dataset_sizes_rev = dataset_sizes[::-1]
    heatmap_data_rev = heatmap_data[::-1]

    print("Heatmap Data Rev")
    print(heatmap_data_rev)
    print("Dataset Sizes Rev")
    print(dataset_sizes_rev)

    plt.figure(0)
    plot_heatmap(
        path=path,
        data=heatmap_data_rev,
        xlabels=m_vals,
        ylabels=dataset_sizes_rev,
    )

    plt.figure(1)
    plot_lineplot(
        path=path,
        xdata=dataset_sizes,
        ydata=min_model_num_params,
        ymetric="Number of Parameters",
    )

    plt.figure(2)
    plot_lineplot(
        path=path,
        xdata=dataset_sizes,
        ydata=min_model_ms,
        ymetric="Hidden Dimension",
    )

    plt.close("all")
    return


def plot_heatmap(path, data, xlabels, ylabels):
    plt.imshow(data, cmap="bone", interpolation="nearest")
    plt.xlabel("Hidden Dimension Size")
    plt.ylabel("Dataset Size")
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    plt.colorbar()
    plt.savefig(
        path + "/plots/" + "heatmap" + ".pdf",
        bbox_inches="tight",
    )
    return


def plot_lineplot(path, xdata, ydata, ymetric):
    plt.xlabel("Dataset Size")
    plt.ylabel(ymetric)
    plt.xticks(ticks=xdata)
    plt.plot(xdata, ydata, linewidth=2, markevery=1, marker="o")
    plt.savefig(
        path + "/plots/" + "lineplot_" + ymetric + ".pdf",
        bbox_inches="tight",
    )
    return
