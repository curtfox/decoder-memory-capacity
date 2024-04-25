import matplotlib.pyplot as plt


def plot_experiment(experiment):
    dataset_sizes = []
    m_vals = []
    heatmap_data = []
    training_threshold = 5.4
    print(experiment.runs)
    for run in experiment.runs:
        if not (run.n in dataset_sizes):
            dataset_sizes.insert(0, run.n)
        if not (run.m in m_vals):
            m_vals.append(run.m)
    i = 0
    min_model_num_params = []
    # print(experiment.runs)
    for _ in dataset_sizes:
        yrow = []
        min_m = 1e8
        for _ in m_vals:
            yrow.append(experiment.runs[i].training_loss_values[-1])
            if (
                experiment.runs[i].training_loss_values[-1] < training_threshold
                and experiment.runs[i].m < min_m
            ):
                min_m = experiment.runs[i].m
                min_model_num_params.append(experiment.runs[i].model_num_params)
            i = i + 1
        heatmap_data.insert(0, yrow)
    print("Heatmap Data")
    print(heatmap_data)
    print("M vals")
    print(m_vals)
    print("Dataset Sizes")
    print(dataset_sizes)
    plot_heatmap(
        experiment=experiment, data=heatmap_data, xlabels=m_vals, ylabels=dataset_sizes
    )

    dataset_sizes = dataset_sizes[::-1]  # reverses order of list
    min_model_num_params = min_model_num_params[::-1]
    # for row in heatmap_data:
    plot_lineplot(
        experiment=experiment, xdata=dataset_sizes, ydata=min_model_num_params
    )

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
    plt.figure(1)
    plt.xlabel("Dataset Size")
    plt.ylabel("Number of Parameters")
    plt.plot(xdata, ydata, linewidth=2, markevery=1, marker="o")
    plt.savefig(
        experiment.path + "/plots/" + "lineplot" + ".pdf",
        bbox_inches="tight",
    )
    return
