import os
import torch
from experiment import Experiment
from run import Run

# Get Path
path = os.path.dirname(os.getcwd())

# Create folders for experiments and plots
if not os.path.exists(os.path.join(path, "experiments")):
    os.makedirs(os.path.join(path, "experiments"))
if not os.path.exists(os.path.join(path, "plots")):
    os.makedirs(os.path.join(path, "plots"))

# Set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

### Set parameters ###
embed_size = 16
sequence_length = 10
dataset = "tinystories"
epochs = 50000
batch_size = "full"
train_subset = False  # change to True to only train linear layers
plot_only = False  # change to True if you want to plot existing experimental results, assuming experiment pkl file already exists
n_vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
m_vals = [4, 8, 16, 32, 64, 128, 256, 512]

runs = []
# Create runs
for n in n_vals:
    for m in m_vals:
        run = Run(n=n, m=m, sequence_length=sequence_length, d=embed_size)
        runs.append(run)

# Run experiment
ex = Experiment(
    dataset=dataset,
    batch_size=batch_size,
    epochs=epochs,
    runs=runs,
    train_subset=train_subset,
)
ex.run_experiment(plot_only=plot_only, path=path, device=device)
