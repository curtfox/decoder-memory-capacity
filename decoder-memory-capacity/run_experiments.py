import os
import torch
import torch
from experiment import Experiment
from run import Run

# Get path
path = os.path.dirname(os.getcwd())
print(path)

if not os.path.exists(os.path.join(path, "experiments")):
    os.makedirs(os.path.join(path, "experiments"))
if not os.path.exists(os.path.join(path, "plots")):
    os.makedirs(os.path.join(path, "plots"))

# Set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set parameters
embed_size = 16
sequence_length = 10
epochs = 50000
batch_size = "full"
plot_only = False
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
    batch_size=batch_size,
    epochs=epochs,
    runs=runs,
)
ex.run_experiment(plot_only=plot_only, path=path, device=device)
