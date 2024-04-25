import os
import torch
import torch
from experiment import Experiment
from run import Run

# nltk.download("punkt")

# Get path
path = os.path.dirname(os.getcwd())

# Set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Set parameters
embed_size = 256
sequence_length = 100
epochs = 500
batch_size = 256
plot_only = True
n_vals = [100, 500, 1000]
m_vals = [128, 256, 512]

runs = []
# Create runs
for n in n_vals:
    for m in m_vals:
        run = Run(n=n, m=m, sequence_length=sequence_length, d=embed_size)
        runs.append(run)

# Run experiment
ex = Experiment(
    path=path,
    batch_size=batch_size,
    epochs=epochs,
    device=device,
    runs=runs,
)
ex.run_experiment(plot_only=plot_only)
