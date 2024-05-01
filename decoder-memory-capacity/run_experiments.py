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
# embed_size <= n, sequence_length*embed_size <= n*vocab_size
embed_size = 16
sequence_length = 10
epochs = 50000
batch_size = "full"
plot_only = False
# n_vals = [100, 200, 300, 400, 500]
n_vals = [500]
m_vals = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# m_vals = [4096]

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
