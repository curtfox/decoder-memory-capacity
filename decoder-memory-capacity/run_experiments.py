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
embed_size = 32
sequence_length = 10
epochs = 10000
batch_size = 64
plot_only = False
n_vals = [100]  # , 250, 500, 1000, 1250, 1500, 1750, 2000]
m_vals = [4, 512, 1024]

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
    device=device,
    runs=runs,
)
ex.run_experiment(plot_only=plot_only, path=path)
