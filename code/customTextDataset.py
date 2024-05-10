from torch.utils.data import Dataset


### Convert data to Python Dataset
class CustomTextDataset(Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        sequence = self.sequence[idx]
        return sequence
