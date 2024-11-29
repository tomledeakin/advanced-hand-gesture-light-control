import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Dataset
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.labels = torch.from_numpy(self.data.iloc[:,0].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_hot_label = self.labels[idx]
        torch_data = torch.from_numpy(self.data.iloc[idx,1:].to_numpy(dtype=np.float32))
        return torch_data, one_hot_label