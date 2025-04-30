import torch

print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# sklearn smth


class TaskAClassifier(nn.Module):

    def __init__(self, emb_dim, h_dim, drop_rate, n_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features=emb_dim, out_features=h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=h_dim),
            nn.Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=h_dim),
            nn.Linear(in_features=h_dim, out_features=n_classes),
        )

    def forward(self):
        return self.network()
