import torch

print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TaskAClassifier(nn.Module):

    def __init__(self, emb_dim, h_dim, n_classes, drop_rate = 0.2,): #TODO: look up where to include a dropout layer
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=h_dim),
            nn.Dropout(drop_rate),

            nn.Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=h_dim),
            nn.Dropout(drop_rate),

            nn.Linear(in_features=h_dim, out_features=n_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)


