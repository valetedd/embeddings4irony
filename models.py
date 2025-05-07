import torch.nn as nn


class TaskAClassifier(nn.Module):

    def __init__(
        self,
        emb_dim,
        h_dim,
        n_classes,
        drop_rate=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=h_dim),
            nn.Dropout(drop_rate),
            nn.Linear(in_features=h_dim, out_features=h_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=h_dim // 2),
            nn.Dropout(drop_rate),
            nn.Linear(in_features=h_dim // 2, out_features=n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        # He initialization for ReLU-based networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        return self.network(inputs)
