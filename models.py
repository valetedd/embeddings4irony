import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Simple attention mechanism to focus on important features in embeddings
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        # x shape: (batch_size, emb_dim)
        attention_weights = F.softmax(self.attention(x), dim=1)
        # Apply attention weights
        context_vector = attention_weights * x
        return context_vector, attention_weights


class TaskAClassifier(nn.Module):

    def __init__(
        self,
        emb_dim,
        n_classes,
        h_dim: int,
        drop_rate=0.2,
        use_attention=True,
        use_learnable_dr=False,
        reduced_dim=None,
    ):
        super().__init__()

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(emb_dim)

        # Learnable dimensionality reduction
        self.use_learnable_dr = use_learnable_dr
        input_dim = emb_dim
        self.dr_layer = None

        if self.use_learnable_dr and reduced_dim is not None:
            self.dr_layer = nn.Linear(input_dim, reduced_dim)
            input_dim = reduced_dim  # Update input dimension
        # Main network
        self.network = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            # output layer
            nn.Linear(in_features=h_dim, out_features=n_classes),
        )
        self._init_weights()

        # Save attention weights for analysis
        self.last_attention_weights = None

    def _init_weights(self):
        # He initialization for ReLU-based networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        if inputs.dtype not in [torch.float32, torch.float16]:
            inputs = inputs.float()
        # Optional attention step
        if self.use_attention:
            # Apply attention to input embeddings
            inputs, attention_weights = self.attention(inputs)
            self.last_attention_weights = attention_weights

        # Optional dimensionality reduction step
        if self.use_learnable_dr and self.dr_layer is not None:
            inputs = F.relu(self.dr_layer(inputs))

        return self.network(inputs)

    def get_attention_weights(self):
        """Return the attention weights from the last forward pass"""
        if not self.use_attention:
            return None
        return self.last_attention_weights

    def l1_penalty(self, coeff):
        penalty = 0
        for name, param in self.named_parameters():
            if "bias" not in name:
                penalty += torch.abs(param).sum()

        return penalty * coeff
