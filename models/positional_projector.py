import torch
import torch.nn as nn

class PositionalEmbeddingProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(PositionalEmbeddingProjector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        # x = self.batch_norm(x)

        x = self.activation(x)
        x = self.dropout(x)
        return x
