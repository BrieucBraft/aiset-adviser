import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class STGraphSAGE(nn.Module):
    """
    Modèle Spatio-Temporel GraphSAGE.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGraphSAGE, self).__init__()
        self.spatial_conv = SAGEConv(in_channels, hidden_channels)
        self.temporal_gru = nn.GRU(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        h = None
        predictions = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            spatial_embedding = self.relu(self.spatial_conv(x_t, edge_index))
            r_in = spatial_embedding.unsqueeze(0)
            temporal_output, h = self.temporal_gru(r_in, h)
            prediction_t = self.linear(temporal_output.squeeze(0))
            predictions.append(prediction_t)
        return torch.stack(predictions, dim=1)