import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class STGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGraphSAGE, self).__init__()
        self.spatial_conv = SAGEConv(in_channels, hidden_channels)
        self.temporal_gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        num_nodes, seq_len, _ = x.shape
        
        spatial_embeddings = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            spatial_embedding_t = self.relu(self.spatial_conv(x_t, edge_index))
            spatial_embeddings.append(spatial_embedding_t)
        
        x_spatially_processed = torch.stack(spatial_embeddings, dim=1)
        
        temporal_output, _ = self.temporal_gru(x_spatially_processed)
        
        predictions = self.linear(temporal_output)
        
        return predictions