# Fichier : models/st_graph_sage.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class STGraphSAGE(nn.Module):
    """Modèle Spatio-Temporel avec deux têtes de sortie pour l'apprentissage semi-supervisé."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGraphSAGE, self).__init__()
        # Couches partagées
        self.spatial_conv = SAGEConv(in_channels, hidden_channels)
        self.temporal_gru = nn.GRU(hidden_channels, hidden_channels)
        self.relu = nn.ReLU()
        
        # Tête de sortie 1: Reconstruction
        self.reconstruction_head = nn.Linear(hidden_channels, out_channels)
        
        # Tête de sortie 2: Classification (prédit un score d'anomalie)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid() # Pour borner le score entre 0 (normal) et 1 (anormal)
        )

    def forward(self, x, edge_index):
        h = None
        reconstructions = []
        classifications = []
        
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            spatial_embedding = self.relu(self.spatial_conv(x_t, edge_index))
            r_in = spatial_embedding.unsqueeze(0)
            temporal_output, h = self.temporal_gru(r_in, h)
            hidden_state_t = temporal_output.squeeze(0)
            
            # Passer l'état caché dans les deux têtes
            reconstructions.append(self.reconstruction_head(hidden_state_t))
            classifications.append(self.classification_head(hidden_state_t))
            
        reconstruction_output = torch.stack(reconstructions, dim=1)
        classification_output = torch.stack(classifications, dim=1).squeeze(-1) # Enlever la dernière dimension
        
        return reconstruction_output, classification_output