import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class STGraphSAGE(nn.Module):
    """
    Modèle Spatio-Temporel GraphSAGE.
    
    Ce modèle apprend à la fois des relations spatiales (via GraphSAGE)
    et temporelles (via un GRU) sur des séquences de graphes.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initialise les couches du modèle.
        
        Args:
            in_channels (int): Nombre de caractéristiques d'entrée par nœud (e.g., temp, pression).
            hidden_channels (int): Dimension de l'espace caché.
            out_channels (int): Nombre de caractéristiques de sortie par nœud.
        """
        super(STGraphSAGE, self).__init__()
        
        # Couche Spatiale : GraphSAGE pour agréger les infos du voisinage
        self.spatial_conv = SAGEConv(in_channels, hidden_channels)
        
        # Couche Temporelle : GRU pour capturer l'évolution dans le temps
        # Le GRU traitera les embeddings spatialement conscients générés par SAGEConv.
        self.temporal_gru = nn.GRU(hidden_channels, hidden_channels)
        
        # Couche de sortie
        self.linear = nn.Linear(hidden_channels, out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        Définit la passe avant du modèle.
        
        Args:
            x (torch.Tensor): Caractéristiques des nœuds sur plusieurs pas de temps.
                              Shape: [nombre_de_noeuds, longueur_sequence, nombre_caracteristiques]
            edge_index (torch.Tensor): Liste des arêtes du graphe.
                                       Shape: [2, nombre_d_aretes]
        """
        # Initialiser l'état caché du GRU
        h = None  # Le GRU initialise à zéro par défaut
        
        predictions = []
        
        # Itérer sur chaque pas de temps de la séquence
        for t in range(x.size(1)):
            # 1. Extraire les caractéristiques du pas de temps actuel
            x_t = x[:, t, :]
            
            # 2. Appliquer la convolution spatiale (GraphSAGE)
            # On obtient une représentation de chaque nœud qui inclut l'info de ses voisins
            spatial_embedding = self.spatial_conv(x_t, edge_index)
            spatial_embedding = self.relu(spatial_embedding)
            
            # 3. Appliquer la récurrence temporelle (GRU)
            # Le GRU a besoin d'une shape [longueur_sequence=1, batch=nb_noeuds, features]
            r_in = spatial_embedding.unsqueeze(0)
            
            # Mettre à jour l'état caché avec les nouvelles informations spatio-temporelles
            temporal_output, h = self.temporal_gru(r_in, h)
            
            # 4. Prédire la sortie pour ce pas de temps
            prediction_t = self.linear(temporal_output.squeeze(0))
            predictions.append(prediction_t)
            
        # Concaténer les prédictions de chaque pas de temps
        output = torch.stack(predictions, dim=1) # Shape: [nb_noeuds, longueur_sequence, out_channels]
        
        return output

# --- Exemple d'instanciation (pour montrer comment l'utiliser) ---
if __name__ == '__main__':
    # Paramètres du modèle
    IN_CHANNELS = 4       # e.g., Température, Pression, Débit, État (On/Off)
    HIDDEN_CHANNELS = 64  # Taille de la représentation interne
    OUT_CHANNELS = 4      # Le modèle prédit les mêmes caractéristiques pour le futur

    # Instancier le modèle
    model = STGraphSAGE(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS
    )

    print("✅ Modèle STGraphSAGE backend initialisé avec succès.")
    print(model)