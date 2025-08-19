import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F # Assurez-vous d'importer F
from backend_model import STGraphSAGE
from frontend_visualization import visualize_building

def create_hvac_building_graph():
    """Crée un exemple de graphe NetworkX pour un système CVC simple."""
    G = nx.Graph()
    
    equipments = {
        'Chiller_01': {'type': 'Chiller'},
        'Boiler_01': {'type': 'Boiler'},
        'Pump_CW_01': {'type': 'Pump'},
        'Pump_HW_01': {'type': 'Pump'},
        'AHU_01': {'type': 'AHU'},
        'VAV_Zone_N': {'type': 'VAV'},
        'VAV_Zone_S': {'type': 'VAV'}
    }
    for name, attrs in equipments.items():
        G.add_node(name, **attrs)

    edges = [
        ('Chiller_01', 'Pump_CW_01'), ('Pump_CW_01', 'AHU_01'),
        ('Boiler_01', 'Pump_HW_01'), ('Pump_HW_01', 'AHU_01'),
        ('AHU_01', 'VAV_Zone_N'), ('AHU_01', 'VAV_Zone_S')
    ]
    G.add_edges_from(edges)
    
    return G

def generate_sample_time_series_data(graph, seq_length=24, num_features=4):
    """Génère des données de séries temporelles factices pour chaque nœud."""
    num_nodes = graph.number_of_nodes()
    time = np.linspace(0, 4 * np.pi, seq_length)
    base_signal = np.sin(time)
    
    node_features = []
    for i in range(num_nodes):
        noise = np.random.randn(seq_length, num_features) * 0.1
        variation = (np.random.rand(num_features) - 0.5) * 2
        node_signal = np.outer(base_signal, variation) + noise
        node_features.append(node_signal)
        
    return np.stack(node_features)


def train_model(model, features, edge_index, epochs=50, learning_rate=0.01):
    """Fonction d'entraînement pour le modèle STGraphSAGE."""
    print("\n--- 🚀 Début de l'entraînement ---")
    
    X = features[:, :-1, :]
    y_true = features[:, 1:, :]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X, edge_index)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    print("--- ✅ Entraînement terminé ---")
    return model

if __name__ == '__main__':
    # 1. Créer le graphe du bâtiment
    building_graph = create_hvac_building_graph()
    print(f"✅ Graphe du bâtiment créé avec {building_graph.number_of_nodes()} nœuds et {building_graph.number_of_edges()} arêtes.")
    
    # 2. Générer les données de séries temporelles
    SEQ_LENGTH = 24
    NUM_FEATURES = 4
    node_data = generate_sample_time_series_data(building_graph, SEQ_LENGTH, NUM_FEATURES)
    print(f"📊 Données de séries temporelles générées. Shape : {node_data.shape}")
    
    # 3. Visualiser le graphe avec le frontend
    visualize_building(building_graph)
    
    # 4. Préparer les données pour le modèle backend
    node_features_tensor = torch.tensor(node_data, dtype=torch.float32)
    node_mapping = {node: i for i, node in enumerate(building_graph.nodes())}
    edges = [[node_mapping[u], node_mapping[v]] for u, v in building_graph.edges()]
    edges.extend([[v, u] for u, v in edges])
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    print("\n--- Préparation pour le modèle backend ---")
    print(f"Shape du tenseur de caractéristiques (X) : {node_features_tensor.shape}")
    print(f"Shape du tenseur d'arêtes (edge_index) : {edge_index_tensor.shape}")

    # 5. Instancier et entraîner le modèle
    model = STGraphSAGE(in_channels=NUM_FEATURES, hidden_channels=64, out_channels=NUM_FEATURES)
    trained_model = train_model(model, node_features_tensor, edge_index_tensor, epochs=50)

    # 6. Utiliser le modèle entraîné pour faire une prédiction
    trained_model.eval()
    with torch.no_grad():
        # y_pred sont les prédictions du modèle
        y_pred = trained_model(node_features_tensor[:, :-1, :], edge_index_tensor)
        # y_true sont les valeurs réelles que le modèle essayait de prédire
        y_true = node_features_tensor[:, 1:, :]

    print(f"\nPrédiction finale après entraînement. Shape: {y_pred.shape}")
    
    # #############################################################################
    # ### 7. Interprétation des Résultats (ANALYSE DES ANOMALIES) - VOTRE CODE ICI ###
    # #############################################################################
    print("\n--- 🔍 Analyse des Anomalies ---")

    # --- 7.1 Identifier l'Équipement Anormal (Dimension Spatiale) ---

    # Calcul de l'erreur quadratique moyenne (MSE) pour chaque noeud
    erreur_par_noeud = F.mse_loss(y_pred, y_true, reduction='none')
    erreur_par_noeud = erreur_par_noeud.mean(dim=(1, 2)) # Moyenne sur le temps et les features

    # Interprétation
    # NOTE : Ce seuil est arbitraire. En pratique, il doit être déterminé
    # en analysant l'erreur de reconstruction sur un jeu de données de validation.
    seuil_anomalie = 0.012  
    
    noeuds_anormaux_indices = torch.where(erreur_par_noeud > seuil_anomalie)[0]
    
    # Inverser le mapping pour retrouver les noms
    inv_node_mapping = {i: node for node, i in node_mapping.items()}

    if not noeuds_anormaux_indices.nelement():
        print("✅ Aucune anomalie détectée (score d'erreur inférieur au seuil pour tous les équipements).")
    else:
        print(f"🔥 {len(noeuds_anormaux_indices)} équipement(s) avec un score d'anomalie élevé trouvé(s) !")
        for index_tensor in noeuds_anormaux_indices:
            index = int(index_tensor.item())
            equipement_nom = inv_node_mapping[index]
            print(f"\n🚨 Anomalie détectée sur l'équipement : {equipement_nom} (Score d'erreur: {erreur_par_noeud[index]:.4f})")

            # --- 7.2 Identifier la Métrique Anormale (Dimension Caractéristique) ---
            # Ce bloc est maintenant à l'intérieur de la boucle pour analyser chaque équipement anormal

            erreur_noeud_anormal = F.mse_loss(y_pred[index], y_true[index], reduction='none')
            erreur_par_feature = erreur_noeud_anormal.mean(dim=0) # Moyenne sur le temps

            features_map = {0: 'Température', 1: 'Pression', 2: 'Débit', 3: 'État'}
            feature_la_plus_anormale_index = int(torch.argmax(erreur_par_feature).item())
            
            print(f"   -> La métrique la plus suspecte pour cet équipement est : '{features_map[feature_la_plus_anormale_index]}'")
    


