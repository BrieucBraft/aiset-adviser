# Fichier : scripts/train.py

import torch
import torch.nn.functional as F
import yaml

from models.st_graph_sage import STGraphSAGE
from utils.data_loader import (create_hvac_building_graph, 
                               generate_sample_time_series_data, 
                               prepare_data_for_model,
                               standardize_features)
from utils.visualization import (visualize_building_graph, 
                                 visualize_training_data,
                                 visualize_anomaly_scores)

def main():
    # --- 1. Charger la configuration ---
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    # --- 2. Charger les données brutes ---
    building_graph = create_hvac_building_graph()
    raw_node_data = generate_sample_time_series_data(
        building_graph, 
        config['data']['sequence_length'], 
        config['model']['in_channels']
    )
    features_raw, edge_index, inv_node_mapping = prepare_data_for_model(building_graph, raw_node_data)
    
    # --- 3. Standardiser les données ---
    features_scaled, scaler = standardize_features(features_raw)
    
    print("✅ Données préparées et standardisées.")
    
    visualize_building_graph(building_graph)

    # --- 4. Initialiser le modèle et l'optimiseur ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    model = STGraphSAGE(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    # --- 5. Boucle d'entraînement (sur données standardisées) ---
    print("\n--- 🚀 Début de l'entraînement ---")
    X = features_scaled[:, :-1, :].to(device)
    y_true_scaled = features_scaled[:, 1:, :].to(device)
    edge_index = edge_index.to(device)
    
    model.train()
    for epoch in range(config['training']['epochs']):
        optimizer.zero_grad()
        y_pred_scaled = model(X, edge_index)
        loss = loss_fn(y_pred_scaled, y_true_scaled)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], Loss: {loss.item():.6f}')

    print("--- ✅ Entraînement terminé ---")
    torch.save(model.state_dict(), config['training']['weights_path'])
    print(f"💾 Poids du modèle sauvegardés dans : {config['training']['weights_path']}")
    
    # --- 6. Évaluation et Visualisation ---
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X, edge_index).cpu()
        y_true_scaled = y_true_scaled.cpu()

    # Dés-standardiser les prédictions pour une visualisation lisible
    mean = scaler['mean'].unsqueeze(0).unsqueeze(0)
    std = scaler['std'].unsqueeze(0).unsqueeze(0)
    y_pred_unscaled = y_pred_scaled * std + mean
    
    visualize_training_data(features_raw, y_pred_unscaled, inv_node_mapping, {0: 'Temp', 1: 'Press', 2: 'Flow', 3: 'State'})

    # --- 7. Analyse des anomalies (sur l'erreur standardisée) ---
    print("\n--- 🔍 Analyse des Anomalies ---")
    error_per_node = F.mse_loss(y_pred_scaled, y_true_scaled, reduction='none').mean(dim=(1, 2))
    anomaly_threshold = config['analysis']['anomaly_threshold']
    
    visualize_anomaly_scores(error_per_node, inv_node_mapping, anomaly_threshold)
    
    anomalous_nodes_indices = torch.where(error_per_node > anomaly_threshold)[0]

    if not anomalous_nodes_indices.nelement():
        print("✅ Aucune anomalie détectée.")
    else:
        print(f"🔥 {len(anomalous_nodes_indices)} équipement(s) suspect(s) trouvé(s) !")
        feature_map = {0: 'Température', 1: 'Pression', 2: 'Débit', 3: 'État'}
        for index_tensor in anomalous_nodes_indices:
            index = index_tensor.item()
            equip_name = inv_node_mapping[index]
            print(f"\n🚨 Anomalie potentielle sur : {equip_name} (Score d'erreur: {error_per_node[index]:.4f})")
            
            error_per_feature = F.mse_loss(y_pred_scaled[index], y_true_scaled[index], reduction='none').mean(dim=0)
            most_anomalous_feature_idx = torch.argmax(error_per_feature).item()
            print(f"   -> Métrique la plus affectée : '{feature_map[most_anomalous_feature_idx]}'")

if __name__ == '__main__':
    main()