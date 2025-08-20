# Fichier: scripts/train.py

import torch
import torch.nn.functional as F
import yaml
import os

from models.st_graph_sage import STGraphSAGE
from utils.data_loader import (load_and_generate_training_data,
                               load_and_generate_test_data,
                               prepare_data_for_model,
                               standardize_features)
from utils.visualization import (visualize_building_graph,
                                 visualize_training_data,
                                 visualize_test_data,
                                 visualize_anomaly_scores)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    train_graphs, train_raw_features_list = load_and_generate_training_data(
        config['data']['sequence_length'],
        config['model']['in_channels']
    )
    
    train_features_scaled_list, scaler = standardize_features(train_raw_features_list)
    print("✅ Données d'entraînement préparées et standardisées.")
    
    for i, graph in enumerate(train_graphs):
        visualize_building_graph(graph, f"training_topology_{i+1}.html")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    model = STGraphSAGE(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    print("\n--- 🚀 Début de l'entraînement ---")
    model.train()
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        for i in range(len(train_graphs)):
            train_features_scaled, train_edge_index, _ = prepare_data_for_model(train_graphs[i], train_features_scaled_list[i])
            
            X_train = train_features_scaled[:, :-1, :].to(device)
            y_train_true_scaled = train_features_scaled[:, 1:, :].to(device)
            train_edge_index = train_edge_index.to(device)

            optimizer.zero_grad()
            y_train_pred_scaled = model(X_train, train_edge_index)
            loss = loss_fn(y_train_pred_scaled, y_train_true_scaled)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_graphs)
            print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], Loss: {avg_loss:.6f}')

    print("--- ✅ Entraînement terminé ---")
    os.makedirs(os.path.dirname(config['training']['weights_path']), exist_ok=True)
    torch.save(model.state_dict(), config['training']['weights_path'])
    print(f"💾 Poids du modèle sauvegardés dans : {config['training']['weights_path']}")

    test_graph, test_raw_features = load_and_generate_test_data(
        config['data']['sequence_length'],
        config['model']['in_channels']
    )
    
    visualize_building_graph(test_graph, "test_topology.html")

    test_features_scaled = (test_raw_features - scaler['mean']) / scaler['std']
    print("✅ Données de test préparées et standardisées avec le scaler de l'entraînement.")

    test_features_scaled_prepared, test_edge_index, test_inv_node_mapping = prepare_data_for_model(test_graph, test_features_scaled)

    print("\n--- 🔍 Analyse des Anomalies sur le graphe de test ---")
    X_test = test_features_scaled_prepared[:, :-1, :].to(device)
    y_test_true_scaled = test_features_scaled_prepared[:, 1:, :].to(device)
    test_edge_index = test_edge_index.to(device)

    model.eval()
    with torch.no_grad():
        y_test_pred_scaled = model(X_test, test_edge_index).cpu()

    mean = scaler['mean'].unsqueeze(0).unsqueeze(0)
    std = scaler['std'].unsqueeze(0).unsqueeze(0)
    y_test_pred_unscaled = y_test_pred_scaled * std + mean
    visualize_test_data(test_raw_features, y_test_pred_unscaled, test_inv_node_mapping, {0: 'Temp', 1: 'Press', 2: 'Flow', 3: 'State'})

    error_per_node_test = F.mse_loss(y_test_pred_scaled, y_test_true_scaled.cpu(), reduction='none').mean(dim=(1, 2))
    anomaly_threshold = config['analysis']['anomaly_threshold']

    visualize_anomaly_scores(error_per_node_test, test_inv_node_mapping, anomaly_threshold, "test_anomaly_scores.html")

    anomalous_nodes_indices_test = torch.where(error_per_node_test > anomaly_threshold)[0]

    if not anomalous_nodes_indices_test.nelement():
        print("✅ Aucune anomalie détectée sur le jeu de test.")
    else:
        print(f"🔥 {len(anomalous_nodes_indices_test)} équipement(s) suspect(s) trouvé(s) sur le jeu de test !")
        feature_map = {0: 'Température', 1: 'Pression', 2: 'Débit', 3: 'État'}
        for index_tensor in anomalous_nodes_indices_test:
            index = index_tensor.item()
            equip_name = test_inv_node_mapping[index]
            print(f"\n🚨 Anomalie potentielle sur : {equip_name} (Score d'erreur: {error_per_node_test[index]:.4f})")

            error_per_feature = F.mse_loss(y_test_pred_scaled[index], y_test_true_scaled[index].cpu(), reduction='none').mean(dim=0)
            most_anomalous_feature_idx = torch.argmax(error_per_feature).item()
            print(f"   -> Métrique la plus affectée : '{feature_map[most_anomalous_feature_idx]}'")

if __name__ == '__main__':
    main()