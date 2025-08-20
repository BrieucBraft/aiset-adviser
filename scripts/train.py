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
                                 visualize_supervised_test_data)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    train_graphs, train_raw_features_list, train_labels_list = load_and_generate_training_data(
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
        out_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("\n--- 🚀 Début de l'entraînement ---")
    model.train()
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        for i in range(len(train_graphs)):
            features_scaled, labels, edge_index, _ = prepare_data_for_model(
                train_graphs[i], train_features_scaled_list[i], train_labels_list[i]
            )
            
            X_train = features_scaled.to(device)
            y_train_labels = labels.to(device)
            edge_index = edge_index.to(device)

            optimizer.zero_grad()
            pred_logits = model(X_train, edge_index)
            loss = loss_fn(pred_logits, y_train_labels)
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

    test_graph, test_raw_features, test_labels = load_and_generate_test_data(
        config['data']['sequence_length'],
        config['model']['in_channels']
    )
    
    visualize_building_graph(test_graph, "test_topology.html")

    test_features_scaled = (test_raw_features - scaler['mean']) / scaler['std']
    print("✅ Données de test standardisées avec le scaler de l'entraînement.")

    features_scaled, labels, edge_index, inv_node_mapping = prepare_data_for_model(
        test_graph, test_features_scaled, test_labels
    )

    print("\n--- 🔍 Évaluation sur le graphe de test ---")
    X_test = features_scaled.to(device)
    edge_index = edge_index.to(device)

    model.eval()
    with torch.no_grad():
        pred_logits_test = model(X_test, edge_index).cpu()
    
    pred_probs_test = torch.sigmoid(pred_logits_test)
    
    visualize_supervised_test_data(
        test_raw_features, 
        labels,
        pred_probs_test, 
        inv_node_mapping, 
        {0: 'Temp', 1: 'Press', 2: 'Flow', 3: 'State'},
        config['analysis']['anomaly_threshold']
    )

    print("\n--- 🔥 Analyse des détections ---")
    for i in range(pred_probs_test.shape[0]):
        node_name = inv_node_mapping[i]
        true_anomalies = labels[i].sum() > 0
        detected_anomalies = (pred_probs_test[i] > config['analysis']['anomaly_threshold']).sum() > 0

        if true_anomalies and detected_anomalies:
            print(f"✅ Succès: Anomalie correctement détectée sur {node_name}")
        elif true_anomalies and not detected_anomalies:
            print(f"❌ Échec: Anomalie MANQUÉE sur {node_name} (faux négatif)")
        elif not true_anomalies and detected_anomalies:
            print(f"⚠️ Attention: Fausse alerte sur {node_name} (faux positif)")

if __name__ == '__main__':
    main()