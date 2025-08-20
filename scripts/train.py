import torch
import torch.nn.functional as F
import yaml
import os
import numpy as np

from models.st_graph_sage import STGraphSAGE
from utils.data_loader import (load_and_generate_training_data,
                               load_and_generate_test_data,
                               prepare_data_for_model,
                               standardize_features,
                               FEATURE_MAP, NUM_FEATURES, NUM_CLASSES, ANOMALY_TYPES)
from utils.visualization import (visualize_building_graph,
                                 visualize_classification_test_data)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    train_graphs, train_raw_features_list, train_labels_list = load_and_generate_training_data(
        config['data']['sequence_length']
    )
    
    train_features_scaled_list, scaler = standardize_features(train_raw_features_list)
    print("✅ Données d'entraînement préparées et standardisées.")
    
    for i, graph in enumerate(train_graphs):
        visualize_building_graph(graph, f"training_topology_{i+1}.html")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    model = STGraphSAGE(
        in_channels=NUM_FEATURES,
        hidden_channels=config['model']['hidden_channels'],
        out_channels=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()

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
            
            loss = loss_fn(pred_logits.view(-1, NUM_CLASSES), y_train_labels.view(-1))
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
        config['data']['sequence_length']
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
    
    pred_probs_test = F.softmax(pred_logits_test, dim=2)
    pred_classes_test = torch.argmax(pred_probs_test, dim=2)
    
    visualize_classification_test_data(
        test_graph,
        test_raw_features, 
        labels,
        pred_classes_test, 
        inv_node_mapping, 
        FEATURE_MAP,
        ANOMALY_TYPES
    )

    print("\n--- 🔥 Analyse des détections ---")
    inv_anomaly_map = {v: k for k, v in ANOMALY_TYPES.items()}
    for i in range(pred_classes_test.shape[0]):
        node_name = inv_node_mapping[i]
        true_anomaly_class = labels[i, -1].item()
        pred_anomaly_class = pred_classes_test[i, -1].item()

        if true_anomaly_class != 0:
            true_class_name = inv_anomaly_map[true_anomaly_class]
            pred_class_name = inv_anomaly_map[pred_anomaly_class]
            if true_anomaly_class == pred_anomaly_class:
                print(f"✅ Succès: Anomalie '{true_class_name}' correctement classifiée sur {node_name}")
            else:
                print(f"❌ Échec: Anomalie '{true_class_name}' sur {node_name} classifiée comme '{pred_class_name}'")

if __name__ == '__main__':
    main()