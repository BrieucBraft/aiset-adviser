import torch
import torch.nn.functional as F
import yaml
import os
import numpy as np
import pickle

from models.st_graph_sage import STGraphSAGE
from utils.data_loader import (load_and_generate_test_data,
                               prepare_data_for_model,
                               FEATURE_MAP, NUM_FEATURES, NUM_CLASSES, ANOMALY_TYPES)
from utils.visualization import (visualize_building_graph,
                                 visualize_classification_test_data)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    test_graph, test_raw_features, test_labels = load_and_generate_test_data(
        config['data']['sequence_length']
    )
    
    visualize_building_graph(test_graph, "test_topology.html")

    scaler_path = os.path.join(os.path.dirname(config['training']['weights_path']), 'scaler.pkl')
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler chargé depuis : {scaler_path}")
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier scaler non trouvé à l'emplacement '{scaler_path}'. Veuillez d'abord entraîner le modèle.")
        return

    test_features_scaled = (test_raw_features - scaler['mean']) / scaler['std']
    print("✅ Données de test standardisées avec le scaler de l'entraînement.")

    features_scaled, labels, edge_index, inv_node_mapping = prepare_data_for_model(
        test_graph, test_features_scaled, test_labels
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    model = STGraphSAGE(
        in_channels=NUM_FEATURES,
        hidden_channels=config['model']['hidden_channels'],
        out_channels_classification=NUM_CLASSES
    ).to(device)

    model.load_state_dict(torch.load(config['training']['weights_path'], weights_only=True))
    print(f"✅ Poids du modèle chargés depuis : {config['training']['weights_path']}")
    
    print("\n--- 🔍 Évaluation sur le graphe de test ---")
    X_test = features_scaled.to(device)
    edge_index = edge_index.to(device)

    model.eval()
    with torch.no_grad():
        pred_logits_test, _ = model(X_test, edge_index)
        pred_logits_test = pred_logits_test.cpu()
    
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
        
        is_true_anomaly = torch.any(labels[i] != ANOMALY_TYPES['NORMAL'])
        
        if is_true_anomaly:
            true_anomaly_class = labels[i][labels[i] != ANOMALY_TYPES['NORMAL']][0].item()
            
            anomaly_period_preds = pred_classes_test[i][labels[i] != ANOMALY_TYPES['NORMAL']]
            pred_anomaly_class = torch.mode(anomaly_period_preds).values.item()

            true_class_name = inv_anomaly_map[true_anomaly_class]
            pred_class_name = inv_anomaly_map[pred_anomaly_class]
            
            if true_anomaly_class == pred_anomaly_class:
                print(f"✅ Succès: Anomalie '{true_class_name}' correctement classifiée sur {node_name}")
            else:
                print(f"❌ Échec: Anomalie '{true_class_name}' sur {node_name} classifiée comme '{pred_class_name}'")

if __name__ == '__main__':
    main()