# Fichier : scripts/train.py
import torch
import torch.nn.functional as F
import yaml

from models.st_graph_sage import STGraphSAGE
from utils.data_loader import (create_hvac_building_graph, 
                               generate_sample_time_series_data, 
                               prepare_data_for_model,
                               standardize_features,
                               generate_labels)
from utils.visualization import (visualize_building_graph, 
                                 visualize_training_data,
                                 visualize_anomaly_scores)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    building_graph = create_hvac_building_graph()
    raw_node_data = generate_sample_time_series_data(building_graph, config['data']['sequence_length'], config['model']['in_channels'])
    features_raw, edge_index, inv_node_mapping = prepare_data_for_model(building_graph, raw_node_data)
    features_scaled, scaler = standardize_features(features_raw)
    
    # Générer les labels
    anomaly_window = (int(config['data']['sequence_length'] * 0.65), int(config['data']['sequence_length'] * 0.85))
    labels = generate_labels(features_raw, building_graph, anomaly_window)

    visualize_building_graph(building_graph)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    model = STGraphSAGE(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    recon_loss_fn = torch.nn.MSELoss()
    class_loss_fn = torch.nn.BCELoss() # Binary Cross-Entropy pour la classification

    # --- SÉPARATION TRAIN/TEST ---
    split_idx = int(config['data']['sequence_length'] * config['training']['train_split_ratio'])
    
    # Données d'entraînement (entrée et cibles)
    X_train = features_scaled[:, :split_idx-1, :].to(device)
    y_recon_true_train = features_scaled[:, 1:split_idx, :].to(device)
    y_class_true_train = labels[:, 1:split_idx].to(device)
    edge_index = edge_index.to(device)
    
    # --- BOUCLE D'ENTRAÎNEMENT HYBRIDE ---
    print("\n--- 🚀 Début de l'entraînement semi-supervisé ---")
    model.train()
    for epoch in range(config['training']['epochs']):
        optimizer.zero_grad()
        
        y_recon_pred, y_class_pred = model(X_train, edge_index)
        
        # 1. Calculer la perte de reconstruction sur toutes les données
        recon_loss = recon_loss_fn(y_recon_pred, y_recon_true_train)
        
        # 2. Calculer la perte de classification UNIQUEMENT sur les points labélisés
        labeled_mask = (y_class_true_train != -1)
        if labeled_mask.any():
            class_loss = class_loss_fn(y_class_pred[labeled_mask], y_class_true_train[labeled_mask])
        else:
            class_loss = torch.tensor(0.0).to(device) # Pas de perte si aucun label
        
        # 3. Combiner les pertes
        total_loss = recon_loss + config['training']['loss_lambda'] * class_loss
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 15 == 0:
            print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], Total Loss: {total_loss.item():.6f} (Recon: {recon_loss.item():.6f}, Class: {class_loss.item():.6f})')

    print("--- ✅ Entraînement terminé ---")
    torch.save(model.state_dict(), config['training']['weights_path'])
    print(f"💾 Poids du modèle sauvegardés dans : {config['training']['weights_path']}")
    
    # --- ÉVALUATION SUR LE JEU DE TEST ---
    model.eval()
    with torch.no_grad():
        # Utiliser les données de test pour l'inférence
        X_test = features_scaled[:, split_idx:-1, :].to(device)
        y_recon_pred_test, _ = model(X_test, edge_index)
        
        # Données réelles pour la comparaison
        y_recon_true_test = features_scaled[:, split_idx+1:, :]

    # Dés-standardiser pour la visualisation
    mean = scaler['mean'].unsqueeze(0).unsqueeze(0)
    std = scaler['std'].unsqueeze(0).unsqueeze(0)
    y_pred_unscaled = y_recon_pred_test.cpu() * std + mean
    
    visualize_training_data(features_raw, y_pred_unscaled, labels, split_idx, inv_node_mapping, {0: 'Temp', 1: 'Press', 2: 'Flow', 3: 'State'})
    
    # L'analyse d'anomalie se fait maintenant sur l'erreur du jeu de test
    error_per_node = F.mse_loss(y_recon_pred_test.cpu(), y_recon_true_test, reduction='none').mean(dim=(1, 2))
    visualize_anomaly_scores(error_per_node, inv_node_mapping, config['analysis']['anomaly_threshold'])

if __name__ == '__main__':
    main()