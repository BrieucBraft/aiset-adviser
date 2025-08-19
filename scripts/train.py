# Fichier : scripts/train.py

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import TensorDataset, DataLoader

from models.st_graph_sage import STGraphSAGE
from utils.data_loader import (create_hvac_building_graph, 
                               generate_long_term_time_series, 
                               prepare_data_for_model,
                               standardize_features,
                               inject_anomalies_and_labels,
                               create_sequences)
# CORRECTION : Utiliser le bon nom de fonction importée
from utils.visualization import (visualize_building_graph, 
                                 visualize_train_test_split,
                                 visualize_anomaly_scores)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    # --- 2. Générer et préparer les données ---
    building_graph = create_hvac_building_graph()
    raw_features_np = generate_long_term_time_series(
        building_graph, config['data']['total_years'], config['data']['hourly_steps']
    )
    
    total_steps = raw_features_np.shape[1]
    split_idx = int(total_steps * (config['data']['train_years'] / config['data']['total_years']))
    
    raw_features_with_anomalies, full_labels = inject_anomalies_and_labels(raw_features_np, building_graph, split_idx)
    
    features_raw = torch.tensor(raw_features_with_anomalies, dtype=torch.float32)
    train_raw = features_raw[:, :split_idx, :]
    test_raw = features_raw[:, split_idx:, :]
    
    train_scaled, scaler = standardize_features(train_raw)
    _, edge_index, inv_node_mapping = prepare_data_for_model(building_graph, raw_features_np)
    
    train_labels = full_labels[:, :split_idx]
    train_sequences, train_sequence_labels = create_sequences(train_scaled, train_labels, config['data']['training_sequence_length'])
    
    train_dataset = TensorDataset(train_sequences, train_sequence_labels)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    print(f"✅ Données long-terme préparées.")
    visualize_building_graph(building_graph)

    # --- 3. Initialiser le modèle et les optimiseurs ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    model = STGraphSAGE(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    recon_loss_fn = torch.nn.MSELoss()
    class_loss_fn = torch.nn.BCELoss()

    # --- 4. Boucle d'entraînement HYBRIDE ---
    print("\n--- 🚀 Début de l'entraînement semi-supervisé ---")
    model.train()
    for epoch in range(config['training']['epochs']):
        for i, (batch_data, batch_labels) in enumerate(train_loader):
            if i >= config['training']['num_batches_per_epoch']: break
            
            # --- SECTION CORRIGÉE ---
            # batch_data arrive en [batch, nodes, seq, feats]
            # batch_labels arrive en [batch, nodes]
            
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            num_nodes = batch_data.shape[1]
            batch_size = batch_data.shape[0]

            # Préparer les données d'entrée et cibles pour la reconstruction
            # On permute pour mettre la dimension des noeuds en premier : [nodes, batch, seq, feats]
            X = batch_data.permute(1, 0, 2, 3)
            X = X[:, :, :-1, :] # Garder tous sauf le dernier pas de temps
            y_true_recon = X[:, :, 1:, :] # Cible : décalée d'un pas de temps
            
            # Remodeler pour le modèle : [nodes*batch, seq, feats]
            seq_len_train = X.shape[2]
            X = X.permute(1, 0, 2, 3).contiguous().view(batch_size * num_nodes, seq_len_train, -1)
            y_true_recon = y_true_recon.permute(1, 0, 2, 3).contiguous().view(batch_size * num_nodes, seq_len_train, -1)
            
            # Préparer les labels pour la classification (simplement aplatir)
            y_true_class = batch_labels.flatten() # Shape [batch*nodes]

            # Adapter l'edge_index pour le batch
            edge_index_batch = edge_index.clone().to(device)
            if batch_size > 1:
                edge_offsets = torch.arange(1, batch_size).to(device) * num_nodes
                for offset in edge_offsets:
                    edge_index_batch = torch.cat([edge_index_batch, edge_index.clone().to(device) + offset], dim=1)
            # --- FIN DE LA SECTION CORRIGÉE ---

            optimizer.zero_grad()
            y_pred_recon, y_pred_class = model(X, edge_index_batch)
            y_pred_class = y_pred_class[:, -1]

            recon_loss = recon_loss_fn(y_pred_recon, y_true_recon)
            
            labeled_mask = (y_true_class != -1)
            if labeled_mask.any():
                class_loss = class_loss_fn(y_pred_class[labeled_mask], y_true_class[labeled_mask])
            else:
                class_loss = torch.tensor(0.0, device=device)
            
            total_loss = recon_loss + config['training']['loss_lambda'] * class_loss
            total_loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], Loss: {total_loss.item():.6f} (Recon: {recon_loss.item():.6f}, Class: {class_loss.item():.6f})')
        
    print("--- ✅ Entraînement terminé ---")
    torch.save(model.state_dict(), config['training']['weights_path'])
    print(f"💾 Poids du modèle sauvegardés dans : {config['training']['weights_path']}")

    # --- 5. Évaluation et Analyse des anomalies sur le jeu de test ---
    print("\n--- 🔍 Évaluation sur les données de test ---")
    model.eval()
    with torch.no_grad():
        test_scaled = (test_raw - scaler['mean']) / scaler['std']
        
        X_test = test_scaled[:, :-1, :].to(device)
        
        _, y_pred_test_class = model(X_test, edge_index.to(device))
        y_pred_test_class = y_pred_test_class.cpu()

    anomaly_scores = y_pred_test_class.mean(dim=1)
    visualize_anomaly_scores(anomaly_scores, inv_node_mapping, config['analysis']['anomaly_threshold'])

    # --- 6. Visualisation de la période complète ---
    print("\n--- 📊 Génération du graphique long-terme (moyennes journalières) ---")
    
    full_raw_data_with_anomalies = torch.cat([train_raw, test_raw], dim=1)
    full_scaled_data = (full_raw_data_with_anomalies - scaler['mean']) / scaler['std']
    
    with torch.no_grad():
        full_pred_recon_scaled, _ = model(full_scaled_data[:, :-1, :].to(device), edge_index.to(device))
        full_pred_recon_scaled = full_pred_recon_scaled.cpu()
        
    mean = scaler['mean'].unsqueeze(0).unsqueeze(0)
    std = scaler['std'].unsqueeze(0).unsqueeze(0)
    full_pred_unscaled = full_pred_recon_scaled * std + mean
    
    full_true_raw = full_raw_data_with_anomalies[:, 1:, :]
    
    hourly_steps = config['data']['hourly_steps']
    num_timesteps = full_true_raw.shape[1]
    trunc_len = (num_timesteps // hourly_steps) * hourly_steps
    
    full_true_raw_trunc = full_true_raw[:, :trunc_len, :]
    full_pred_unscaled_trunc = full_pred_unscaled[:, :trunc_len, :]
    full_labels_trunc = full_labels[:, 1:][:, :trunc_len]

    num_days = trunc_len // hourly_steps
    daily_avg_full = full_true_raw_trunc.reshape(full_true_raw_trunc.shape[0], num_days, hourly_steps, -1).mean(dim=2)
    daily_avg_pred = full_pred_unscaled_trunc.reshape(full_pred_unscaled_trunc.shape[0], num_days, hourly_steps, -1).mean(dim=2)
    daily_labels = full_labels_trunc.reshape(full_labels_trunc.shape[0], num_days, hourly_steps).max(dim=2).values
    train_idx_daily = split_idx // hourly_steps

    # CORRECTION : Utiliser le bon nom de fonction
    visualize_train_test_split(
        full_features=daily_avg_full,
        y_pred_unscaled=daily_avg_pred,
        labels=daily_labels,
        train_idx=train_idx_daily,
        inv_node_mapping=inv_node_mapping,
        feature_map={0: 'Temp', 1: 'Press', 2: 'Flow', 3: 'State'}
    )

if __name__ == '__main__':
    main()