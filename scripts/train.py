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
from utils.visualization import (visualize_building_graph, 
                                 visualize_train_test_split,
                                 visualize_anomaly_scores)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

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

    print("\n--- 🚀 Début de l'entraînement semi-supervisé ---")
    model.train()
    for epoch in range(config['training']['epochs']):
        for i, (batch_data, batch_labels) in enumerate(train_loader):
            if i >= config['training']['num_batches_per_epoch']: break
            
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            num_nodes = batch_data.shape[1]
            batch_size = batch_data.shape[0]

            permuted_data = batch_data.permute(1, 0, 2, 3)
            X = permuted_data[:, :, :-1, :]
            y_true_recon = permuted_data[:, :, 1:, :]
            
            seq_len_train = X.shape[2]
            X = X.permute(1, 0, 2, 3).contiguous().view(batch_size * num_nodes, seq_len_train, -1)
            y_true_recon = y_true_recon.permute(1, 0, 2, 3).contiguous().view(batch_size * num_nodes, seq_len_train, -1)
            
            y_true_class = batch_labels.flatten()

            edge_index_batch = edge_index.clone().to(device)
            if batch_size > 1:
                edge_offsets = torch.arange(0, batch_size).to(device) * num_nodes
                edge_index_batch = torch.cat([edge_index.clone().to(device) + offset for offset in edge_offsets], dim=1)

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

    print("\n--- 🔍 Évaluation sur les données de test ---")
    model.eval()
    with torch.no_grad():
        test_scaled = (test_raw - scaler['mean']) / scaler['std']
        
        X_test = test_scaled.permute(1, 0, 2).contiguous().view(1 * test_scaled.shape[0], test_scaled.shape[1], -1)
        X_test = X_test[:, :-1, :]
        
        _, y_pred_test_class = model(X_test.to(device), edge_index.to(device))
        y_pred_test_class = y_pred_test_class[:, -1].cpu()

    anomaly_scores = y_pred_test_class
    visualize_anomaly_scores(anomaly_scores, inv_node_mapping, config['analysis']['anomaly_threshold'])

    print("\n--- 📊 Génération du graphique long-terme (moyennes journalières) ---")
    
    full_raw_data_with_anomalies = torch.cat([train_raw, test_raw], dim=1)
    full_scaled_data = (full_raw_data_with_anomalies - scaler['mean']) / scaler['std']
    
    with torch.no_grad():
        X_full = full_scaled_data.permute(1, 0, 2).contiguous().view(1 * full_scaled_data.shape[0], full_scaled_data.shape[1], -1)
        X_full = X_full[:, :-1, :]
        
        full_pred_recon_scaled, _ = model(X_full.to(device), edge_index.to(device))
        full_pred_recon_scaled = full_pred_recon_scaled.cpu()
        
    mean = scaler['mean'].unsqueeze(0)
    std = scaler['std'].unsqueeze(0)
    
    reshaped_pred = full_pred_recon_scaled.view(full_scaled_data.shape[0], -1, full_scaled_data.shape[2])
    full_pred_unscaled = reshaped_pred * std + mean
    
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