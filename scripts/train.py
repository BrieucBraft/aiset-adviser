import torch
import torch.nn.functional as F
import yaml
import os
import numpy as np
import pickle

from models.st_graph_sage import STGraphSAGE
from utils.data_loader import (load_and_generate_training_data,
                               prepare_data_for_model,
                               standardize_features,
                               NUM_FEATURES, NUM_CLASSES, UNLABELED_ID)

def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Configuration chargée.")

    train_graphs, train_raw_features_list, train_labels_list = load_and_generate_training_data(
        config['data']['sequence_length']
    )
    
    train_features_scaled_list, scaler = standardize_features(train_raw_features_list)
    print("✅ Données d'entraînement préparées et standardisées.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    model = STGraphSAGE(
        in_channels=NUM_FEATURES,
        hidden_channels=config['model']['hidden_channels'],
        out_channels_classification=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    classification_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=UNLABELED_ID)
    reconstruction_loss_fn = torch.nn.MSELoss()

    print("\n--- 🚀 Début de l'entraînement semi-supervisé ---")
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
            
            pred_logits, reconstructed_X = model(X_train, edge_index)
            
            y_train_flat = y_train_labels.view(-1)
            labeled_mask = y_train_flat != UNLABELED_ID
            
            class_loss = 0
            if labeled_mask.any():
                class_loss = classification_loss_fn(pred_logits.view(-1, NUM_CLASSES)[labeled_mask], y_train_flat[labeled_mask])

            unlabeled_mask = ~labeled_mask
            recon_loss = 0
            if unlabeled_mask.any():
                recon_loss = reconstruction_loss_fn(
                    reconstructed_X.view(-1, NUM_FEATURES)[unlabeled_mask],
                    X_train.view(-1, NUM_FEATURES)[unlabeled_mask]
                )
            
            loss = class_loss + config['training'].get('reconstruction_weight', 0.5) * recon_loss
            
            if torch.is_tensor(loss) and loss != 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_graphs) if len(train_graphs) > 0 else 0
            print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], Loss: {avg_loss:.6f}')

    print("--- ✅ Entraînement terminé ---")
    os.makedirs(os.path.dirname(config['training']['weights_path']), exist_ok=True)
    torch.save(model.state_dict(), config['training']['weights_path'])
    print(f"💾 Poids du modèle sauvegardés dans : {config['training']['weights_path']}")

    scaler_path = os.path.join(os.path.dirname(config['training']['weights_path']), 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler sauvegardé dans : {scaler_path}")

if __name__ == '__main__':
    main()