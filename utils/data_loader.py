import networkx as nx
import numpy as np
import torch
import pandas as pd

def create_hvac_building_graph():
    G = nx.Graph()
    equipments = {
        'Boiler_01': {'type': 'Boiler'},
        'Pump_HW_01': {'type': 'Pump'},
        'AHU_01': {'type': 'AHU'}
    }
    for name, attrs in equipments.items():
        G.add_node(name, **attrs)

    edges = [('Boiler_01', 'Pump_HW_01'), ('Pump_HW_01', 'AHU_01')]
    G.add_edges_from(edges)
    
    print(f"✅ Graphe simplifié créé avec {G.number_of_nodes()} nœuds.")
    return G

def generate_long_term_time_series(graph, total_years=4, hourly_steps=24):
    num_nodes = graph.number_of_nodes()
    num_features = 4
    total_days = int(365.25 * total_years)
    total_steps = total_days * hourly_steps
    time_index = pd.date_range(start='2021-01-01', periods=total_steps, freq='h')

    day_of_year = time_index.dayofyear
    hour_of_day = time_index.hour
    yearly_cycle = -np.cos(2 * np.pi * day_of_year / 365.25)
    daily_cycle = -np.cos(2 * np.pi * hour_of_day / 24)
    weekly_cycle = (time_index.dayofweek < 5).astype(float)

    features = np.zeros((num_nodes, total_steps, num_features))
    node_map = {name: i for i, name in enumerate(graph.nodes())}
    
    boiler_idx = node_map.get('Boiler_01')
    if boiler_idx is not None:
        activation = np.clip(-yearly_cycle, 0, 1) * np.clip(daily_cycle, 0, 1) * weekly_cycle
        features[boiler_idx, :, 3] = (activation > 0.3).astype(float) + np.random.randn(total_steps) * 0.05
        features[boiler_idx, :, 0] = features[boiler_idx, :, 3] * (70 + 10 * activation)
        features[boiler_idx, :, 1] = features[boiler_idx, :, 3] * (1.5 + 0.5 * activation)
        features[boiler_idx, :, 2] = features[boiler_idx, :, 3] * (90 + 10 * activation)

    pump_idx = node_map.get('Pump_HW_01')
    if pump_idx is not None and boiler_idx is not None:
        features[pump_idx, :, 3] = features[boiler_idx, :, 3]
        features[pump_idx, :, 0] = features[boiler_idx, :, 0] + np.random.randn(total_steps) * 0.2
        features[pump_idx, :, 1] = features[pump_idx, :, 3] * (3 + 1 * np.clip(daily_cycle, 0, 1))
        features[pump_idx, :, 2] = features[pump_idx, :, 3] * (90 + 10 * np.clip(daily_cycle, 0, 1))
        
    ahu_idx = node_map.get('AHU_01')
    if ahu_idx is not None:
        activation = (np.clip(daily_cycle, 0, 1) * weekly_cycle > 0.3).astype(float)
        features[ahu_idx, :, 3] = activation
        features[ahu_idx, :, 0] = 21 + 2 * daily_cycle * activation
        features[ahu_idx, :, 1] = activation * 0.15
        features[ahu_idx, :, 2] = activation * (200 + 100 * np.random.rand(total_steps))

    return features

def inject_anomalies_and_labels(features, graph, train_split_idx):
    num_nodes, total_steps = features.shape[0], features.shape[1]
    labels = -1 * np.ones_like(features[:, :, 0])
    node_map = {name: i for i, name in enumerate(graph.nodes())}
    
    print("🏷️  Labélisation de périodes normales sporadiques...")
    for _ in range(5):
        node_idx = np.random.randint(0, num_nodes)
        start_idx = np.random.randint(0, train_split_idx - 48)
        end_idx = start_idx + 24
        labels[node_idx, start_idx:end_idx] = 0

    print("🔧 Injection d'anomalies sporadiques...")
    test_start_offset = train_split_idx
    test_duration = total_steps - train_split_idx
    
    pump_idx = node_map.get('Pump_HW_01')
    if pump_idx is not None:
        anomaly_start = test_start_offset + np.random.randint(0, test_duration - 4)
        anomaly_end = anomaly_start + 3
        features[pump_idx, anomaly_start:anomaly_end, 1] *= 2.5
        labels[pump_idx, anomaly_start:anomaly_end] = 1
    
    ahu_idx = node_map.get('AHU_01')
    if ahu_idx is not None:
        anomaly_start = test_start_offset + np.random.randint(0, test_duration - 25)
        anomaly_end = anomaly_start + 24
        features[ahu_idx, anomaly_start:anomaly_end, 2] = 5.0
        labels[ahu_idx, anomaly_start:anomaly_end] = 1
        
    return features, torch.tensor(labels, dtype=torch.float32)

def create_sequences(data, labels, seq_length):
    data_sequences, label_sequences = [], []
    num_nodes, total_len, num_features = data.shape
    
    for i in range(total_len - seq_length + 1):
        end_idx = i + seq_length
        if end_idx > total_len:
            break
        data_sequences.append(data[:, i:end_idx, :])
        label_sequences.append(labels[:, end_idx-1])
        
    return torch.stack(data_sequences).permute(1, 0, 2, 3), torch.stack(label_sequences).permute(1, 0)

def standardize_features(features_tensor):
    mean = torch.mean(features_tensor, dim=(0, 1))
    std = torch.std(features_tensor, dim=(0, 1))
    std[std == 0] = 1.0
    scaler = {'mean': mean, 'std': std}
    standardized_features = (features_tensor - mean) / std
    print("✅ Données standardisées.")
    return standardized_features, scaler

def prepare_data_for_model(graph, node_data):
    node_features_tensor = torch.tensor(node_data, dtype=torch.float32)
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edges = [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()]
    edges.extend([[v, u] for u, v in edges])
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    inv_node_mapping = {i: node for node, i in node_mapping.items()}
    return node_features_tensor, edge_index_tensor, inv_node_mapping