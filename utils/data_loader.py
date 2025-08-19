# Fichier : utils/data_loader.py

import networkx as nx
import numpy as np
import torch

def create_hvac_building_graph():
    """Crée un exemple de graphe NetworkX pour un système CVC simple."""
    G = nx.Graph()
    
    equipments = {
        'Chiller_01': {'type': 'Chiller'},
        'Pump_CW_01': {'type': 'Pump'},   # Chilled Water Pump
        'Pump_HW_01': {'type': 'Pump'},   # Hot Water Pump
        'AHU_01': {'type': 'AHU'},       # Air Handling Unit
        'VAV_Zone_N': {'type': 'VAV'},     # North Zone
    }
    for name, attrs in equipments.items():
        G.add_node(name, **attrs)

    edges = [
        ('Chiller_01', 'Pump_CW_01'), ('Pump_CW_01', 'AHU_01'),
        ('Pump_HW_01', 'AHU_01'), ('AHU_01', 'VAV_Zone_N')
    ]
    G.add_edges_from(edges)
    
    return G

def generate_sample_time_series_data(graph, seq_length=48, num_features=4):
    """Génère des données de séries temporelles réalistes pour un système CVC."""
    nodes = list(graph.nodes())
    node_map = {name: i for i, name in enumerate(nodes)}
    features = np.zeros((len(nodes), seq_length, num_features))
    time = np.linspace(0, 2 * np.pi, seq_length)
    day_cycle = 0.5 * (1 - np.cos(time))
    on_off_state = (day_cycle > 0.5).astype(float)
    
    ranges = {
        'Chiller': {'temp': [5, 8], 'pressure': [5, 7], 'flow': [90, 100], 'state': [0, 1]},
        'Boiler': {'temp': [60, 80], 'pressure': [1.5, 2], 'flow': [85, 95], 'state': [0, 1]},
        'Pump': {'temp': [0, 0], 'pressure': [1, 4], 'flow': [80, 100], 'state': [0, 1]},
        'AHU': {'temp': [18, 22], 'pressure': [0.1, 0.2], 'flow': [1000, 1500], 'state': [0, 1]},
        'VAV': {'temp': [20, 24], 'pressure': [0.05, 0.1], 'flow': [100, 300], 'state': [0, 1]}
    }

    for i, node_name in enumerate(nodes):
        node_type = graph.nodes[node_name]['type']
        r = ranges[node_type]
        noise = lambda: np.random.randn(seq_length) * 0.05
        
        state_signal = on_off_state * r['state'][1] + noise() * 0.1
        features[i, :, 3] = np.clip(state_signal, r['state'][0], r['state'][1])
        
        pressure_signal = features[i, :, 3] * (r['pressure'][0] + (r['pressure'][1] - r['pressure'][0]) * day_cycle) + noise()
        flow_signal = features[i, :, 3] * (r['flow'][0] + (r['flow'][1] - r['flow'][0]) * day_cycle) + noise()
        features[i, :, 1] = np.clip(pressure_signal, 0, r['pressure'][1] * 1.1)
        features[i, :, 2] = np.clip(flow_signal, 0, r['flow'][1] * 1.1)
        
        if node_type in ['Chiller', 'Boiler', 'AHU', 'VAV']:
            temp_signal = r['temp'][0] + (r['temp'][1] - r['temp'][0]) * day_cycle + noise()
        elif node_type == 'Pump':
            source_node = list(graph.neighbors(node_name))[0]
            source_idx = node_map[source_node]
            source_type = graph.nodes[source_node]['type']
            source_range = ranges[source_type]
            temp_signal = source_range['temp'][0] + (source_range['temp'][1] - source_range['temp'][0]) * day_cycle + noise()
        
        features[i, :, 0] = temp_signal

    pump_hw_index = node_map.get('Pump_HW_01')
    if pump_hw_index is not None:
        start_anomaly = int(seq_length * 0.65)
        end_anomaly = int(seq_length * 0.85)
        features[pump_hw_index, start_anomaly:end_anomaly, 3] = 1.0 
        features[pump_hw_index, start_anomaly:end_anomaly, 1] *= 0.2 
        features[pump_hw_index, start_anomaly:end_anomaly, 2] *= 0.15
        print("🔧 Anomalie injectée sur 'Pump_HW_01' (chute de pression/débit).")
        
    return features

def standardize_features(features_tensor):
    """Standardise les caractéristiques."""
    mean = torch.mean(features_tensor, dim=(0, 1))
    std = torch.std(features_tensor, dim=(0, 1))
    std[std == 0] = 1
    scaler = {'mean': mean, 'std': std}
    standardized_features = (features_tensor - mean) / std
    print("✅ Données standardisées (moyenne=0, écart-type=1).")
    return standardized_features, scaler

def generate_labels(features, graph, anomaly_window):
    """Génère des labels pour l'entraînement semi-supervisé."""
    num_nodes, seq_length = features.shape[0], features.shape[1]
    # -1: non-labélisé, 0: normal, 1: anormal
    labels = -1 * np.ones((num_nodes, seq_length))
    
    # Marquer la fenêtre d'anomalie injectée comme "anormale"
    pump_hw_index = list(graph.nodes()).index('Pump_HW_01')
    start_anomaly, end_anomaly = anomaly_window
    labels[pump_hw_index, start_anomaly:end_anomaly] = 1
    
    # Marquer quelques points aléatoires comme "normaux" pour guider le modèle
    for _ in range(20): # 20 exemples normaux
        node_idx = np.random.randint(0, num_nodes)
        time_idx = np.random.randint(0, seq_length)
        # S'assurer de ne pas écraser une anomalie
        if labels[node_idx, time_idx] == -1:
            labels[node_idx, time_idx] = 0
            
    print("🏷️  Labels générés pour l'entraînement semi-supervisé.")
    return torch.tensor(labels, dtype=torch.float32)

def prepare_data_for_model(graph, node_data):
    """Convertit les données du graphe en tenseurs PyTorch."""
    node_features_tensor = torch.tensor(node_data, dtype=torch.float32)
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edges = [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()]
    edges.extend([[v, u] for u, v in edges])
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    inv_node_mapping = {i: node for node, i in node_mapping.items()}
    return node_features_tensor, edge_index_tensor, inv_node_mapping
