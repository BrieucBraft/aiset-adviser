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
    """
    Standardise les caractéristiques (features) pour avoir une moyenne de 0 et un écart-type de 1.
    Le calcul se fait par caractéristique (colonne), sur tous les nœuds et tout le temps.
    
    Args:
        features_tensor (torch.Tensor): Le tenseur des données brutes.
    
    Returns:
        torch.Tensor: Le tenseur des données standardisées.
        dict: Le "scaler" contenant les moyennes et écart-types pour chaque feature.
    """
    # Calculer la moyenne et l'écart-type pour chaque feature (dimension 2)
    # en les agrégeant sur les dimensions des nœuds (0) et du temps (1)
    mean = torch.mean(features_tensor, dim=(0, 1))
    std = torch.std(features_tensor, dim=(0, 1))
    
    # Éviter la division par zéro si une feature est constante
    std[std == 0] = 1
    
    scaler = {'mean': mean, 'std': std}
    standardized_features = (features_tensor - mean) / std
    
    print("✅ Données standardisées (moyenne=0, écart-type=1).")
    return standardized_features, scaler

def prepare_data_for_model(graph, node_data):
    """Convertit les données du graphe en tenseurs PyTorch."""
    node_features_tensor = torch.tensor(node_data, dtype=torch.float32)
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edges = [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()]
    edges.extend([[v, u] for u, v in edges])
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    inv_node_mapping = {i: node for node, i in node_mapping.items()}
    return node_features_tensor, edge_index_tensor, inv_node_mapping
