# Fichier : utils/data_loader.py

import networkx as nx
import numpy as np
import torch

def create_hvac_building_graph():
    """Crée un exemple de graphe NetworkX pour un système CVC simple."""
    G = nx.Graph()
    
    equipments = {
        'Chiller_01': {'type': 'Chiller'},
        'Boiler_01': {'type': 'Boiler'},
        'Pump_CW_01': {'type': 'Pump'},   # Chilled Water Pump
        'Pump_HW_01': {'type': 'Pump'},   # Hot Water Pump
        'AHU_01': {'type': 'AHU'},       # Air Handling Unit
        'VAV_Zone_N': {'type': 'VAV'},     # North Zone
        'VAV_Zone_S': {'type': 'VAV'}      # South Zone
    }
    for name, attrs in equipments.items():
        G.add_node(name, **attrs)

    edges = [
        ('Chiller_01', 'Pump_CW_01'), ('Pump_CW_01', 'AHU_01'),
        ('Boiler_01', 'Pump_HW_01'), ('Pump_HW_01', 'AHU_01'),
        ('AHU_01', 'VAV_Zone_N'), ('AHU_01', 'VAV_Zone_S')
    ]
    G.add_edges_from(edges)
    
    return G

def generate_sample_time_series_data(graph, seq_length=48, num_features=4):
    """Génère des données de séries temporelles réalistes pour un système CVC."""
    
    nodes = list(graph.nodes())
    node_map = {name: i for i, name in enumerate(nodes)}
    features = np.zeros((len(nodes), seq_length, num_features))
    
    time = np.linspace(0, 2 * np.pi, seq_length) # Cycle de 24h
    
    # Définir des plages de fonctionnement réalistes [min, max]
    ranges = {
        'Chiller': {'temp': [5, 8], 'pressure': [5, 7], 'flow': [90, 100], 'state': [0, 1]},
        'Boiler': {'temp': [60, 80], 'pressure': [1.5, 2], 'flow': [85, 95], 'state': [0, 1]},
        'Pump': {'temp': [0, 0], 'pressure': [1, 4], 'flow': [80, 100], 'state': [0, 1]}, # Temp sera héritée
        'AHU': {'temp': [18, 22], 'pressure': [0.1, 0.2], 'flow': [1000, 1500], 'state': [0, 1]},
        'VAV': {'temp': [20, 24], 'pressure': [0.05, 0.1], 'flow': [100, 300], 'state': [0, 1]}
    }
    
    # Simuler un cycle jour/nuit (ON pendant les heures de travail)
    day_cycle = 0.5 * (1 - np.cos(time)) # Cycle doux de 0 à 1
    on_off_state = (day_cycle > 0.5).astype(float)
    
    for i, node_name in enumerate(nodes):
        node_type = graph.nodes[node_name]['type']
        r = ranges[node_type]
        noise = lambda: np.random.randn(seq_length) * 0.05
        
        # État (Feature 3)
        state_signal = on_off_state * r['state'][1] + noise() * 0.1
        features[i, :, 3] = np.clip(state_signal, r['state'][0], r['state'][1])
        
        # Pression et Débit (Features 1 et 2)
        pressure_signal = features[i, :, 3] * (r['pressure'][0] + (r['pressure'][1] - r['pressure'][0]) * day_cycle) + noise()
        flow_signal = features[i, :, 3] * (r['flow'][0] + (r['flow'][1] - r['flow'][0]) * day_cycle) + noise()
        features[i, :, 1] = np.clip(pressure_signal, 0, r['pressure'][1] * 1.1)
        features[i, :, 2] = np.clip(flow_signal, 0, r['flow'][1] * 1.1)
        
        # Température (Feature 0)
        if node_type in ['Chiller', 'Boiler', 'AHU', 'VAV']:
            temp_signal = r['temp'][0] + (r['temp'][1] - r['temp'][0]) * day_cycle + noise()
        elif node_type == 'Pump':
            # La pompe hérite de la température de sa source
            source_node = list(graph.neighbors(node_name))[0]
            source_idx = node_map[source_node]
            source_type = graph.nodes[source_node]['type']
            source_range = ranges[source_type]
            temp_signal = source_range['temp'][0] + (source_range['temp'][1] - source_range['temp'][0]) * day_cycle + noise()
        
        features[i, :, 0] = temp_signal

    # Injecter une anomalie réaliste
    # La pompe à eau chaude ('Pump_HW_01') a une chute de pression et de débit
    # alors qu'elle est censée être ON.
    pump_hw_index = node_map.get('Pump_HW_01')
    if pump_hw_index is not None:
        start_anomaly = int(seq_length * 0.65)
        end_anomaly = int(seq_length * 0.85)
        # S'assurer que la pompe est ON pendant l'anomalie pour que ce soit visible
        features[pump_hw_index, start_anomaly:end_anomaly, 3] = 1.0 
        # Chute de pression et débit
        features[pump_hw_index, start_anomaly:end_anomaly, 1] *= 0.2 
        features[pump_hw_index, start_anomaly:end_anomaly, 2] *= 0.15
        print("🔧 Anomalie injectée sur 'Pump_HW_01' (chute de pression/débit).")
        
    return features

def prepare_data_for_model(graph, node_data):
    """Convertit les données du graphe en tenseurs PyTorch."""
    node_features_tensor = torch.tensor(node_data, dtype=torch.float32)
    
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edges = [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()]
    edges.extend([[v, u] for u, v in edges])
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    inv_node_mapping = {i: node for node, i in node_mapping.items()}
    
    return node_features_tensor, edge_index_tensor, inv_node_mapping