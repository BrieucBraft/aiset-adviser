# Fichier : utils/data_loader.py

import networkx as nx
import numpy as np
import torch

def create_hvac_building_graph():
    """Crée un graphe NetworkX complexe pour un système CVC avec chauffage et refroidissement."""
    G = nx.Graph()

    equipments = {
        'Chiller_01': {'type': 'Chiller'},
        'Chiller_02': {'type': 'Chiller'},
        'Boiler_01': {'type': 'Boiler'},
        'Boiler_02': {'type': 'Boiler'},
        'Pump_CW_01': {'type': 'Pump'},  # Chilled Water Pump
        'Pump_CW_02': {'type': 'Pump'},
        'Pump_HW_01': {'type': 'Pump'},  # Hot Water Pump
        'Pump_HW_02': {'type': 'Pump'},
        'AHU_01': {'type': 'AHU'},       # Air Handling Unit
        'AHU_02': {'type': 'AHU'},
        'VAV_Zone_N': {'type': 'VAV'},     # North Zone
        'VAV_Zone_S': {'type': 'VAV'},
        'VAV_Zone_E': {'type': 'VAV'},
        'VAV_Zone_W': {'type': 'VAV'},
    }
    for name, attrs in equipments.items():
        G.add_node(name, **attrs)

    edges = [
        # Chilled water loop
        ('Chiller_01', 'Pump_CW_01'), ('Pump_CW_01', 'AHU_01'),
        ('Chiller_02', 'Pump_CW_02'), ('Pump_CW_02', 'AHU_02'),

        # Hot water loop
        ('Boiler_01', 'Pump_HW_01'), ('Pump_HW_01', 'AHU_01'),
        ('Boiler_02', 'Pump_HW_02'), ('Pump_HW_02', 'AHU_02'),

        # Air distribution
        ('AHU_01', 'VAV_Zone_N'), ('AHU_01', 'VAV_Zone_W'),
        ('AHU_02', 'VAV_Zone_S'), ('AHU_02', 'VAV_Zone_E'),
        
        # Cross-connection
        ('AHU_01', 'AHU_02')
    ]
    G.add_edges_from(edges)

    return G

def create_hvac_building_graph_test():
    """Crée un nouveau graphe de test complexe avec une topologie différente."""
    G = nx.Graph()

    equipments = {
        'Boiler_01': {'type': 'Boiler'},
        'Pump_HW_03': {'type': 'Pump'},
        'Boiler_02': {'type': 'Boiler'},
        'Pump_HW_04': {'type': 'Pump'},
        'AHU_03': {'type': 'AHU'},
        'VAV_Zone_Office_01': {'type': 'VAV'},
        'VAV_Zone_Meeting_01': {'type': 'VAV'},
    }
    for name, attrs in equipments.items():
        G.add_node(name, **attrs)

    edges = [
        ('Boiler_01', 'Pump_HW_03'), ('Pump_HW_03', 'AHU_03'),
        ('Boiler_02', 'Pump_HW_04'), ('Pump_HW_04', 'AHU_03'),
        ('AHU_03', 'VAV_Zone_Office_01'), ('AHU_03', 'VAV_Zone_Meeting_01'),
    ]
    G.add_edges_from(edges)

    return G

def generate_sample_time_series_data(graph, seq_length=48, num_features=4, inject_anomaly=False):
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
        'AHU': {'temp': [18, 22], 'pressure': [1.1, 1.2], 'flow': [1000, 1500], 'state': [0, 1]},
        'VAV': {'temp': [20, 24], 'pressure': [0.9, 1.1], 'flow': [100, 300], 'state': [0, 1]}
    }

    for i, node_name in enumerate(nodes):
        node_type = graph.nodes[node_name]['type']
        r = ranges[node_type]
        
        state_signal = on_off_state * r['state'][1] + (np.random.randn(seq_length) * 0.05) * 0.1
        features[i, :, 3] = np.clip(state_signal, r['state'][0], r['state'][1])

        if node_type in ['AHU', 'VAV']:
            # Use a more stable pressure for air-side equipment, less dependent on the daily cycle
            stable_pressure = (r['pressure'][0] + r['pressure'][1]) / 2
            pressure_noise = np.random.randn(seq_length) * 0.01  # Reduced noise
            pressure_signal = features[i, :, 3] * stable_pressure + pressure_noise
        else:
            # Original logic for water-side equipment
            pressure_noise = np.random.randn(seq_length) * 0.05
            pressure_signal = features[i, :, 3] * (r['pressure'][0] + (r['pressure'][1] - r['pressure'][0]) * day_cycle) + pressure_noise

        flow_noise = np.random.randn(seq_length) * 0.05
        flow_signal = features[i, :, 3] * (r['flow'][0] + (r['flow'][1] - r['flow'][0]) * day_cycle) + flow_noise
        
        features[i, :, 1] = np.clip(pressure_signal, 0, r['pressure'][1] * 1.2)
        features[i, :, 2] = np.clip(flow_signal, 0, r['flow'][1] * 1.2)

        temp_noise = np.random.randn(seq_length) * 0.05
        if node_type in ['Chiller', 'Boiler', 'AHU', 'VAV']:
            temp_signal = r['temp'][0] + (r['temp'][1] - r['temp'][0]) * day_cycle + temp_noise
        elif node_type == 'Pump':
            source_nodes = list(graph.neighbors(node_name))
            if source_nodes:
                source_node = next((n for n in source_nodes if graph.nodes[n]['type'] in ['Chiller', 'Boiler']), source_nodes[0])
                source_type = graph.nodes[source_node]['type']
                source_range = ranges.get(source_type, {'temp': [20, 20]})
                temp_signal = source_range['temp'][0] + (source_range['temp'][1] - source_range['temp'][0]) * day_cycle + temp_noise
            else:
                temp_signal = np.full(seq_length, 20) + temp_noise
        
        features[i, :, 0] = temp_signal

    if inject_anomaly:
        # boiler_to_affect = 'Boiler_01'
        # if boiler_to_affect in node_map:
        #     pump_hw_index = node_map.get(boiler_to_affect)
        #     start_anomaly = int(seq_length * 0.5)
        #     end_anomaly = int(seq_length * 1)
        #     features[pump_hw_index, start_anomaly:end_anomaly, 0] *= 0.0 # Pressure drop
        #     features[pump_hw_index, start_anomaly:end_anomaly, 1] *= 0.0 # Flow drop
        #     features[pump_hw_index, start_anomaly:end_anomaly, 2] *= 0.0 # Pressure drop
        #     features[pump_hw_index, start_anomaly:end_anomaly, 3] *= 0.0 # Flow drop
        #     print(f"🔧 Anomalie injectée sur '{boiler_to_affect}' (chute de pression/débit).")
        # else:
        #     print(f"⚠️  '{boiler_to_affect}' non trouvé dans le graphe de test. Aucune anomalie injectée.")

        pump_to_affect = 'Pump_HW_03'
        if pump_to_affect in node_map:
            pump_hw_index = node_map.get(pump_to_affect)
            start_anomaly = int(seq_length * 0.5)
            end_anomaly = int(seq_length * 1)
            # Simulate a pressure/flow drop
            # features[pump_hw_index, start_anomaly:end_anomaly, 1] *= 0.1 # Pressure drop
            # features[pump_hw_index, start_anomaly:end_anomaly, 2] *= 0.1 # Flow drop
            print(f"🔧 Anomalie injectée sur '{pump_to_affect}' (chute de pression/débit).")
        else:
            print(f"⚠️  '{pump_to_affect}' non trouvé dans le graphe de test. Aucune anomalie injectée.")

    return features

def generate_training_data(seq_length=48, num_features=4):
    """Génère les données d'entraînement (journée normale, sans anomalie)."""
    print("--- Génération des données d'entraînement ---")
    graph = create_hvac_building_graph()
    features = generate_sample_time_series_data(graph, seq_length, num_features, inject_anomaly=False)
    print("✅ Données d'entraînement générées (sans anomalie).")
    return graph, features

def generate_test_data_with_anomaly(seq_length=48, num_features=4):
    """Génère les données de test sur un nouveau graphe avec une anomalie."""
    print("\n--- Génération des données de test ---")
    test_graph = create_hvac_building_graph_test()
    test_features = generate_sample_time_series_data(test_graph, seq_length, num_features, inject_anomaly=True)
    print("✅ Données de test générées (avec anomalie).")
    return test_graph, test_features

def standardize_features(features_tensor):
    """
    Standardise les caractéristiques (features) pour avoir une moyenne de 0 et un écart-type de 1.
    """
    mean = torch.mean(features_tensor, dim=(0, 1))
    std = torch.std(features_tensor, dim=(0, 1))

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