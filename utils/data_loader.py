import networkx as nx
import numpy as np
import torch
import os
import pickle
import glob
import sys
import random

def load_and_generate_training_data(seq_length=48, num_features=4, anomaly_fraction=0.5):
    print("--- Chargement et génération des données d'entraînement ---")
    training_files = glob.glob("data/training/*.gpickle")
    
    if not training_files:
        print("❌ Erreur: Aucun fichier de graphe trouvé dans 'data/training/'.", file=sys.stderr)
        print("   Veuillez d'abord exécuter 'python -m scripts.generate_graphs'.", file=sys.stderr)
        sys.exit(1)
        
    all_graphs, all_features, all_labels = [], [], []

    for file_path in training_files:
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
            inject_anomaly = random.random() < anomaly_fraction
            features, labels = generate_sample_time_series_data(graph, seq_length, num_features, inject_anomaly=inject_anomaly)
            
            all_graphs.append(graph)
            all_features.append(torch.tensor(features, dtype=torch.float32))
            all_labels.append(torch.tensor(labels, dtype=torch.float32))
            
            status = "avec anomalie" if inject_anomaly else "sans anomalie"
            print(f"✅ Données générées pour {os.path.basename(file_path)} ({status})")
            
    return all_graphs, all_features, all_labels

def load_and_generate_test_data(seq_length=48, num_features=4):
    print("\n--- Chargement et génération des données de test ---")
    test_file = "data/testing/test_graph_anomaly.gpickle"

    if not os.path.exists(test_file):
        print(f"❌ Erreur: Le fichier de test '{test_file}' n'a pas été trouvé.", file=sys.stderr)
        print("   Veuillez d'abord exécuter 'python -m scripts.generate_graphs'.", file=sys.stderr)
        sys.exit(1)

    with open(test_file, "rb") as f:
        graph = pickle.load(f)
        features, labels = generate_sample_time_series_data(graph, seq_length, num_features, inject_anomaly=True)
        print(f"✅ Données de test générées pour {os.path.basename(test_file)}")
        return graph, torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def generate_sample_time_series_data(graph, seq_length=48, num_features=4, inject_anomaly=False):
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    node_map = {name: i for i, name in enumerate(nodes)}
    
    features = np.zeros((num_nodes, seq_length, num_features))
    anomaly_labels = np.zeros((num_nodes, seq_length, 1))

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
            stable_pressure = (r['pressure'][0] + r['pressure'][1]) / 2
            pressure_noise = np.random.randn(seq_length) * 0.01
            pressure_signal = features[i, :, 3] * stable_pressure + pressure_noise
        else:
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
        possible_nodes = [node for node, data in graph.nodes(data=True) if data['type'] in ['Pump', 'Boiler', 'Chiller']]
        if not possible_nodes:
            possible_nodes = list(graph.nodes())

        node_to_affect = random.choice(possible_nodes)
        node_index = node_map.get(node_to_affect)
        
        start_anomaly = int(seq_length * random.uniform(0.4, 0.6))
        end_anomaly = int(seq_length * random.uniform(0.8, 1.0))
        
        features[node_index, start_anomaly:end_anomaly, 1] *= random.uniform(0.05, 0.2)
        features[node_index, start_anomaly:end_anomaly, 2] *= random.uniform(0.05, 0.2)
        anomaly_labels[node_index, start_anomaly:end_anomaly, 0] = 1.0
        
        print(f"🔧 Anomalie injectée sur '{node_to_affect}'.")

    return features, anomaly_labels

def standardize_features(feature_list):
    all_features_tensor = torch.cat(feature_list, dim=0)
    mean = torch.mean(all_features_tensor, dim=(0, 1))
    std = torch.std(all_features_tensor, dim=(0, 1))
    std[std == 0] = 1
    scaler = {'mean': mean, 'std': std}
    
    standardized_list = [(features - mean) / std for features in feature_list]
    
    print("✅ Données standardisées (moyenne=0, écart-type=1).")
    return standardized_list, scaler

def prepare_data_for_model(graph, features, labels):
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edges = [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()]
    edges.extend([[v, u] for u, v in edges])
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    inv_node_mapping = {i: node for node, i in node_mapping.items()}
    return features, labels, edge_index_tensor, inv_node_mapping