import networkx as nx
import numpy as np
import torch

def create_hvac_building_graph():
    """Crée un exemple de graphe NetworkX pour un système CVC simple."""
    G = nx.Graph()
    
    equipments = {
        'Chiller_01': {'type': 'Chiller'},
        'Boiler_01': {'type': 'Boiler'},
        'Pump_CW_01': {'type': 'Pump'},
        'Pump_HW_01': {'type': 'Pump'},
        'AHU_01': {'type': 'AHU'},
        'VAV_Zone_N': {'type': 'VAV'},
        'VAV_Zone_S': {'type': 'VAV'}
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
    """Génère des données de séries temporelles factices pour chaque nœud."""
    num_nodes = graph.number_of_nodes()
    time = np.linspace(0, 8 * np.pi, seq_length)
    base_signal = np.sin(time)
    
    node_features = []
    for i in range(num_nodes):
        noise = np.random.randn(seq_length, num_features) * 0.15
        variation = (np.random.rand(num_features) - 0.5) * 2
        node_signal = np.outer(base_signal, variation) + noise
        node_features.append(node_signal)
        
    # Optionnel : Injecter une anomalie pour la visualisation
    # Sur le dernier quart du temps, on ajoute une forte déviation à la pompe à eau chaude
    if 'Pump_HW_01' in graph.nodes():
        pump_hw_index = list(graph.nodes()).index('Pump_HW_01')
        start_anomaly = int(seq_length * 0.75)
        node_features[pump_hw_index][start_anomaly:, 1] += 1.5 # Grosse augmentation de pression
        
    return np.stack(node_features)

def prepare_data_for_model(graph, node_data):
    """Convertit les données du graphe en tenseurs PyTorch."""
    node_features_tensor = torch.tensor(node_data, dtype=torch.float32)
    
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edges = [[node_mapping[u], node_mapping[v]] for u, v in graph.edges()]
    edges.extend([[v, u] for u, v in edges])
    edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    inv_node_mapping = {i: node for node, i in node_mapping.items()}
    
    return node_features_tensor, edge_index_tensor, inv_node_mapping