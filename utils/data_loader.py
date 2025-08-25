import networkx as nx
import numpy as np
import torch
import os
import pickle
import glob
import sys
import random

FEATURE_MAP = {
    'VAV': ['Zone Temp (°C)', 'Damper Position (%)', 'Airflow (m³/s)', 'Temp Setpoint (°C)', 'Is Active'],
    'AHU': ['Supply Temp (°C)', 'Static Pressure (Pa)', 'Airflow (m³/s)', 'Fan Speed (%)', 'Power Draw (kW)'],
    'Pump': ['Supply Temp (°C)', 'Pressure Diff (kPa)', 'Flow Rate (m³/h)', 'Speed (%)', 'Power Draw (kW)'],
    'Chiller': ['Supply Temp (°C)', 'Return Temp (°C)', 'Water Flow (m³/h)', 'Load (%)', 'Power Draw (kW)'],
    'Boiler': ['Supply Temp (°C)', 'Return Temp (°C)', 'Water Flow (m³/h)', 'Load (%)', 'Power Draw (kW)']
}
NUM_FEATURES = 5

ANOMALY_TYPES = {
    'NORMAL': 0,
    'PUMP_FAILURE': 1,
    'SENSOR_STUCK_VAV': 2,
    'BOILER_LOCKOUT': 3,
    'CHILLER_FAILURE': 4,
    'DAMPER_STUCK_AHU': 5
}
NUM_CLASSES = len(ANOMALY_TYPES)
UNLABELED_ID = -1

ANOMALY_TARGET_MAP = {
    'PUMP_FAILURE': 'Pump',
    'SENSOR_STUCK_VAV': 'VAV',
    'BOILER_LOCKOUT': 'Boiler',
    'CHILLER_FAILURE': 'Chiller',
    'DAMPER_STUCK_AHU': 'AHU',
}

def load_and_generate_training_data(seq_length=96, anomaly_fraction=0.75, unlabeled_fraction=0.3):
    print("--- Chargement et génération des données d'entraînement ---")
    training_files = glob.glob("data/training/*.gpickle")
    
    if not training_files:
        print("❌ Erreur: Aucun fichier de graphe trouvé dans 'data/training/'.", file=sys.stderr)
        sys.exit(1)
        
    all_graphs, all_features, all_labels = [], [], []
    num_files_to_generate = len(training_files) * 10

    for i in range(num_files_to_generate):
        file_path = random.choice(training_files)
        with open(file_path, "rb") as f:
            graph = pickle.load(f)
            
            is_unlabeled = random.random() < unlabeled_fraction
            inject_anomaly = random.random() < anomaly_fraction
            anomaly_type = 'NORMAL'
            if inject_anomaly:
                anomaly_type = random.choice(list(ANOMALY_TYPES.keys())[1:])

            features, labels = generate_realistic_time_series_data(graph, seq_length, anomaly_type)
            
            if is_unlabeled:
                labels.fill(UNLABELED_ID)
                status = "non-étiqueté"
            else:
                status = f"étiqueté ({anomaly_type})"
            
            all_graphs.append(graph)
            all_features.append(torch.tensor(features, dtype=torch.float32))
            all_labels.append(torch.tensor(labels, dtype=torch.long))
            
            print(f"✅ Données générées pour {os.path.basename(file_path)} (batch {i+1}, {status})")
            
    return all_graphs, all_features, all_labels

def load_and_generate_test_data(seq_length=96):
    print("\n--- Chargement et génération des données de test ---")
    test_file = "data/testing/test_graph_anomaly.gpickle"

    if not os.path.exists(test_file):
        print(f"❌ Erreur: Le fichier de test '{test_file}' n'a pas été trouvé.", file=sys.stderr)
        sys.exit(1)

    with open(test_file, "rb") as f:
        graph = pickle.load(f)
        anomaly_type = random.choice(list(ANOMALY_TYPES.keys())[1:])
        features, labels = generate_realistic_time_series_data(graph, seq_length, anomaly_type)
        print(f"✅ Données de test générées pour {os.path.basename(test_file)} (Anomalie: {anomaly_type})")
        return graph, torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def generate_realistic_time_series_data(graph, seq_length=96, anomaly_type='NORMAL'):
    nodes = list(graph.nodes())
    node_map = {name: i for i, name in enumerate(nodes)}
    num_nodes = len(nodes)
    
    features = np.full((num_nodes, seq_length, NUM_FEATURES), np.nan)
    anomaly_labels = np.zeros((num_nodes, seq_length), dtype=int)

    setpoints = {'VAV_temp': 22.0, 'AHU_supply_temp_cool': 12.0, 'AHU_supply_temp_heat': 35.0, 
                 'Chiller_supply_temp': 7.0, 'Boiler_supply_temp': 60.0}

    time_of_day = np.linspace(0, 24, seq_length)
    occupancy = np.exp(-((time_of_day - 13.5)**2) / (2 * 2.5**2))
    occupancy[time_of_day < 7] = 0; occupancy[time_of_day > 18] = 0
    external_heat_load = occupancy * 8
    
    node_states = {node: {'temp': 18.0} for node in nodes}

    anomaly_params = {'node': None, 'start_time': -1, 'stuck_value': 21.0}
    if anomaly_type != 'NORMAL':
        anomaly_params['start_time'] = int(seq_length * (11/24.0))
        
        target_type = ANOMALY_TARGET_MAP.get(anomaly_type)

        target_nodes = []
        if target_type:
            target_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == target_type]

        if target_nodes: 
            anomaly_params['node'] = random.choice(target_nodes)

        if anomaly_params['node']:
            node_idx = node_map[anomaly_params['node']]
            anomaly_labels[node_idx, anomaly_params['start_time']:] = ANOMALY_TYPES[anomaly_type]
            print(f"🔧 Anomalie ({anomaly_type}) programmée sur '{anomaly_params['node']}'.")

    for t in range(seq_length):
        is_occupied = occupancy[t] > 0.1
        
        for node_name, data in graph.nodes(data=True):
            if data.get('type') == 'VAV':
                idx = node_map[node_name]
                current_temp = node_states[node_name]['temp']

                if anomaly_type == 'SENSOR_STUCK_VAV' and node_name == anomaly_params['node'] and t >= anomaly_params['start_time']:
                    current_temp = anomaly_params['stuck_value']

                temp_setpoint = setpoints['VAV_temp'] if is_occupied else 18.0
                temp_error = current_temp - temp_setpoint
                damper_pos = min(100, max(0, 50 + temp_error * 25)) if is_occupied else 0
                
                ahu_name = next(graph.neighbors(node_name))
                ahu_fan_speed = features[node_map[ahu_name], t-1, 3] if t > 0 else 0
                airflow = (damper_pos / 100.0) * (ahu_fan_speed / 100.0) * 0.5

                ahu_supply_temp = node_states[ahu_name].get('supply_temp', current_temp)
                thermal_effect = (node_states[node_name]['temp'] - ahu_supply_temp) * airflow * 0.2
                new_real_temp = node_states[node_name]['temp'] + (external_heat_load[t] * 0.15) - thermal_effect + random.uniform(-0.1, 0.1)
                node_states[node_name]['temp'] = new_real_temp

                features[idx, t] = [current_temp, damper_pos, airflow, temp_setpoint, 1 if is_occupied else 0]

        for node_name, data in graph.nodes(data=True):
            if data.get('type') == 'AHU':
                idx = node_map[node_name]
                connected_vavs = [n for n in graph.neighbors(node_name) if graph.nodes[n].get('type') == 'VAV']
                
                if not connected_vavs or not is_occupied:
                    node_states[node_name] = {'supply_temp': 20.0, 'mode': 'off'}
                    features[idx, t, :] = [20.0, 0, 0, 0, 0]
                    continue

                avg_vav_temp = np.mean([node_states[vav]['temp'] for vav in connected_vavs])
                
                system_mode = node_states[node_name].get('mode', 'off')
                if avg_vav_temp > setpoints['VAV_temp'] + 0.5: system_mode = 'cool'
                elif avg_vav_temp < setpoints['VAV_temp'] - 0.5: system_mode = 'heat'
                
                node_states[node_name]['mode'] = system_mode
                
                total_airflow_demand = sum(features[node_map[vav], t, 2] for vav in connected_vavs)
                fan_speed = min(100, max(20 if is_occupied else 0, total_airflow_demand * 250))
                
                if anomaly_type == 'DAMPER_STUCK_AHU' and node_name == anomaly_params['node'] and t >= anomaly_params['start_time']:
                    fan_speed = max(fan_speed, 90)
                    total_airflow_demand *= 0.3

                static_pressure = 250 * (fan_speed / 100.0)
                power_draw = 15 * (fan_speed / 100.0)**3

                supply_temp = 20.0
                if system_mode == 'cool':
                    pump_cw = next((n for n in graph.neighbors(node_name) if 'CW' in n and graph.nodes[n].get('type') == 'Pump'), None)
                    if pump_cw and t > 0 and features[node_map[pump_cw], t-1, 2] > 1:
                        supply_temp = setpoints['AHU_supply_temp_cool']
                elif system_mode == 'heat':
                    pump_hw = next((n for n in graph.neighbors(node_name) if 'HW' in n and graph.nodes[n].get('type') == 'Pump'), None)
                    if pump_hw and t > 0 and features[node_map[pump_hw], t-1, 2] > 1:
                        supply_temp = setpoints['AHU_supply_temp_heat']
                
                node_states[node_name]['supply_temp'] = supply_temp
                features[idx, t, :] = [supply_temp, static_pressure, total_airflow_demand, fan_speed, power_draw]

        for node_name, data in graph.nodes(data=True):
            if data.get('type') in ['Pump', 'Chiller', 'Boiler']:
                idx = node_map[node_name]
                
                is_pump = data.get('type') == 'Pump'
                plant_type = 'Chiller' if (is_pump and 'CW' in node_name) or data.get('type') == 'Chiller' else 'Boiler'
                required_mode = 'cool' if plant_type == 'Chiller' else 'heat'
                
                ahus_calling = [ahu for ahu, state in node_states.items() if graph.nodes[ahu].get('type') == 'AHU' and state.get('mode') == required_mode]
                is_active = len(ahus_calling) > 0 and is_occupied
                
                anomaly_active = node_name == anomaly_params['node'] and t >= anomaly_params['start_time']
                
                speed = 85 if is_active else 0
                load = 70 if is_active else 0
                
                if is_pump:
                    if anomaly_type == 'PUMP_FAILURE' and anomaly_active: speed = 0
                    pressure_diff = 150 * (speed / 100.0)
                    flow_rate = 10 * (speed / 100.0)
                    power = 5 * (speed / 100.0)**2
                    source = next((n for n in graph.neighbors(node_name) if graph.nodes[n].get('type') in ['Chiller', 'Boiler']), None)
                    temp = features[node_map[source], t-1, 0] if t > 0 and source else 20.0
                    features[idx, t, :] = [temp, pressure_diff, flow_rate, speed, power]
                else:
                    if anomaly_type == 'BOILER_LOCKOUT' and anomaly_active: load = 0
                    if anomaly_type == 'CHILLER_FAILURE' and anomaly_active: load *= 0.15
                        
                    power = 50 * (load / 100.0)
                    base_temp = setpoints['Chiller_supply_temp'] if data.get('type') == 'Chiller' else setpoints['Boiler_supply_temp']
                    return_temp = base_temp + (5 if data.get('type') == 'Chiller' else -10) * (load/100.0)
                    pump_node = next(graph.neighbors(node_name))
                    flow = features[node_map[pump_node], t, 2] if is_active and t > 0 else 0
                    features[idx, t, :] = [base_temp, return_temp, flow, load, power]

    return np.nan_to_num(features), anomaly_labels

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