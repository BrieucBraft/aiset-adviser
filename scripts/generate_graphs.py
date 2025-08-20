# Fichier: scripts/generate_graphs.py

import networkx as nx
import os
import pickle

def create_and_save_graphs():
    """
    Defines and saves multiple graph topologies for training and testing.
    """
    # Create directories if they don't exist
    os.makedirs("data/training", exist_ok=True)
    os.makedirs("data/testing", exist_ok=True)

    # --- Training Graph 1: Comprehensive System (Heating & Cooling) ---
    G1 = nx.Graph()
    G1_equipments = {
        'Chiller_01': {'type': 'Chiller'}, 'Chiller_02': {'type': 'Chiller'},
        'Boiler_01': {'type': 'Boiler'}, 'Boiler_02': {'type': 'Boiler'},
        'Pump_CW_01': {'type': 'Pump'}, 'Pump_CW_02': {'type': 'Pump'},
        'Pump_HW_01': {'type': 'Pump'}, 'Pump_HW_02': {'type': 'Pump'},
        'AHU_01': {'type': 'AHU'}, 'AHU_02': {'type': 'AHU'},
        'VAV_Zone_N': {'type': 'VAV'}, 'VAV_Zone_S': {'type': 'VAV'},
        'VAV_Zone_E': {'type': 'VAV'}, 'VAV_Zone_W': {'type': 'VAV'},
    }
    G1.add_nodes_from(G1_equipments.items())
    G1.add_edges_from([
        ('Chiller_01', 'Pump_CW_01'), ('Pump_CW_01', 'AHU_01'),
        ('Chiller_02', 'Pump_CW_02'), ('Pump_CW_02', 'AHU_02'),
        ('Boiler_01', 'Pump_HW_01'), ('Pump_HW_01', 'AHU_01'),
        ('Boiler_02', 'Pump_HW_02'), ('Pump_HW_02', 'AHU_02'),
        ('AHU_01', 'VAV_Zone_N'), ('AHU_01', 'VAV_Zone_W'),
        ('AHU_02', 'VAV_Zone_S'), ('AHU_02', 'VAV_Zone_E'),
        ('AHU_01', 'AHU_02')
    ])
    with open("data/training/training_graph_1.gpickle", "wb") as f:
        pickle.dump(G1, f)
    print("✅ Saved training_graph_1.gpickle")

    # --- Training Graph 2: Simple Heating System ---
    G2 = nx.Graph()
    G2_equipments = {
        'Boiler_A': {'type': 'Boiler'},
        'Pump_A': {'type': 'Pump'},
        'AHU_A': {'type': 'AHU'},
        'VAV_Office': {'type': 'VAV'},
    }
    G2.add_nodes_from(G2_equipments.items())
    G2.add_edges_from([
        ('Boiler_A', 'Pump_A'),
        ('Pump_A', 'AHU_A'),
        ('AHU_A', 'VAV_Office'),
    ])
    with open("data/training/training_graph_2.gpickle", "wb") as f:
        pickle.dump(G2, f)
    print("✅ Saved training_graph_2.gpickle")

    # --- Training Graph 3: Cooling System with Redundancy ---
    G3 = nx.Graph()
    G3_equipments = {
        'Chiller_X': {'type': 'Chiller'},
        'Pump_X1': {'type': 'Pump'},
        'Pump_X2': {'type': 'Pump'},
        'AHU_X': {'type': 'AHU'},
        'VAV_Lab1': {'type': 'VAV'},
        'VAV_Lab2': {'type': 'VAV'},
    }
    G3.add_nodes_from(G3_equipments.items())
    G3.add_edges_from([
        ('Chiller_X', 'Pump_X1'), ('Chiller_X', 'Pump_X2'),
        ('Pump_X1', 'AHU_X'), ('Pump_X2', 'AHU_X'),
        ('AHU_X', 'VAV_Lab1'), ('AHU_X', 'VAV_Lab2')
    ])
    with open("data/training/training_graph_3.gpickle", "wb") as f:
        pickle.dump(G3, f)
    print("✅ Saved training_graph_3.gpickle")


    # --- Test Graph: Complex Heating System ---
    GT = nx.Graph()
    GT_equipments = {
        'Boiler_T1': {'type': 'Boiler'}, 'Boiler_T2': {'type': 'Boiler'},
        'Pump_HW_T1': {'type': 'Pump'}, 'Pump_HW_T2': {'type': 'Pump'},
        'AHU_T1': {'type': 'AHU'}, 'AHU_T2': {'type': 'AHU'},
        'VAV_Conference_A': {'type': 'VAV'}, 'VAV_Conference_B': {'type': 'VAV'},
        'VAV_Lobby': {'type': 'VAV'},
    }
    GT.add_nodes_from(GT_equipments.items())
    GT.add_edges_from([
        ('Boiler_T1', 'Pump_HW_T1'), ('Pump_HW_T1', 'AHU_T1'),
        ('Boiler_T2', 'Pump_HW_T2'), ('Pump_HW_T2', 'AHU_T2'),
        ('AHU_T1', 'VAV_Conference_A'), ('AHU_T1', 'VAV_Lobby'),
        ('AHU_T2', 'VAV_Conference_B'),
    ])
    with open("data/testing/test_graph_anomaly.gpickle", "wb") as f:
        pickle.dump(GT, f)
    print("✅ Saved test_graph_anomaly.gpickle")

if __name__ == '__main__':
    create_and_save_graphs()