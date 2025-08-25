import networkx as nx
import os
import pickle

def create_and_save_graphs():
    os.makedirs("data/training", exist_ok=True)
    os.makedirs("data/testing", exist_ok=True)

    # --- Training Graph 1: Comprehensive System ---
    G1 = nx.Graph()
    G1.add_nodes_from([
        ('Chiller_01', {'type': 'Chiller'}), ('Chiller_02', {'type': 'Chiller'}),
        ('Boiler_01', {'type': 'Boiler'}), ('Boiler_02', {'type': 'Boiler'}),
        ('Pump_CW_01', {'type': 'Pump'}), ('Pump_CW_02', {'type': 'Pump'}),
        ('Pump_HW_01', {'type': 'Pump'}), ('Pump_HW_02', {'type': 'Pump'}),
        ('AHU_01', {'type': 'AHU'}), ('AHU_02', {'type': 'AHU'}),
        ('VAV_Zone_N', {'type': 'VAV'}), ('VAV_Zone_S', {'type': 'VAV'}),
        ('VAV_Zone_E', {'type': 'VAV'}), ('VAV_Zone_W', {'type': 'VAV'}),
    ])
    G1.add_edges_from([
        ('Chiller_01', 'Pump_CW_01'), ('Pump_CW_01', 'AHU_01'),
        ('Chiller_02', 'Pump_CW_02'), ('Pump_CW_02', 'AHU_02'),
        ('Boiler_01', 'Pump_HW_01'), ('Pump_HW_01', 'AHU_01'),
        ('Boiler_02', 'Pump_HW_02'), ('Pump_HW_02', 'AHU_02'),
        ('AHU_01', 'VAV_Zone_N'), ('AHU_01', 'VAV_Zone_W'),
        ('AHU_02', 'VAV_Zone_S'), ('AHU_02', 'VAV_Zone_E'),
        ('AHU_01', 'AHU_02')
    ])
    with open("data/training/training_graph_1.gpickle", "wb") as f: pickle.dump(G1, f)
    print("✅ Saved training_graph_1.gpickle")

    # --- Training Graph 2: Simple Heating System ---
    G2 = nx.Graph()
    G2.add_nodes_from([
        ('Boiler_A', {'type': 'Boiler'}), ('Pump_A', {'type': 'Pump'}),
        ('AHU_A', {'type': 'AHU'}), ('VAV_Office', {'type': 'VAV'}),
    ])
    G2.add_edges_from([('Boiler_A', 'Pump_A'), ('Pump_A', 'AHU_A'), ('AHU_A', 'VAV_Office')])
    with open("data/training/training_graph_2.gpickle", "wb") as f: pickle.dump(G2, f)
    print("✅ Saved training_graph_2.gpickle")

    # --- Training Graph 3: Cooling System with Redundancy ---
    G3 = nx.Graph()
    G3.add_nodes_from([
        ('Chiller_X', {'type': 'Chiller'}), ('Pump_X1', {'type': 'Pump'}),
        ('Pump_X2', {'type': 'Pump'}), ('AHU_X', {'type': 'AHU'}),
        ('VAV_Lab1', {'type': 'VAV'}), ('VAV_Lab2', {'type': 'VAV'}),
    ])
    G3.add_edges_from([
        ('Chiller_X', 'Pump_X1'), ('Chiller_X', 'Pump_X2'),
        ('Pump_X1', 'AHU_X'), ('Pump_X2', 'AHU_X'),
        ('AHU_X', 'VAV_Lab1'), ('AHU_X', 'VAV_Lab2')
    ])
    with open("data/training/training_graph_3.gpickle", "wb") as f: pickle.dump(G3, f)
    print("✅ Saved training_graph_3.gpickle")

    # --- Training Graph 4: Large Multi-AHU Heating System ---
    G4 = nx.Graph()
    G4.add_nodes_from([
        ('Boiler_MAIN_1', {'type': 'Boiler'}), ('Boiler_MAIN_2', {'type': 'Boiler'}),
        ('Pump_HW_P1', {'type': 'Pump'}), ('Pump_HW_P2', {'type': 'Pump'}),
        ('AHU_EAST', {'type': 'AHU'}), ('AHU_WEST', {'type': 'AHU'}),
        ('VAV_Lobby', {'type': 'VAV'}), ('VAV_Conf_1', {'type': 'VAV'}), ('VAV_Conf_2', {'type': 'VAV'}),
    ])
    G4.add_edges_from([
        ('Boiler_MAIN_1', 'Pump_HW_P1'), ('Pump_HW_P1', 'AHU_EAST'),
        ('Boiler_MAIN_2', 'Pump_HW_P2'), ('Pump_HW_P2', 'AHU_WEST'),
        ('AHU_EAST', 'VAV_Lobby'), ('AHU_EAST', 'VAV_Conf_1'), ('AHU_WEST', 'VAV_Conf_2'),
    ])
    with open("data/training/training_graph_4.gpickle", "wb") as f: pickle.dump(G4, f)
    print("✅ Saved training_graph_4.gpickle")

    # --- Training Graph 5: Simple Cooling System ---
    G5 = nx.Graph()
    G5.add_nodes_from([
        ('Chiller_B', {'type': 'Chiller'}), ('Pump_B', {'type': 'Pump'}),
        ('AHU_B', {'type': 'AHU'}), ('VAV_ServerRoom', {'type': 'VAV'}),
    ])
    G5.add_edges_from([('Chiller_B', 'Pump_B'), ('Pump_B', 'AHU_B'), ('AHU_B', 'VAV_ServerRoom')])
    with open("data/training/training_graph_5.gpickle", "wb") as f: pickle.dump(G5, f)
    print("✅ Saved training_graph_5.gpickle")
    
    # --- Test Graph: Complex Multi-Loop System ---
    GT = nx.Graph()
    GT.add_nodes_from([
        # Primary Plant Loops
        ('Chiller_P1', {'type': 'Chiller'}), ('Chiller_P2', {'type': 'Chiller'}),
        ('Boiler_P1', {'type': 'Boiler'}), ('Boiler_P2', {'type': 'Boiler'}),
        ('Pump_CHW_P1', {'type': 'Pump'}), ('Pump_CHW_P2', {'type': 'Pump'}), # Primary Chilled Water
        ('Pump_HW_P1', {'type': 'Pump'}), ('Pump_HW_P2', {'type': 'Pump'}),   # Primary Hot Water
        
        # Secondary Distribution Loops
        ('Pump_CHW_S1', {'type': 'Pump'}), ('Pump_HW_S1', {'type': 'Pump'}), # Secondary Pumps
        
        # Air Side
        ('AHU_CORE', {'type': 'AHU'}), ('AHU_FLOOR_1', {'type': 'AHU'}), ('AHU_FLOOR_2', {'type': 'AHU'}),
        
        # Terminal Units (VAVs)
        ('VAV_F1_Office1', {'type': 'VAV'}), ('VAV_F1_Office2', {'type': 'VAV'}), ('VAV_F1_Conf', {'type': 'VAV'}),
        ('VAV_F2_Lab1', {'type': 'VAV'}), ('VAV_F2_Lab2', {'type': 'VAV'}), ('VAV_F2_Breakroom', {'type': 'VAV'}),
        ('VAV_CORE_Lobby', {'type': 'VAV'}), ('VAV_CORE_Atrium', {'type': 'VAV'}),
    ])
    
    GT.add_edges_from([
        # Primary Chilled Water Loop (connects chillers to secondary pump)
        ('Chiller_P1', 'Pump_CHW_P1'), ('Chiller_P2', 'Pump_CHW_P2'),
        ('Pump_CHW_P1', 'Pump_CHW_S1'), ('Pump_CHW_P2', 'Pump_CHW_S1'),

        # Primary Hot Water Loop (connects boilers to secondary pump)
        ('Boiler_P1', 'Pump_HW_P1'), ('Boiler_P2', 'Pump_HW_P2'),
        ('Pump_HW_P1', 'Pump_HW_S1'), ('Pump_HW_P2', 'Pump_HW_S1'),
        
        # Secondary Distribution to AHUs
        ('Pump_CHW_S1', 'AHU_CORE'), ('Pump_CHW_S1', 'AHU_FLOOR_1'), ('Pump_CHW_S1', 'AHU_FLOOR_2'),
        ('Pump_HW_S1', 'AHU_CORE'), ('Pump_HW_S1', 'AHU_FLOOR_1'), ('Pump_HW_S1', 'AHU_FLOOR_2'),
        
        # Air Distribution from AHUs to VAVs
        ('AHU_FLOOR_1', 'VAV_F1_Office1'), ('AHU_FLOOR_1', 'VAV_F1_Office2'), ('AHU_FLOOR_1', 'VAV_F1_Conf'),
        ('AHU_FLOOR_2', 'VAV_F2_Lab1'), ('AHU_FLOOR_2', 'VAV_F2_Lab2'), ('AHU_FLOOR_2', 'VAV_F2_Breakroom'),
        ('AHU_CORE', 'VAV_CORE_Lobby'), ('AHU_CORE', 'VAV_CORE_Atrium'),
    ])
    
    with open("data/testing/test_graph_anomaly.gpickle", "wb") as f: pickle.dump(GT, f)
    print("✅ Saved complex test_graph_anomaly.gpickle")

if __name__ == '__main__':
    create_and_save_graphs()