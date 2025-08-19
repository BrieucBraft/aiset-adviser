# Fichier : utils/visualization.py

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def visualize_building_graph(graph: nx.Graph):
    """Crée une visualisation de la topologie et l'enregistre dans un fichier HTML."""
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/building_topology.html"

    pos = nx.spring_layout(graph, seed=42)

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    color_map = {'Chiller': 'blue', 'Boiler': 'red', 'Pump': 'orange', 'AHU': 'green', 'VAV': 'purple'}
    
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = data.get('type', 'Unknown')
        node_text.append(f"<b>{node}</b><br>Type: {node_type}")
        node_color.append(color_map.get(node_type, 'gray'))
        node_size.append(25 if node_type in ['Chiller', 'Boiler', 'AHU'] else 15)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text', textposition="top center",
        marker=dict(showscale=False, color=node_color, size=node_size, line_width=2)
    )
    node_trace.text = list(graph.nodes())
    node_trace.hovertext = node_text

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=dict(text='🏢 Topologie du Bâtiment CVC', font=dict(size=16)),
        showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    fig.write_html(output_path)
    print(f"✅ Visualisation de la topologie enregistrée dans : {output_path}")

def visualize_predictions_and_errors(y_true, y_pred, inv_node_mapping, feature_map):
    """Visualise les données, les prédictions et l'erreur pour CHAQUE feature et l'enregistre."""
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/predictions_analysis.html"
    
    num_nodes, seq_length, num_features = y_true.shape
    
    fig = make_subplots(
        rows=num_nodes, cols=num_features,
        shared_xaxes=True,
        subplot_titles=[f"{inv_node_mapping[i]} - {feature_map[j]}" for i in range(num_nodes) for j in range(num_features)]
    )
    
    for i in range(num_nodes): # Pour chaque équipement
        for j in range(num_features): # Pour chaque feature
            # Extraire les séries temporelles pour ce noeud et cette feature
            true_series = y_true[i, :, j]
            pred_series = y_pred[i, :, j]
            
            # Calculer l'erreur pour cette série spécifique
            error_series = (true_series - pred_series) ** 2
            
            # Ajouter les traces au bon sous-graphique
            fig.add_trace(go.Scatter(y=true_series, name='Réel', mode='lines', line=dict(color='blue'), showlegend=(i==0 and j==0)), row=i+1, col=j+1)
            fig.add_trace(go.Scatter(y=pred_series, name='Prédiction', mode='lines', line=dict(color='orange', dash='dash'), showlegend=(i==0 and j==0)), row=i+1, col=j+1)
            
            # Créer un axe Y secondaire pour l'erreur
            fig.add_trace(go.Scatter(y=error_series, name='Erreur (MSE)', mode='lines', line=dict(color='rgba(255, 0, 0, 0.6)'), yaxis=f"y{i*num_features+j+num_nodes*num_features+1}", showlegend=(i==0 and j==0)), row=i+1, col=j+1)
            
            # Configurer l'axe Y secondaire
            fig.update_layout({
                f'yaxis{i*num_features+j+num_nodes*num_features+1}': {
                    'overlaying': f'y{i*num_features+j+1}',
                    'side': 'right',
                    'showgrid': False,
                    'zeroline': False,
                    'showticklabels': False
                }
            })

    fig.update_layout(height=250*num_nodes, title_text="📈 Analyse Détaillée par Équipement et Caractéristique")
    
    # Mettre à jour les axes Y principaux
    for i in range(1, num_nodes * num_features + 1):
        fig.update_yaxes(row=(i-1)//num_features + 1, col=(i-1)%num_features + 1, showgrid=False, zeroline=False)

    fig.write_html(output_path)
    print(f"✅ Analyse détaillée des prédictions enregistrée dans : {output_path}")