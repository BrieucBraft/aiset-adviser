import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def visualize_building_graph(graph: nx.Graph):
    """Crée une visualisation de la topologie et l'enregistre dans un fichier HTML."""
    # S'assurer que le dossier des rapports existe
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/building_topology.html"

    pos = nx.spring_layout(graph, seed=42)
    edge_trace = go.Scatter(
        x=[pos[edge[0]][0] for edge in graph.edges()] + [pos[edge[1]][0] for edge in graph.edges()],
        y=[pos[edge[0]][1] for edge in graph.edges()] + [pos[edge[1]][1] for edge in graph.edges()],
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
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

    # Enregistrer le fichier au lieu de l'afficher
    fig.write_html(output_path)
    print(f"✅ Visualisation de la topologie enregistrée dans : {output_path}")

def visualize_predictions_and_errors(y_true, y_pred, inv_node_mapping, feature_map):
    """Visualise les données, les prédictions, l'erreur, et enregistre dans un fichier HTML."""
    # S'assurer que le dossier des rapports existe
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/predictions_analysis.html"
    
    num_nodes, seq_length, num_features = y_true.shape
    
    fig = make_subplots(
        rows=num_nodes, cols=1,
        shared_xaxes=True,
        subplot_titles=[inv_node_mapping[i] for i in range(num_nodes)]
    )
    
    error_per_node = (y_true - y_pred) ** 2
    
    for i in range(num_nodes):
        # Afficher la première feature pour la clarté
        feature_index = 0
        fig.add_trace(go.Scatter(y=y_true[i, :, feature_index], name=f'Réel ({feature_map[feature_index]})', mode='lines', line=dict(color='blue'), legendgroup=f'group{i}'), row=i+1, col=1)
        fig.add_trace(go.Scatter(y=y_pred[i, :, feature_index], name='Prédiction', mode='lines', line=dict(color='orange', dash='dash'), legendgroup=f'group{i}'), row=i+1, col=1)
        
        # Afficher l'erreur de reconstruction totale pour ce nœud sur un axe Y secondaire
        node_error = error_per_node[i].mean(dim=1)
        fig.add_trace(go.Scatter(y=node_error, name='Erreur (MSE)', mode='lines', line=dict(color='red', width=3), legendgroup=f'group{i}'), row=i+1, col=1)

    fig.update_layout(height=250*num_nodes, title_text="📈 Comparaison Prédictions vs Réalité et Score d'Erreur", showlegend=False)

    # Enregistrer le fichier au lieu de l'afficher
    fig.write_html(output_path)
    print(f"✅ Analyse des prédictions enregistrée dans : {output_path}")