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

def visualize_training_data(full_features, y_pred_unscaled, inv_node_mapping, feature_map):
    """
    Visualise la séquence d'entraînement complète (entrée + cible) et superpose les prédictions.
    """
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/training_predictions.html"
    
    num_nodes, seq_length, num_features = full_features.shape
    
    fig = make_subplots(
        rows=num_nodes, cols=num_features,
        shared_xaxes=True,
        subplot_titles=[f"{inv_node_mapping[i]} - {feature_map[j]}" for i in range(num_nodes) for j in range(num_features)]
    )
    
    for i in range(num_nodes):
        for j in range(num_features):
            # Afficher la séquence complète des données réelles
            fig.add_trace(go.Scatter(y=full_features[i, :, j], name='Données réelles', mode='lines', line=dict(color='blue'), showlegend=False), row=i+1, col=j+1)
            
            # Superposer les prédictions avec un décalage de 1 (car on prédit T+1)
            # On ajoute un 'None' au début pour que la série commence au bon endroit
            prediction_series = [None] + list(y_pred_unscaled[i, :, j])
            fig.add_trace(go.Scatter(y=prediction_series, name='Prédiction', mode='lines', line=dict(color='orange', dash='dash'), showlegend=False), row=i+1, col=j+1)

    fig.update_layout(height=250*num_nodes, title_text="📈 Visualisation des Données d'Entraînement et Prédictions")
    fig.write_html(output_path)
    print(f"✅ Visualisation des données d'entraînement enregistrée dans : {output_path}")


def visualize_anomaly_scores(error_per_node, inv_node_mapping, threshold):
    """
    Crée un bar chart pour visualiser les scores d'anomalie de chaque équipement.
    """
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/anomaly_scores.html"
    
    node_names = [inv_node_mapping[i] for i in range(len(error_per_node))]
    scores = error_per_node.detach().numpy()
    colors = ['red' if score > threshold else 'green' for score in scores]
    
    fig = go.Figure([go.Bar(x=node_names, y=scores, marker_color=colors)])
    
    # Ajouter la ligne de seuil
    fig.add_shape(type="line", x0=-0.5, y0=threshold, x1=len(node_names)-0.5, y1=threshold,
                  line=dict(color="orange", width=2, dash="dash"))
    
    fig.update_layout(
        title="📊 Scores d'Anomalie par Équipement",
        xaxis_title="Équipement",
        yaxis_title="Erreur de Reconstruction (MSE Pondérée)",
    )
    fig.write_html(output_path)
    print(f"✅ Scores d'anomalie enregistrés dans : {output_path}")

def visualize_training_data(full_features, y_pred_unscaled, labels, train_idx, inv_node_mapping, feature_map):
    """
    Visualise les données, les prédictions, les labels, et la séparation train/test.
    """
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/training_predictions.html"
    
    num_nodes, seq_length, num_features = full_features.shape
    
    fig = make_subplots(
        rows=num_nodes, cols=1,
        shared_xaxes=True,
        subplot_titles=[inv_node_mapping[i] for i in range(num_nodes)]
    )
    
    for i in range(num_nodes):
        # Afficher la première feature pour la clarté visuelle
        feature_index = 0
        
        # Données réelles
        fig.add_trace(go.Scatter(y=full_features[i, :, feature_index], name='Données réelles', mode='lines', line=dict(color='blue')), row=i+1, col=1)
        
        # Prédictions (avec décalage)
        prediction_series = [None] * (train_idx + 1) + list(y_pred_unscaled[i, :, feature_index])
        fig.add_trace(go.Scatter(y=prediction_series, name='Prédiction (Test)', mode='lines', line=dict(color='orange', dash='dash')), row=i+1, col=1)

        # Ajouter la ligne de séparation train/test
        fig.add_vline(x=train_idx, line_width=2, line_dash="dash", line_color="grey", row=i+1, col=1)

        # Ajouter les marqueurs pour les labels
        labeled_points = (labels[i, :] != -1).nonzero().squeeze()
        for point_idx in labeled_points:
            is_anomaly = labels[i, point_idx] == 1
            fig.add_trace(go.Scatter(
                x=[point_idx], y=[full_features[i, point_idx, feature_index]],
                mode='markers',
                marker=dict(
                    color='red' if is_anomaly else 'green',
                    symbol='x' if is_anomaly else 'circle',
                    size=10,
                    line=dict(width=2, color='black')
                ),
                name='Anomalie' if is_anomaly else 'Normal'
            ), row=i+1, col=1)

    fig.update_layout(height=250*num_nodes, title_text="📈 Visualisation Train/Test et Labels", showlegend=False)
    fig.write_html(output_path)
    print(f"✅ Visualisation des données d'entraînement enregistrée dans : {output_path}")