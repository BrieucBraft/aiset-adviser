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
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    node_trace = go.Scatter(x=[], y=[], mode='markers+text', hoverinfo='text', textposition="top center")
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    color_map = {'Boiler': 'red', 'Pump': 'orange', 'AHU': 'green'}
    
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = data.get('type', 'Unknown')
        node_text.append(f"<b>{node}</b><br>Type: {node_type}")
        node_color.append(color_map.get(node_type, 'gray'))
        node_size.append(25 if node_type in ['Boiler', 'AHU'] else 15)
    
    node_trace.x = node_x
    node_trace.y = node_y
    node_trace.text = list(graph.nodes())
    node_trace.hovertext = node_text
    node_trace.marker = dict(showscale=False, color=node_color, size=node_size, line_width=2)
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=dict(text='🏢 Topologie du Bâtiment CVC', font=dict(size=16)),
        showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    fig.write_html(output_path)
    print(f"✅ Visualisation de la topologie enregistrée dans : {output_path}")

def visualize_anomaly_scores(anomaly_scores, inv_node_mapping, threshold):
    """Crée un bar chart pour visualiser les scores d'anomalie de chaque équipement."""
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/anomaly_scores.html"
    
    node_names = [inv_node_mapping[i] for i in range(len(anomaly_scores))]
    scores = anomaly_scores.detach().numpy()
    colors = ['red' if score > threshold else 'green' for score in scores]
    
    fig = go.Figure([go.Bar(x=node_names, y=scores, marker_color=colors)])
    fig.add_shape(type="line", x0=-0.5, y0=threshold, x1=len(node_names)-0.5, y1=threshold,
                  line=dict(color="orange", width=2, dash="dash"))
    
    fig.update_layout(
        title="📊 Scores d'Anomalie par Équipement (Sortie Classification)",
        xaxis_title="Équipement",
        yaxis_title="Score d'Anomalie Prédit (0=Normal, 1=Anormal)",
    )
    fig.write_html(output_path)
    print(f"✅ Scores d'anomalie enregistrés dans : {output_path}")

# LA FONCTION EST MAINTENANT RENOMMÉE CORRECTEMENT
def visualize_train_test_split(full_features, y_pred_unscaled, labels, train_idx, inv_node_mapping, feature_map):
    """
    Visualise les données, les prédictions superposées, les labels, et la séparation train/test.
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
        feature_index = 0
        
        # Données réelles complètes (partie train)
        fig.add_trace(go.Scatter(
            y=full_features[i, :train_idx, feature_index], 
            name='Données d\'entraînement', 
            mode='lines', 
            line=dict(color='blue')
        ), row=i+1, col=1)

        # Données réelles complètes (partie test)
        fig.add_trace(go.Scatter(
            x=list(range(train_idx, seq_length)), 
            y=full_features[i, train_idx:, feature_index], 
            name='Données de test (Réel)', 
            mode='lines', 
            line=dict(color='deepskyblue')
        ), row=i+1, col=1)
        
        # Prédictions sur TOUTE la période (avec décalage de 1)
        prediction_series = [None] + list(y_pred_unscaled[i, :, feature_index])
        fig.add_trace(go.Scatter(
            y=prediction_series, 
            name='Prédiction', 
            mode='lines', 
            line=dict(color='orange', dash='dash')
        ), row=i+1, col=1)

        # Ligne de séparation train/test
        fig.add_vline(x=train_idx, line_width=2, line_dash="dash", line_color="black", row=i+1, col=1)

        # Marqueurs pour les labels
        labeled_points = (labels[i, :] != -1).nonzero(as_tuple=True)[0]
        if labeled_points.numel() > 0:
            normal_points = labeled_points[labels[i, labeled_points] == 0]
            anomalous_points = labeled_points[labels[i, labeled_points] == 1]

            if normal_points.numel() > 0:
                fig.add_trace(go.Scatter(
                    x=normal_points, y=full_features[i, normal_points, feature_index], mode='markers',
                    marker=dict(color='green', symbol='circle-open', size=8, line=dict(width=2)),
                    name='Normal Labélisé'
                ), row=i+1, col=1)
            
            if anomalous_points.numel() > 0:
                fig.add_trace(go.Scatter(
                    x=anomalous_points, y=full_features[i, anomalous_points, feature_index], mode='markers',
                    marker=dict(color='red', symbol='x-thin', size=8, line=dict(width=3)),
                    name='Anomalie Labélisée'
                ), row=i+1, col=1)

    fig.update_layout(height=300*num_nodes, title_text="📈 Visualisation Train/Test et Labels (Moyennes Journalières)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.write_html(output_path)
    print(f"✅ Visualisation des données d'entraînement enregistrée dans : {output_path}")