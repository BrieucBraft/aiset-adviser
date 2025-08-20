import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import plotly.colors

def visualize_building_graph(graph: nx.Graph, filename: str = "building_topology.html"):
    os.makedirs("reports", exist_ok=True)
    output_path = os.path.join("reports", filename)

    pos = nx.spring_layout(graph, seed=42)
    edge_x, edge_y = [], []
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
        title=dict(text=f'🏢 Topologie du Bâtiment CVC ({filename.split("_")[0].capitalize()})', font=dict(size=16)),
        showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    fig.write_html(output_path)
    print(f"✅ Visualisation de la topologie enregistrée dans : {output_path}")

def visualize_classification_test_data(graph, features, true_labels, pred_classes, inv_node_mapping, feature_map_dict, anomaly_types):
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/classification_test_results.html"
    
    num_nodes, seq_length, _ = features.shape
    node_types = nx.get_node_attributes(graph, 'type')
    inv_anomaly_map = {v: k for k, v in anomaly_types.items()}

    # Define a distinct color for each anomaly type
    color_palette = plotly.colors.qualitative.Plotly
    anomaly_colors = {
        0: 'rgba(0, 176, 80, 0.6)', # Green for NORMAL
        1: 'rgba(255, 0, 0, 0.6)',   # Red for PUMP_FAILURE
        2: 'rgba(255, 192, 0, 0.6)',# Orange for SENSOR_STUCK
        3: 'rgba(112, 48, 160, 0.6)'# Purple for BOILER_LOCKOUT
    }

    fig = make_subplots(
        rows=num_nodes, cols=2,
        shared_xaxes=True,
        column_widths=[0.75, 0.25],
        subplot_titles=[
            title for i in range(num_nodes)
            for title in (f"<b>{inv_node_mapping[i]}</b> ({node_types.get(inv_node_mapping[i], '')})", "Anomaly Classification")
        ],
        specs=[[{"secondary_y": True}, {"secondary_y": False}]] * num_nodes
    )

    for i in range(num_nodes):
        node_name = inv_node_mapping[i]
        node_type = node_types.get(node_name)
        feature_names = feature_map_dict.get(node_type, [])
        
        # --- Left Plot (Features) ---
        for j, feat_name in enumerate(feature_names):
            use_secondary_axis = any(kw in feat_name.lower() for kw in ['pressure', 'power', 'position'])
            fig.add_trace(go.Scatter(
                y=features[i, :, j], name=feat_name, mode='lines', showlegend=False
            ), row=i+1, col=1, secondary_y=use_secondary_axis)

        # --- Right Plot (Anomaly Classification State Chart) ---
        time_steps = list(range(seq_length))
        true_classes = true_labels[i].tolist()
        pred_classes_list = pred_classes[i].tolist()

        # Add a bar for each time step for the true and predicted class
        fig.add_trace(go.Bar(
            x=time_steps, y=[1] * seq_length,
            name='Vraie Classe',
            marker_color=[anomaly_colors.get(c, 'grey') for c in true_classes],
            text=[inv_anomaly_map.get(c, "N/A") for c in true_classes],
            hoverinfo='text',
            showlegend=(i==0)
        ), row=i+1, col=2)
        
        fig.add_trace(go.Bar(
            x=time_steps, y=[1] * seq_length,
            name='Classe Prédite',
            marker_color=[anomaly_colors.get(c, 'grey') for c in pred_classes_list],
            text=[inv_anomaly_map.get(c, "N/A") for c in pred_classes_list],
            hoverinfo='text',
            showlegend=(i==0)
        ), row=i+1, col=2)

        fig.update_layout(barmode='stack')
        
        # Configure the y-axis to be categorical for the state chart
        fig.update_yaxes(
            tickvals=[0.5, 1.5],
            ticktext=['Vraie', 'Prédite'],
            row=i+1, col=2,
            zeroline=False,
            showgrid=False
        )
        
        fig.update_yaxes(showgrid=False, row=i+1, col=1, secondary_y=False)
        fig.update_yaxes(showgrid=False, row=i+1, col=1, secondary_y=True)

    fig.update_layout(
        height=350*num_nodes, 
        title_text="<b>📉 Visualisation des Résultats de Classification d'Anomalies</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html(output_path)
    print(f"✅ Visualisation des résultats de test enregistrée dans : {output_path}")