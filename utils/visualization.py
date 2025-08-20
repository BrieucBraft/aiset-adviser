import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

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

def visualize_supervised_test_data(graph, features, true_labels, pred_probs, inv_node_mapping, feature_map_dict, threshold):
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/supervised_test_results.html"
    
    num_nodes, _, _ = features.shape
    node_types = nx.get_node_attributes(graph, 'type')
    
    fig = make_subplots(
        rows=num_nodes, cols=2,
        shared_xaxes=True,
        column_widths=[0.75, 0.25],
        subplot_titles=[
            title for i in range(num_nodes)
            for title in (f"<b>{inv_node_mapping[i]}</b> ({node_types.get(inv_node_mapping[i], '')})", "Anomaly Score")
        ],
        specs=[[{"secondary_y": True}, {"secondary_y": False}]] * num_nodes
    )

    for i in range(num_nodes):
        node_name = inv_node_mapping[i]
        node_type = node_types.get(node_name)
        feature_names = feature_map_dict.get(node_type, [])
        
        # --- Left Plot (Features) ---
        for j, feat_name in enumerate(feature_names):
            # Assign features with large values to the secondary y-axis
            use_secondary_axis = any(kw in feat_name.lower() for kw in ['pressure', 'power', 'position'])
            
            fig.add_trace(go.Scatter(
                y=features[i, :, j],
                name=feat_name,
                mode='lines',
            ), row=i+1, col=1, secondary_y=use_secondary_axis)

        # --- Right Plot (Anomaly Scores) ---
        fig.add_trace(go.Scatter(
            y=true_labels[i].squeeze(),
            name='Vraie Anomalie',
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=6),
            showlegend=(i==0)
        ), row=i+1, col=2)

        fig.add_trace(go.Scatter(
            y=pred_probs[i].squeeze(),
            name='Prédiction (Prob.)',
            mode='lines',
            line=dict(color='rgba(0, 0, 255, 0.8)', dash='dash'),
            showlegend=(i==0)
        ), row=i+1, col=2)
        
        fig.add_hline(y=threshold, line_dash="dot", line_color="grey", row=i+1, col=2)
        fig.update_yaxes(range=[-0.1, 1.1], row=i+1, col=2, title_text="Probabilité")
        
        # Configure axes
        fig.update_yaxes(showgrid=False, row=i+1, col=1, secondary_y=False)
        fig.update_yaxes(showgrid=False, row=i+1, col=1, secondary_y=True)

    fig.update_layout(
        height=350*num_nodes, 
        title_text="<b>📉 Visualisation des Résultats de Test Supervisés</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html(output_path)
    print(f"✅ Visualisation des résultats de test enregistrée dans : {output_path}")