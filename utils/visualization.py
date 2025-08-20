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

def visualize_supervised_test_data(features, true_labels, pred_probs, inv_node_mapping, feature_map, threshold):
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/supervised_test_results.html"
    
    num_nodes, seq_length, num_features = features.shape
    
    fig = make_subplots(
        rows=num_nodes, cols=2,
        shared_xaxes=True,
        column_widths=[0.7, 0.3],
        subplot_titles=[
            title for i in range(num_nodes) 
            for title in (f"{inv_node_mapping[i]} - Features", f"{inv_node_mapping[i]} - Anomaly Score")
        ]
    )

    for i in range(num_nodes):
        for j in range(num_features):
            fig.add_trace(go.Scatter(
                y=features[i, :, j], 
                name=feature_map[j],
                mode='lines',
                legendgroup=f"node_{i}_features",
                showlegend=(i==0)
            ), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            y=true_labels[i].squeeze(), 
            name='Vraie Anomalie', 
            mode='lines', 
            line=dict(color='rgba(255, 0, 0, 0.6)', width=4),
            legendgroup='anomaly',
            showlegend=(i==0)
        ), row=i+1, col=2)
        
        fig.add_trace(go.Scatter(
            y=pred_probs[i].squeeze(), 
            name='Prédiction (Prob.)', 
            mode='lines', 
            line=dict(color='rgba(0, 100, 255, 0.7)', dash='dash'),
            legendgroup='prediction',
            showlegend=(i==0)
        ), row=i+1, col=2)

        fig.add_hline(y=threshold, line_dash="dot", line_color="grey", row=i+1, col=2)

    fig.update_layout(height=250*num_nodes, title_text="📉 Visualisation des Résultats de Test Supervisés")
    fig.write_html(output_path)
    print(f"✅ Visualisation des résultats de test enregistrée dans : {output_path}")