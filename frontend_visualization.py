import networkx as nx
import plotly.graph_objects as go

def visualize_building(graph: nx.Graph):
    """
    Crée une visualisation interactive d'un graphe de bâtiment avec Plotly.
    
    Args:
        graph (nx.Graph): Le graphe NetworkX représentant le bâtiment.
    """
    # 1. Obtenir les positions des nœuds pour la visualisation
    pos = nx.spring_layout(graph, seed=42)

    # 2. Créer la trace pour les arêtes
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 3. Créer la trace pour les nœuds
    node_x, node_y = [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Personnaliser l'apparence des nœuds (couleur, taille, texte)
    node_text = []
    node_color = []
    node_size = []
    
    # Définir les couleurs par type d'équipement
    color_map = {
        'Chiller': 'blue',
        'Boiler': 'red',
        'Pump': 'orange',
        'AHU': 'green',
        'VAV': 'purple'
    }

    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        node_info = f"<b>{node}</b><br>Type: {node_type}"
        node_text.append(node_info)
        node_color.append(color_map.get(node_type, 'gray'))
        node_size.append(25 if node_type in ['Chiller', 'Boiler', 'AHU'] else 15)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2))
    
    node_trace.text = [node_id for node_id in graph.nodes()]
    node_trace.hovertext = node_text

    # 4. Créer la figure et l'afficher
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    # Le titre est maintenant un dictionnaire qui contient le texte et la police
                    title=dict(
                        text='🏢 Visualisation du Graphe du Bâtiment CVC',
                        font=dict(
                            size=16
                        )
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    print("🚀 Affichage de la visualisation interactive...")
    fig.write_html("my_interactive_plot.html")
    # fig.show()