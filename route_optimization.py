import csv
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

def load_data(filename):
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            nodes = headers[1:]  # Column headers represent nodes

            graph_data = {}
            for row in reader:
                origin_city = row[0]
                for i, distance in enumerate(row[1:], start=1):
                    if distance.isdigit():
                        destination_city = headers[i]
                        graph_data[(origin_city, destination_city)] = int(distance)
        return nodes, graph_data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None

def create_graph(nodes, graph_data):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from((origin, dest, weight) for (origin, dest), weight in graph_data.items())
    return G

def find_optimized_path(G, start, end, unavailable_nodes=None):
    if unavailable_nodes is None:
        unavailable_nodes = []

    # Remove unavailable nodes from the graph
    temp_G = G.copy()
    temp_G.remove_nodes_from(unavailable_nodes)

    try:
        # Find the primary path
        primary_path = nx.dijkstra_path(temp_G, start, end, weight='weight')
        primary_time = nx.dijkstra_path_length(temp_G, start, end, weight='weight')
    except nx.NetworkXNoPath:
        return None, None, None, None

    # Check if unavailable nodes affect the primary path
    alternate_path, alternate_time = None, None
    for node in unavailable_nodes:
        if node in primary_path:
            temp_G.remove_node(node)
            try:
                alternate_path = nx.dijkstra_path(temp_G, start, end, weight='weight')
                alternate_time = nx.dijkstra_path_length(temp_G, start, end, weight='weight')
                break
            except nx.NetworkXNoPath:
                continue  # No alternate path exists for this configuration

    return primary_path, primary_time, alternate_path, alternate_time

def plot_graph(G, primary_path, alternate_path, dep_city, arr_city, streamlit=False):
    pos = nx.spring_layout(G)
    fig = plt.figure(figsize=(10, 10))
    # plt.figure(figsize=(10, 10))

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    # Highlight paths
    if primary_path:
        primary_edges = list(zip(primary_path, primary_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=primary_edges, edge_color='orange', width=3, label="Primary Path")
    if alternate_path:
        alternate_edges = list(zip(alternate_path, alternate_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=alternate_edges, edge_color='red', width=3, label="Alternate Path")

    plt.title(f"Flight Path from {dep_city} to {arr_city}", fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.show()
    
    if streamlit:
        st.pyplot(fig)


if __name__ == '__main__':
    filename = 'data/Cities_FlightDuration_Mins.csv'
    nodes, graph_data = load_data(filename)

    if nodes and graph_data:
        G = create_graph(nodes, graph_data)

        # Cities list
        dep_city = 'Kanpur'
        arr_city = 'Indore'
        unavailable_nodes = []

        primary_path, primary_time, alternate_path, alternate_time = find_optimized_path(G, dep_city, arr_city, unavailable_nodes)
        if primary_path:
            print(f"Primary path: {primary_path}, Flight duration: {primary_time} mins")
            if alternate_path:
                print(f"Alternate path: {alternate_path}, Flight duration: {alternate_time} mins")
            plot_graph(G, primary_path, alternate_path, dep_city, arr_city)
        else:
            print("No path exists between the specified nodes.")
