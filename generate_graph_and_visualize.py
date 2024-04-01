from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

from load_data import load_data


def generate_graph(features, labels):
    """
    Build a k-NN graph from the given features.
    Returns the graph G.
    """
    # Use k-NN algorithm to find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Create a graph
    G = nx.Graph()

    # Add nodes and edges
    for i in range(len(features)):
        G.add_node(i, label=labels[i])

    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i < neighbor:  # Avoid adding duplicates
                G.add_edge(i, neighbor)

    return G


def visualize(G):
    """
    Visualize the graph G.
    """
    # Prepare for drawing
    pos = nx.random_layout(G)  # Random layout
    color_map = [G.nodes[i]['label'] for i in G.nodes]  # Set colors based on labels

    # Draw the graph
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_color=color_map, node_size=20, cmap=cm.get_cmap('tab10'), with_labels=False)
    plt.title('MNIST k-NN Graph Visualization')
    plt.show()


def main():
    features, labels = load_data(limit=100)
    G = generate_graph(features, labels)
    visualize(G)


if __name__ == '__main__':
    main()
