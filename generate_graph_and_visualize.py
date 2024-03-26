import open_clip
import torch
from torchvision import datasets
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from tqdm import tqdm

# Load the MNIST dataset
trainset = datasets.MNIST('data', download=True, train=True)

# Load the OpenCLIP model and preprocessing tools
model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')

# Initialize lists to store features and labels
features = []
labels = []

# Process a small subset of the dataset, for example, the first 1000 images
for i, (image, label) in enumerate(tqdm(trainset)):
    if i >= 100:  # Limit the number of processed images
        break
    preprocessed_image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        image_feature = model.encode_image(preprocessed_image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

    features.append(image_feature.squeeze().numpy())  # Remove batch dimension and convert to numpy
    labels.append(label)

# Convert to numpy arrays
features = np.array(features)

# Use k-NN algorithm to find nearest neighbors
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(features)
distances, indices = nbrs.kneighbors(features)

# Create graph
G = nx.Graph()

# Add nodes and edges
for i in range(len(features)):
    G.add_node(i, label=labels[i])

for i, neighbors in enumerate(indices):
    for neighbor in neighbors:
        if i < neighbor:  # Avoid adding duplicates
            G.add_edge(i, neighbor)

# Prepare for plotting
pos = nx.random_layout(G)  # Random layout
color_map = [G.nodes[i]['label'] for i in G.nodes]  # Set colors based on labels

# Plot
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_color=color_map, node_size=20, cmap=cm.get_cmap('tab10'), with_labels=False)
plt.title('MNIST k-NN Graph Visualization')
plt.show()
