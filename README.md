# Winding Number on Graphs

### Files
- load_data.py
  - Load MNIST dataset
  - Embed images with OpenClip
  - Generate points with features and labels
- generate_data.py
  - Generate points of simple graphs
    - Split area into different parts based on some rules 
    - Define draw_dividers based on the same rules
- graph_visualizer.py
- graph.py
  - Build graph
- pipeline.py
  - Sample stroke directions
  - Calculate winding numbers
  - Calculate total variance
  - Calculate features
  - Semi-supervised Kmeans to predict labels
