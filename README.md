# Winding Number on Graphs

### Author
- Boyang Liu
- Junting Lyu

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

### How to run

##### Prepare environments
```
pip install -r requirements.txt
```

##### Generate graph
```
python graph.py
```
```
usage: graph.py [-h] [--hard] [-c CATEGORIES] [-n POINTS] [-k KNN_K] [-t TRAIN_RATIO]

options:
  -h, --help            show this help message and exit
  --hard
  -c CATEGORIES, --categories CATEGORIES
                        Number of categories for the graph splitter
  -n POINTS, --points POINTS
                        Number of points to generate
  -k KNN_K, --knn_k KNN_K
                        K value for KNN
  -t TRAIN_RATIO, -r TRAIN_RATIO, --train_ratio TRAIN_RATIO
                        Training ratio
```

##### Pipeline
```
python pipeline.py
```
```
usage: pipeline.py [-h] [-c CATEGORIES] [-n POINTS] [-k KNN_K] [-t TRAIN_RATIO] [-i ITER_KMEANS] [--sample_n SAMPLE_N]
                   [--hard] [-fd FEATURE_DIMENSION] [--text] [--save_img]

options:
  -h, --help            show this help message and exit
  -c CATEGORIES, --categories CATEGORIES
                        Number of categories for the graph splitter
  -n POINTS, --points POINTS
                        Number of points to generate
  -k KNN_K, --knn_k KNN_K
                        K value for KNN
  -t TRAIN_RATIO, -r TRAIN_RATIO, --train_ratio TRAIN_RATIO
                        Training ratio
  -i ITER_KMEANS, --iter_kmeans ITER_KMEANS
                        Kmeans Iterations
  --sample_n SAMPLE_N   Number of sampled stroke directions
  --hard
  -fd FEATURE_DIMENSION, --feature_dimension FEATURE_DIMENSION
                        Feature dimension
  --text
  --save_img
```
