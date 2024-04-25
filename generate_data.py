import numpy as np
from matplotlib import pyplot as plt


def get_category_color(label, num_categories):
    """

    args:
    - label: (0 -> n-1)
    - num_categories:
    """
    cmap = plt.get_cmap('viridis', num_categories)  # colormap: 'viridis', 'jet', 'hsv
    return cmap(label)


def get_labels_from_splitted_graph(positions, n=2, xy=2):
    # rectangle range: -xy, xy
    if n == 2:
        labels = split_graph_2(positions)
    else:
        print("split_n:", n)
        raise "Unsupported split n"

    return labels


def split_graph_2(positions, r=1.5):
    """
    circle: in and out
    """
    labels = np.array([1 if np.linalg.norm(p) < r else 0 for p in positions])
    return labels
