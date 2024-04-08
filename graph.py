from __future__ import annotations

import os

import numpy as np
from sklearn.neighbors import NearestNeighbors

from load_data import load_data


class Point:
    def __init__(self, ID, feature, label):
        self.ID = ID
        self.feature = feature
        self.label = label

    @staticmethod
    def get_distance(v1: Point, v2: Point):
        return np.linalg.norm(v1.feature - v2.feature)

    @staticmethod
    def gaussian_similarity(x1, x2, sigma):
        """
        Compute the Gaussian similarity between two points.
        """
        # Calculate the Euclidean distance squared
        distance_squared = np.sum(Point.get_distance(x1, x2) ** 2)

        # Compute the Gaussian similarity
        similarity = np.exp(-distance_squared / (2 * sigma ** 2))

        return similarity


class Edge:
    def __init__(self, v1: Point, v2: Point):
        self.v1 = v1
        self.v2 = v2

    def is_stroke(self):
        return self.v1.label != self.v2.label


class Graph:
    def __init__(self):
        # ID: Point
        self.vertices = {}
        # (i, j) -> edge
        self.edges = {}

        self.num_points = 100
        self.knn_k = 5

    def build_graph(self, regenerate=False):

        dir_graph = f"graphs/{self.num_points}/"
        if os.path.isdir(dir_graph) and not regenerate:
            features = np.load(f'{dir_graph}/features.npy')
            labels = np.load(f'{dir_graph}/labels.npy')
        else:
            features, labels = load_data(limit=self.num_points)
            os.makedirs(dir_graph, exist_ok=True)
            np.save(f'{dir_graph}/features.npy', features)
            np.save(f'{dir_graph}/labels.npy', labels)

        for ID, (feature, label) in enumerate(zip(features, labels)):
            self.vertices[ID] = Point(ID, feature, label)
        self._generate_edges_by_knn(features)

    def _generate_edges_by_knn(self, features):
        n_neighbors = NearestNeighbors(n_neighbors=self.knn_k, algorithm='auto').fit(features)
        distances, indices = n_neighbors.kneighbors(features)

        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                if i < neighbor:  # Avoid adding duplicates
                    self.edges[(i, neighbor)] = Edge(self.vertices[i], self.vertices[neighbor])

    def get_edge(self, i, j):
        # assert i < j
        return self.edges[tuple(sorted([i, j]))]

    def get_vertex(self, i):
        return self.vertices[i]


if __name__ == '__main__':
    g = Graph()

    g.build_graph(regenerate=False)

    print(len(g.vertices))
    print(len(g.edges))
