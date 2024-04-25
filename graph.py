from __future__ import annotations

import os

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from load_data import load_data


class Point:
    def __init__(self, ID, feature, label, labeled):
        self.ID = ID
        self.feature = feature
        self.label = label
        self.labeled = labeled

        self.predicted_ft = None
        self.predicted_lb = None

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
        if not self.v1.labeled or not self.v2.labeled:
            return False
        # print(self.v1.label, self.v2.label)
        return self.v1.label != self.v2.label





class Graph:
    def __init__(self):
        # ID: Point
        self.vertices = {}
        # (i, j) -> edge
        self.edges = {}
        # [(i, j), ...]
        self.strokes_ij = []

        self.num_points = 50
        self.knn_k = 5
        self.train_ratio = 0.7

        # For simple graph
        self.xy = 2  # -xy, +xy
        self.r = 1.5

    def get_all_labels(self):
        labels = set()
        for v in self.vertices.values():
            if v.labeled:
                labels.add(v.label)
        return sorted(list(labels))

    def build_simple_graph(self):
        features = np.random.uniform(-self.xy, self.xy, (self.num_points, 2))
        labels = np.array([1 if np.linalg.norm(f) < self.r else 0 for f in features])

        for ID, (feature, label) in enumerate(zip(features, labels)):
            if ID < self.train_ratio * len(labels):
                self.vertices[ID] = Point(ID, feature, label, labeled=True)
            else:
                self.vertices[ID] = Point(ID, feature, label, labeled=False)

        self._generate_edges_by_knn(features)

        self.strokes_ij = [(i, j) for (i, j), edge in self.edges.items() if edge.is_stroke()]

    def visualize_simple_graph(self, pre=False):
        # pre=True means show predicted labels

        def _get_lb(p: Point):
            if pre:
                return p.predicted_lb
            else:
                return p.label

        fig, ax = plt.subplots()
        for point in self.vertices.values():
            color = 'red' if _get_lb(point) == 1 else 'blue'
            marker = 'o' if point.labeled else 'x'
            ax.scatter(point.feature[0], point.feature[1], c=color, marker=marker)

        # Draw edges between points
        for (i, j) in self.edges.keys():
            point1 = self.vertices[i].feature
            point2 = self.vertices[j].feature
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'gray', linestyle='-',
                    linewidth=0.5)  # Draw line between points

        # Draw a circle for visual boundary
        circle = plt.Circle((0, 0), self.r, color='green', fill=False)
        ax.add_artist(circle)

        plt.xlim(-self.xy, self.xy)
        plt.ylim(-self.xy, self.xy)

        ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal

        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.title(f'Graph Visualization\nK={self.knn_k}, N={self.num_points}, TrainRatio={self.train_ratio}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)

        # Legend
        ax.scatter([], [], c='red', marker='o', label='Labeled Inside')
        ax.scatter([], [], c='red', marker='x', label='Unlabeled Inside')
        ax.scatter([], [], c='blue', marker='o', label='Labeled Outside')
        ax.scatter([], [], c='blue', marker='x', label='Unlabeled Outside')
        plt.legend(loc='upper right')

        plt.show()

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
            # in future work, we need to guarantee that there's no new labels
            if ID < self.train_ratio * len(labels):
                self.vertices[ID] = Point(ID, feature, label, labeled=True)
            else:
                self.vertices[ID] = Point(ID, feature, label, labeled=False)
        self._generate_edges_by_knn(features)
        self.strokes_ij = [(i, j) for (i, j), edge in self.edges.items() if edge.is_stroke()]

    def _generate_edges_by_knn(self, features):
        n_neighbors = NearestNeighbors(n_neighbors=self.knn_k, algorithm='auto').fit(features)
        distances, indices = n_neighbors.kneighbors(features)
        print("indices")
        for index, indice in enumerate(indices):
            print(index, indice)

        edges = set()
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                # print(i,neighbor)
                # make sure i < j
                if i < neighbor:
                    edges.add((i, neighbor))
                else:
                    edges.add((neighbor, i))

        for i, j in edges:
            self.edges[(i, j)] = Edge(self.vertices[i], self.vertices[j])

    def get_edge(self, i, j):
        # assert i < j
        return self.edges[tuple(sorted([i, j]))]

    def get_vertex(self, i):
        return self.vertices[i]

    def get_strokes_ij(self):
        return self.strokes_ij


if __name__ == '__main__':
    g = Graph()

    # g.build_graph(regenerate=False)

    g.build_simple_graph()
    g.visualize_simple_graph()

    print(len(g.vertices))
    print(len(g.edges))
