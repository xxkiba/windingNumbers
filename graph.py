from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import generate_data
import graph_visualizer
from load_data import load_data


class Point:
    def __init__(self, ID, position, label, labeled):
        self.ID = ID
        self.position = position
        self.label = label
        self.labeled = labeled

        self.predicted_ft = None
        self.predicted_lb = None

    @staticmethod
    def get_distance(v1: Point, v2: Point):
        return np.linalg.norm(v1.position - v2.position)

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
    def __init__(self, num_categories=2, num_points=50, knn_k=5, train_ratio=0.7, print_text=False):
        # ID: Point
        self.vertices = {}
        # (i, j) -> edge
        self.edges = {}
        # [(i, j), ...]
        self.strokes_ij = []

        self.num_categories = num_categories
        self.num_points = num_points
        self.knn_k = knn_k
        self.train_ratio = train_ratio
        self.print_text = print_text

        self.splitter = generate_data.get_graph_spitter(self.num_categories, self.num_points)
        self.visualizer = graph_visualizer.GraphVisualizer()

    def get_all_labels(self):
        labels = set()
        for v in self.vertices.values():
            if v.labeled:
                labels.add(v.label)
        return sorted(list(labels))

    def build_simple_graph(self):
        positions, labels = self.splitter.split_graph_and_generate_points()

        for ID, (position, label) in enumerate(zip(positions, labels)):
            if ID < self.train_ratio * len(labels):
                self.vertices[ID] = Point(ID, position, label, labeled=True)
            else:
                self.vertices[ID] = Point(ID, position, label, labeled=False)

        self._generate_edges_by_knn(positions)

        self.strokes_ij = [(i, j) for (i, j), edge in self.edges.items() if edge.is_stroke()]

    def visualize_simple_graph(self, pre=False):
        # pre=True means show predicted labels

        if pre:
            def _get_lb(p: Point):
                return p.predicted_lb
        else:
            def _get_lb(p: Point):
                return p.label

        self.visualizer.visualize_simple_graph(
            splitter=self.splitter,
            vertices=self.vertices,
            edges=self.edges,
            get_lb_function=_get_lb,
            title=f"Graph Visualization\n"
                  f"CLS={self.num_categories}, N={self.num_points}, K={self.knn_k}, TrainRatio={self.train_ratio}",
            display_text=pre and self.print_text,
        )

        plt.show()

    def visualize_simple_graph_with_winding_number_heatmap_and_stroke_directions(
            self, winding_numbers, stroke_directions
    ):
        """
        :param: winding_numbers: [wn1, wn2, ...]
        :param: stroke_directions {(from_v_id, to_v_id): ±1, ...}
        """

        def _get_lb(p: Point):
            return p.predicted_lb

        stroke_directions_from_to_tuple = [
            (from_id, to_id) if dir_val == 1 else (to_id, from_id)
            for (from_id, to_id), dir_val in stroke_directions.items()
        ]

        self.visualizer.visualize_simple_graph_with_winding_number_heatmap_and_stroke_directions(
            splitter=self.splitter,
            vertices=self.vertices,
            edges=self.edges,
            winding_numbers=winding_numbers,
            stroke_directions_from_to_tuple=stroke_directions_from_to_tuple,
            get_lb_function=_get_lb,
            title=f"Graph Visualization\n"
                  f"CLS={self.num_categories}, N={self.num_points}, K={self.knn_k}, TrainRatio={self.train_ratio}",
            display_feature=self.print_text,
        )


    def build_graph(self, regenerate=False):

        dir_graph = f"graphs/{self.num_points}/"
        if os.path.isdir(dir_graph) and not regenerate:
            positions = np.load(f'{dir_graph}/positions.npy')
            labels = np.load(f'{dir_graph}/labels.npy')
        else:
            positions, labels = load_data(limit=self.num_points)
            os.makedirs(dir_graph, exist_ok=True)
            np.save(f'{dir_graph}/positions.npy', positions)
            np.save(f'{dir_graph}/labels.npy', labels)

        for ID, (position, label) in enumerate(zip(positions, labels)):
            # in future work, we need to guarantee that there's no new labels
            if ID < self.train_ratio * len(labels):
                self.vertices[ID] = Point(ID, position, label, labeled=True)
            else:
                self.vertices[ID] = Point(ID, position, label, labeled=False)
        self._generate_edges_by_knn(positions)
        self.strokes_ij = [(i, j) for (i, j), edge in self.edges.items() if edge.is_stroke()]

    def _generate_edges_by_knn(self, positions):
        """
        Nearest Neighbour will contain themselves
        So knn_k + 1
        """
        n_neighbors = NearestNeighbors(n_neighbors=self.knn_k + 1, algorithm='auto').fit(positions)
        distances, indices = n_neighbors.kneighbors(positions)
        # print("indices")
        # for index, indice in enumerate(indices):
        #     print(index, indice)
        # print(distances)
        # print(indices)

        edges = set()
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                # print(i,neighbor)
                # make sure i < j
                if i < neighbor:
                    edges.add((i, neighbor))
                elif i > neighbor:
                    edges.add((neighbor, i))
                # pass if i == neighbour

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("-c", "--categories", type=int, default=2, help="Number of categories for the graph splitter")
    parser.add_argument("-n", "--points", type=int, default=50, help="Number of points to generate")
    parser.add_argument("-k", "--knn_k", type=int, default=5, help="K value for KNN")
    parser.add_argument("-t", "-r", "--train_ratio", type=float, default=0.7, help="Training ratio")
    args = parser.parse_args()

    g = Graph(num_categories=args.categories, num_points=args.points, knn_k=args.knn_k, train_ratio=args.train_ratio)

    # g.build_graph(regenerate=False)

    g.build_simple_graph()
    g.visualize_simple_graph()

    print(len(g.vertices))
    print(len(g.edges))
