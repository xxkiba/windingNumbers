import itertools
import random

import numpy as np
from tqdm import tqdm

from graph import *


class Pipeline:
    def __init__(self, sample_n=10):
        self.g = Graph()
        self.sample_n = sample_n

    def run(self):
        # self.g.build_graph(regenerate=False)
        self.g.build_simple_graph()
        # print("Graph Generated")

        laplacian = self.get_laplacian()

        labels = self.g.get_all_labels()
        all_combinations = list(self.generate_combinations(labels))
        # print(all_combinations)
        # [{(0, 1): -1}, {(0, 1): 1}]

        sample_combinations = random.sample(all_combinations, min(len(all_combinations), self.sample_n))

        tv_wns = []
        for stroke_direction in tqdm(sample_combinations):
            sigmas = self.get_sigmas(stroke_direction)
            wn = self.calculate_winding_numbers(laplacian, sigmas)
            tv = self.calculate_total_variance(wn)
            tv_wns.append(dict(
                wn=wn,
                tv=tv,
            ))

        fts = self.get_features(tv_wns)

        # fts = [p.feature for p in self.g.vertices.values()]

        self.predict_labels(fts)

        self.g.visualize_simple_graph(pre=True)

    @staticmethod
    def generate_combinations(labels):
        # Generate all possible combinations of pairs of labels
        label_pairs = list(itertools.combinations(labels, 2))
        # The total number of combinations is 2 raised to the power of the number of label pairs
        total_combinations = 2 ** len(label_pairs)

        # Iterate over all possible combinations
        for i in range(total_combinations):
            # Use bit manipulation to determine the direction for each pair of labels
            combo = {}
            for j, pair in enumerate(label_pairs):
                # If the j-th bit is 1, direction is 1; otherwise, it's -1
                direction = 1 if (i & (1 << j)) else -1
                combo[pair] = direction
            yield combo

    def get_laplacian(self):
        """

        :return: laplacian (|V|, |V|)
        """

        n = len(self.g.vertices)

        adj = np.zeros((n, n))

        for (i, j), value in self.g.edges.items():
            if not value.is_stroke():
                # print(i, j)
                adj[i, j] = Point.gaussian_similarity(self.g.vertices[i], self.g.vertices[j], 1)
                adj[j, i] = Point.gaussian_similarity(self.g.vertices[i], self.g.vertices[j], 1)

        degrees = np.zeros((n, n))

        for i in range(n):
            degrees[i, i] = np.sum(adj[i, :])

        laplacian = degrees - adj

        return laplacian

    def get_sigmas(self, stroke_direction):
        """

        :param: stroke_direction: {(label1, label2): ±1}
        :return: sigmas: sigma vector ( |S|, 1), value = ±1
        """
        # print("strokes ", stroke_direction)
        n = len(self.g.vertices)
        sigmas = np.zeros((n, n))
        for (i, j), value in self.g.edges.items():
            if value.is_stroke:
                for (_i, _j), _value in stroke_direction.items():
                    if (self.g.get_vertex(i).label, self.g.get_vertex(j).label) == (_i, _j) or (self.g.get_vertex(j).label, self.g.get_vertex(i).label) == (_i, _j):
                        sigmas[i, j] = _value


        print(sigmas)
        return sigmas

    def calculate_winding_numbers(self, laplacian, sigmas):
        """
        Solve the linear system:
        - L * W = 0
        - W1 - W2 = sigmas

        :param: laplacian:
        :param: sigmas:
        :return: wn: winding number values for vertices: (|S|, 1)
        """
        n = len(self.g.vertices)
        w_ij = len(self.g.strokes_ij)
        b_2 = np.zeros(w_ij)
        weights = np.zeros((w_ij, n))
        k = 0
        for (i, j) in self.g.strokes_ij:
            weights[k, i] = 1
            weights[k, j] = -1

            b_2[k] = sigmas[i, j]
            # print(sigmas[i, j])
            print(i, j)
            k += 1
        b_1 = np.zeros(n)
        b = np.append(b_1, b_2, axis=0)
        A = np.append(laplacian, weights, axis=0)
        # print(b_2)
        ## least square solution
        w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        print(w)
        return w

    def calculate_total_variance(self, wn):
        """
        tv = sum (wi - wj) for eij in E/S

        :param: wn: winding number values for vertices: (|S|, 1)
        :return: tv: total variance: float
        """
        tv = 0
        for (i, j), edge in self.g.edges.items():
            if not edge.is_stroke():
                tv += abs(wn[i] - wn[j])
        return tv

    def get_features(self, tv_wns, dimension=5):
        """

        consider different situations of stoke directions
        feature is the minimum dimension winding numbers

        :param: tv_wns: [{tv: , wn: }, ...]

        :return: ft: (|V|, dimension)
        """

        sorted_tv_wns = sorted(tv_wns, key=lambda x: x["tv"], reverse=True)

        top_wns = [tv_wn["wn"] for tv_wn in sorted_tv_wns[:dimension]]

        # d × n -> n × d
        features = list(zip(*top_wns))
        features = [np.array(ft, dtype=float) for ft in features]

        return features

    def predict_labels(self, fts):
        """
        spectral cluster to assign labels to vertices

        :return:
        """

        for i, ft in enumerate(fts):
            self.g.vertices[i].predicted_ft = ft

        assignments, _ = self.cluster_by_kmeans()

        for vi, lb in assignments.items():
            self.g.vertices[vi].predicted_lb = lb

    def cluster_by_kmeans(self, num_iterations=5):
        """

        :return: assignments: {vid: lb}, centroids
        """

        # Step 1: Initialize centroids
        labels = set(v.label for v in self.g.vertices.values() if v.labeled)
        centroids = {label: np.array([0.0, 0.0]) for label in labels}
        counts = {label: 0 for label in labels}

        # Calculate initial centroid positions
        for v in self.g.vertices.values():
            if v.labeled:
                centroids[v.label] += v.predicted_ft
                counts[v.label] += 1

        # Average the sums to get initial centroids
        for label in labels:
            if counts[label] > 0:
                centroids[label] /= counts[label]

        # print(centroids)
        # print(counts)

        assignments = {}

        # print("Kmeans clustering to predict labels...")
        # K-means iteration
        for _ in tqdm(range(num_iterations)):  # Run for a fixed number of iterations
            # Step 2: Assign vertices to the nearest centroid
            for v_id, v in self.g.vertices.items():
                closest = min(centroids, key=lambda x: np.linalg.norm(v.predicted_ft - centroids[x]))
                assignments[v_id] = closest

            # Step 3: Update centroids
            new_centroids = {label: np.array([0.0, 0.0]) for label in labels}
            new_counts = {label: 0 for label in labels}

            for v_id, closest in assignments.items():
                new_centroids[closest] += self.g.vertices[v_id].feature
                new_counts[closest] += 1

            # Average the sums to update centroids
            for label in labels:
                if new_counts[label] > 0:
                    new_centroids[label] = new_centroids[label] / new_counts[label]

            centroids = new_centroids

            # # for experiment
            # for i, lb in assignments.items():
            #     self.g.vertices[i].predicted_lb = lb
            # self.visualize_results()

        return assignments, centroids

    def visualize_results(self):
        self.g.visualize_simple_graph(pre=True)


if __name__ == '__main__':
    p = Pipeline()
    p.run()
