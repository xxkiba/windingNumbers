import argparse
import itertools
import random
import time

import numpy as np
from tqdm import tqdm

from graph import *


def convert_seconds_to_hms(seconds):
    # if seconds < 1:
    #     return f"{round(seconds, 4)}s"
    if seconds <= 60:
        return f"{round(seconds, 2)}"

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Pipeline:
    def __init__(self, num_categories=5, num_points=100, knn_k=5, train_ratio=0.7, kmeans_iterations=10,
                 save_img=False, print_text=False,
                 sample_n=100, simple=True, feature_dimension=5):
        self.g = Graph(num_categories, num_points, knn_k, train_ratio,
                       save_img=save_img, print_text=print_text)

        self.c = num_categories
        self.n = num_points
        self.k = knn_k
        self.r = train_ratio

        self.simple = simple

        self.sample_n = sample_n

        self.predicted_ft_d = feature_dimension

        self.kmeans_iterations = kmeans_iterations

    def run(self):
        if self.simple:
            self.g.build_simple_graph()
        else:
            self.g.build_graph(regenerate=False)
        print("Graph Generated")

        print("Solving Laplace...")
        laplacian = self.get_laplacian()

        print("Generating sampled stroke direction combinations...")
        labels = self.g.get_all_labels()
        # all_combinations = list(self.generate_combinations(labels))
        # # print(all_combinations)
        # # [{(0, 1): -1}, {(0, 1): 1}]
        # sample_combinations = random.sample(all_combinations, min(len(all_combinations), self.sample_n))

        sample_combinations = self.generate_sampled_combinations(labels)
        print(len(sample_combinations))
        print(sample_combinations[0])

        start_t = time.time()

        print("Start Solving Equation Systems...")
        tv_wns = []
        for stroke_direction in tqdm(sample_combinations):
            sigmas = self.get_sigmas(stroke_direction)
            wn = self.calculate_winding_numbers(laplacian, sigmas)
            # print(stroke_direction)
            # print(wn)
            tv = self.calculate_total_variance(wn)
            tv_wns.append(dict(
                wn=wn,
                tv=tv,
                sd=stroke_direction,
            ))

        print("Calculating features...")
        fts = self.get_features(tv_wns)

        # print(fts)

        # fts = [p.feature for p in self.g.vertices.values()]
        print("Kmeans...")
        self.predict_labels(fts)

        print("===========")
        t_str = convert_seconds_to_hms(time.time() - start_t)
        print(f"{t_str}s")
        print("===========")

        print("Done! Summarizing results...")

        acc = self.cal_accuracy()

        print(f"{self.n} & {self.k} & {self.r} & {self.sample_n} & {self.predicted_ft_d} & \\textbf{{{acc:.2f}}} & {t_str} \\\\ \\hline")

        if self.simple:
            suffix = f"Feature Dimension={self.predicted_ft_d}, " \
                     f"Acc={acc}"
            self.g.visualize_simple_graph_with_winding_number_heatmap_and_stroke_directions(
                winding_numbers=self.save_for_draw["wns"],
                stroke_directions=self.save_for_draw["sds"],
                suffix=suffix,
            )

            suffix = f"Feature Dimension={self.predicted_ft_d}, " \
                     f"Sample Stroke Directions Num={self.sample_n}\n" \
                     f"Acc={acc}"
            self.g.visualize_simple_graph(pre=True, suffix=suffix)

    def generate_sampled_combinations(self, labels):
        # Generate all pairs of labels
        label_pairs = list(itertools.combinations(labels, 2))
        num_pairs = len(label_pairs)

        # All possible combinations would be 2 ** num_pairs (very large for big num_pairs)
        if 2 ** num_pairs < self.sample_n:
            # raise ValueError("Requested number of unique samples exceeds the total possible combinations")
            self.sample_n = 2 ** num_pairs

        """
        For simple graph, we need to change predicted_ft_d
        """
        if self.predicted_ft_d > self.sample_n:
            self.predicted_ft_d = self.sample_n

        # List to hold k different combinations
        unique_combinations = set()

        # Generate combinations until we have k unique ones
        while len(unique_combinations) < self.sample_n:
            # Generate one combination where each label pair is randomly assigned a 1 or -1
            random_combination = tuple(random.choice([-1, 1]) for _ in range(num_pairs))
            unique_combinations.add(random_combination)

        # Convert each tuple in the set back to a dictionary format for output
        combinations = []
        for combo in unique_combinations:
            combination_dict = {label_pairs[i]: combo[i] for i in range(num_pairs)}
            combinations.append(combination_dict)

        return combinations

    @staticmethod
    def generate_combinations(labels):
        """
        cannot work when there are many combinations
        """
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
                # adj[i, j] = 1
                # adj[j, i] = 1

        degrees = np.zeros((n, n))

        for i in range(n):
            degrees[i, i] = np.sum(adj[i, :])

        laplacian = degrees - adj
        # print(laplacian)

        return laplacian

    def get_sigmas(self, stroke_direction):
        """

        :param: stroke_direction: {(label1, label2): ±1}
        :return: sigmas: Matrix with opposite values on symmetric positions across the diagonal
        i.e. sigma[i, j] = -sigma[j, i]
        """
        # print("strokes ", stroke_direction)
        n = len(self.g.vertices)
        sigmas = np.zeros((n, n))
        for (i, j), value in self.g.edges.items():
            if value.is_stroke():
                for (_i, _j), _value in stroke_direction.items():
                    if (self.g.get_vertex(i).label, self.g.get_vertex(j).label) == (_i, _j):
                        sigmas[i, j] = _value
                        sigmas[j, i] = -_value
                    elif (self.g.get_vertex(j).label, self.g.get_vertex(i).label) == (_i, _j):
                        sigmas[j, i] = _value
                        sigmas[i, j] = -_value

        # print(sigmas)
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

            k += 1

        b_1 = np.zeros(n)
        b = np.append(b_1, b_2, axis=0)
        A = np.append(laplacian, weights, axis=0)
        # least square solution
        w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
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

    def get_features(self, tv_wns):
        """

        consider different situations of stoke directions
        feature is the minimum dimension winding numbers

        :param: tv_wns: [{tv: , wn: }, ...]

        :return: ft: (|V|, dimension)
        """

        sorted_tv_wns = sorted(tv_wns, key=lambda x: x["tv"], reverse=False)

        top_wns = [tv_wn["wn"] for tv_wn in sorted_tv_wns[:self.predicted_ft_d]]

        self.save_for_draw = dict(
            wns=sorted_tv_wns[0]["wn"],
            sds=sorted_tv_wns[0]["sd"],
        )

        # d × n -> n × d
        features = list(zip(*top_wns))
        features = [np.array(ft, dtype=float) for ft in features]
        print(np.array(features).shape)

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

    def cluster_by_kmeans(self):
        """
        semi-supervised Kmeans
        :return: assignments: {vid: lb}, centroids
        """

        num_iterations = self.kmeans_iterations

        # Step 1: Initialize centroids
        labels = set(v.label for v in self.g.vertices.values() if v.labeled)
        centroids = {label: np.array([0.0 for _ in range(self.predicted_ft_d)]) for label in labels}
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
                if not v.labeled:
                    closest = min(centroids, key=lambda x: np.linalg.norm(v.predicted_ft - centroids[x]))
                    assignments[v_id] = closest
                else:
                    assignments[v_id] = v.label

            # Step 3: Update centroids
            new_centroids = {label: np.array([0.0 for _ in range(self.predicted_ft_d)]) for label in labels}
            new_counts = {label: 0 for label in labels}

            for v_id, closest in assignments.items():
                new_centroids[closest] += self.g.vertices[v_id].predicted_ft
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

    def cal_accuracy(self):
        n = 0
        c = 0
        for vi, v in self.g.vertices.items():
            if v.labeled:
                continue
            if v.label == v.predicted_lb:
                c += 1
            n += 1
        acc = round(c / n, 4)
        print(f"{c} / {n} = {acc}")
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--categories", type=int, default=10, help="Number of categories for the graph splitter")
    parser.add_argument("-n", "--points", type=int, default=200, help="Number of points to generate")
    parser.add_argument("-k", "--knn_k", type=int, default=5, help="K value for KNN")
    parser.add_argument("-t", "-r", "--train_ratio", type=float, default=0.7, help="Training ratio")

    parser.add_argument("-i", "--iter_kmeans", type=int, default=10, help="Kmeans Iterations")

    parser.add_argument("--sample_n", default=100, type=int, help="Number of sampled stroke directions")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("-fd", "--feature_dimension", type=int, default=10, help="Feature dimension")

    parser.add_argument("--text", action="store_true")
    parser.add_argument("--save_img", action="store_true")

    args = parser.parse_args()

    print(args)

    p = Pipeline(num_categories=args.categories, num_points=args.points, knn_k=args.knn_k, train_ratio=args.train_ratio,
                 kmeans_iterations=args.iter_kmeans,
                 sample_n=args.sample_n,
                 save_img=args.save_img,
                 print_text=args.text,
                 simple=not args.hard, feature_dimension=args.feature_dimension)
    p.run()
