import itertools

from graph import *


class Pipeline:
    def __init__(self):
        self.g = Graph()

    def run(self):
        # self.g.build_graph(regenerate=False)
        self.g.build_simple_graph()
        print("Graph Generated")

        labels = self.g.get_all_labels()
        all_combinations = list(self.generate_combinations(labels))
        print(all_combinations)
        # [{(0, 1): -1}, {(0, 1): 1}]

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

        for i in range(n):
            for j in range(n):
                if self.g.edges[(i, j)] or self.g.edges[(j, i)]:
                    adj[i, j] = Point.gaussian_similarity(self.g.vertices[i], self.g.vertices[j], 1)
                    adj[j, i] = Point.gaussian_similarity(self.g.vertices[i], self.g.vertices[j], 1)

        degrees = np.zeros((n, n))

        for i in range(n):
            degrees[i, i] = np.sum(adj[i, :])

        laplacian = degrees - adj

        return laplacian

    def get_sigmas(self, stroke_direction):
        """

        :param stroke_direction: {(label1, label2): ±1}
        :return: sigmas: sigma vector (|S|, 1), value = ±1
        """

    def calculate_winding_numbers(self, laplacian, sigmas):
        """
        Solve the linear system:
        - L * W = 0
        - W1 - W2 = sigmas

        :param laplacian:
        :param sigmas:
        :return: wn: winding number values for vertices: (|S|, 1)
        """

    def calculate_total_variance(self, wn):
        """
        tv = sum (wi - wj) for eij in E/S

        :param wn: winding number values for vertices: (|S|, 1)
        :return: tv: total variance: float
        """

    def get_features(self, dimension=8):
        """

        consider different situations of stoke directions
        feature is the minimum dimension winding numbers

        :return: ft: (|V|, dimension)
        """

    def cluster(self):
        """
        spectral cluster to assign labels to vertices

        :return:
        """

    def cluster_by_kmeans(self):
        pass


if __name__ == '__main__':
    p = Pipeline()
    p.run()
