from graph import *


class Pipeline:
    def __init__(self):
        self.g = Graph()
        # self.g.build_graph(regenerate=False)
        self.g.build_simple_graph()

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
