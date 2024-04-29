from matplotlib import pyplot as plt

from generate_data import GraphSplitter, XY


class GraphVisualizer:
    def __init__(self):
        self.xy = XY

    @staticmethod
    def get_category_color_from_cmap(label, num_categories):
        """
        Returns a color from the colormap based on the label and number of categories.
        """
        cmap = plt.get_cmap('viridis', num_categories)
        return cmap(label / (num_categories - 1))
        # return cmap(label)

    @staticmethod
    def get_category_color(label, num_categories):
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
        if num_categories > len(colors):
            raise ValueError("Number of categories exceeds the number of predefined colors")
        return colors[label % num_categories]

    def visualize_simple_graph(self, splitter: GraphSplitter, vertices, edges, get_lb_function,
                               title="Simple Graph", display_feature=False):
        """
        Visualizes the graph with options to show predicted labels.
        """
        fig, ax = plt.subplots()

        ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal

        # Determine the number of categories by checking labels or predicted labels
        num_categories = splitter.get_num_categories()

        for point in vertices.values():
            lb = get_lb_function(point)
            color = self.get_category_color(lb, num_categories)
            marker = 'o' if point.labeled else 'x'
            ax.scatter(point.position[0], point.position[1], c=[color], marker=marker)

        # Draw edges between points
        for (i, j) in edges.keys():
            point1 = vertices[i].position
            point2 = vertices[j].position
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'gray', linestyle='-',
                    linewidth=0.5)

        splitter.draw_dividers(ax)

        plt.xlim(-self.xy, self.xy)
        plt.ylim(-self.xy, self.xy)

        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        # plt.grid(True)
        plt.legend([plt.Line2D([0], [0], marker='o', color='w', label='Labeled', markerfacecolor='g', markersize=20),
                    plt.Line2D([0], [0], marker='x', color='w', label='Unlabeled', markerfacecolor='g', markersize=20)],
                   ['Labeled', 'Unlabeled'], loc='upper right')

        # Legend
        ax.scatter([], [], c='black', marker='o', label='Labeled')
        ax.scatter([], [], c='black', marker='x', label='Unlabeled')
        # ax.scatter([], [], c='blue', marker='ox', label='Labeled Outside')
        plt.legend(loc='upper right')

        if display_feature:
            self.visualize_feature(ax, vertices)

        plt.show()

    @staticmethod
    def visualize_feature(ax, vertices):
        print("Texting features")
        for point in vertices.values():
            predicted_ft = tuple([round(v, 1) for v in point.predicted_ft])
            ax.text(point.position[0], point.position[1],
                    str(predicted_ft), fontsize=10, ha='right')