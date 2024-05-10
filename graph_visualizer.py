import os

from matplotlib import pyplot as plt

from generate_data import GraphSplitter, XY


class GraphVisualizer:
    def __init__(self, save_img=False):
        self.xy = XY
        self.save_img = save_img
        self.dir_img = "images/"
        os.makedirs(self.dir_img, exist_ok=True)

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
                               title="Simple Graph", display_text=False):
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

        if display_text:
            for point in vertices.values():
                predicted_ft = tuple([round(v, 1) for v in point.predicted_ft])
                # text = f"{predicted_ft}|{point.label}|{point.predicted_lb}"
                # text = f"{predicted_ft}|ori={point.label}|pre={point.predicted_lb}"
                text = f"{predicted_ft}"
                ax.text(point.position[0], point.position[1], text, fontsize=10, ha='right')

        if self.save_img:
            if display_text:
                title = title + "_with_predicted_ft"
            suffix = title.replace("\n", '_')
            plt.savefig(f"{self.dir_img}/results__{suffix}.png")
        else:
            plt.show()

    def visualize_simple_graph_with_winding_number_heatmap_and_stroke_directions(
            self, splitter: GraphSplitter, vertices, edges, get_lb_function,
            winding_numbers, stroke_directions_from_to_tuple,
            title="Simple Graph", display_text=False):
        """
        Visualizes the graph with options to show predicted labels.

        :param: winding_numbers: [wn1, wn2, ...]
        :param: stroke_directions_from_to_tuple: [(from_v_id: to_v_id), ...]
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal

        # Normalize winding numbers for coloring
        norm = plt.Normalize(min(winding_numbers), max(winding_numbers))
        cmap = plt.get_cmap('viridis')

        # Draw vertices with color based on winding number
        for vertex_id, vertex in vertices.items():
            wn = winding_numbers[vertex_id]  # Get winding number for the vertex
            color = cmap(norm(wn))
            marker = 'o' if vertex.labeled else 'x'
            ax.scatter(vertex.position[0], vertex.position[1], color=color, marker=marker)

        # print(sorted(stroke_directions_from_to_tuple))

        # Draw edges
        stroke_directions_set = set(stroke_directions_from_to_tuple)
        for (i, j), edge in edges.items():

            point1 = vertices[i].position
            point2 = vertices[j].position

            if edge.is_stroke():
                l1 = vertices[i].label
                l2 = vertices[j].label

                if (l1, l2) in stroke_directions_set:
                    pass
                else:
                    assert (l2, l1) in stroke_directions_set
                    # change direction
                    point1 = vertices[j].position
                    point2 = vertices[i].position

                # Draw directed edges
                ax.annotate("", xy=point2, xytext=point1,
                            arrowprops=dict(
                                arrowstyle="simple, head_width=0.7, head_length=1.2", color='cyan', lw=0.2
                            ))
            else:
                # Draw undirected edges
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'gray', linestyle='-', linewidth=0.5)

        splitter.draw_dividers(ax)

        plt.xlim(-self.xy, self.xy)
        plt.ylim(-self.xy, self.xy)
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        # Colorbar for winding numbers
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Winding Number')

        # plt.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Labeled'),
        #             plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='r', markersize=10, label='Unlabeled')],
        #            loc='upper right')

        if display_text:
            for point, wn in zip(vertices.values(), winding_numbers):
                text = f"{round(wn, 2)}"
                ax.text(point.position[0], point.position[1], text, fontsize=10, ha='right')

        if self.save_img:
            if display_text:
                title = title + "_with_winding_number"
            suffix = title.replace("\n", '_')
            plt.savefig(f"{self.dir_img}/winding_number_with_stroke_direction__{suffix}.png")
        else:
            plt.show()
