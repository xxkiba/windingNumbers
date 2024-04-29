import argparse

import numpy as np
from matplotlib import pyplot as plt

"""
rectangle: -XY -> XY
"""
XY = 10
R = 6  # 2


def get_graph_spitter(num_categories, num_points):
    if num_categories == 2:
        return TwoCategorySplitter(num_categories, num_points)
    elif num_categories == 5:
        return FiveCategorySplitter(num_categories, num_points)
    else:
        raise ValueError("Unsupported number of categories")


class GraphSplitter:
    def __init__(self, num_categories, num_points):
        self.num_categories = num_categories
        self.num_points = num_points
        
        self.xy = XY
        self.r = R

        self.divider_color = "red"
        
    def get_num_categories(self):
        return self.num_categories
    
    def get_num_points(self):
        return self.num_points

    def generate_positions(self):
        positions = np.random.uniform(-self.xy, self.xy, (self.num_points, 2))
        return positions

    def split_graph_and_generate_points(self):
        """ Return positions, labels """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def draw_dividers(self, ax):
        raise NotImplementedError("This method should be overridden by subclasses.")


class TwoCategorySplitter(GraphSplitter):
    """
    Split the graph into two categories:
    - Inside a circle (radius r)
    - Outside the circle
    """
    def split_graph_and_generate_points(self):
        positions = self.generate_positions()
        labels = np.array([1 if np.linalg.norm(p) < self.r else 0 for p in positions])
        return positions, labels

    def draw_dividers(self, ax):
        circle = plt.Circle((0, 0), self.r, color=self.divider_color, fill=False, linestyle='--')
        ax.add_artist(circle)


class FiveCategorySplitter(GraphSplitter):
    """
    Split the graph into five categories:
    - Category 1: Inside a central circle (radius r)
    - Category 2: Upper right quadrant outside the circle
    - Category 3: Upper left quadrant outside the circle
    - Category 4: Lower left quadrant outside the circle
    - Category 5: Lower right quadrant outside the circle
    """

    def split_graph_and_generate_points(self):
        positions = self.generate_positions()
        labels = []
        for p in positions:
            if np.linalg.norm(p) < self.r:
                labels.append(1)
            elif p[0] >= 0 and p[1] >= 0:
                labels.append(2)
            elif p[0] < 0 and p[1] >= 0:
                labels.append(3)
            elif p[0] < 0 and p[1] < 0:
                labels.append(4)
            elif p[0] >= 0 and p[1] < 0:
                labels.append(5)
        labels = np.array(labels)

        return positions, labels

    def draw_dividers(self, ax):
        # Central circle
        circle = plt.Circle((0, 0), self.r, color=self.divider_color, fill=False, linestyle='--')
        ax.add_artist(circle)

        # Quadrant lines
        points = [(self.r, 0), (0, self.r), (-self.r, 0), (0, -self.r)]
        ends = [(self.xy, 0), (0, self.xy), (-self.xy, 0), (0, -self.xy)]

        for start, end in zip(points, ends):
            ax.plot([start[0], end[0]], [start[1], end[1]], color=self.divider_color, linestyle='--')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default=2, type=int, help="number of categories")
    parser.add_argument("--p", default=100, type=int, help="number of points")
    args = parser.parse_args()

    # Example usage:
    splitter = get_graph_spitter(num_categories=args.c, num_points=args.p)
    positions, labels = splitter.split_graph_and_generate_points()

    fig, ax = plt.subplots()

    ax.set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal

    from graph_visualizer import GraphVisualizer
    ax.scatter(positions[:, 0], positions[:, 1],
               c=[GraphVisualizer.get_category_color(l, args.c) for l in labels])
    splitter.draw_dividers(ax)
    plt.show()


if __name__ == '__main__':
    main()

