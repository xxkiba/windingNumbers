import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path

"""
rectangle: -XY -> XY
"""
XY = 10
R = 6  # 2


def get_graph_spitter(num_categories, num_points):

    SplitterClassDict = {
        2: TwoCategorySplitter,
        3: ThreeCategorySplitter,
        5: FiveCategorySplitter,
        10: TenCategorySplitter,
    }

    try:
        return SplitterClassDict[num_categories](num_categories, num_points)
    except KeyError:
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


class ThreeCategorySplitter(GraphSplitter):
    """
    Split the graph into three categories:
    1. Left of the S-curve
    2. Inside an ellipse
    3. Outside the ellipse and right of the S-curve

    Uses instance attributes to control the shapes and positions of the dividers.
    """
    def __init__(self, num_categories, num_points):
        super().__init__(num_categories, num_points)
        self.s_curve_func = lambda x: 2 * np.sin(x) - 3  # Lowering the S-curve
        self.ellipse_center = (0, 3)  # Center of the ellipse higher in the y-axis
        self.ellipse_axes = (7, 4)  # Increased semi-major (a) and semi-minor (b) axes

    def split_graph_and_generate_points(self):
        positions = self.generate_positions()
        labels = []
        for p in positions:
            if self.is_below_s_curve(p):
                labels.append(0)  # Left of the S-curve
            elif self.is_inside_ellipse(p):
                labels.append(1)  # Inside the ellipse
            else:
                labels.append(2)  # Outside the ellipse and right of the S-curve
        return positions, np.array(labels)

    def is_below_s_curve(self, point):
        x, y = point
        s_curve_y = self.s_curve_func(x)  # Using the function for the S-curve
        return y < s_curve_y

    def is_inside_ellipse(self, point):
        x, y = point
        center_x, center_y = self.ellipse_center
        a, b = self.ellipse_axes
        ellipse_eq = ((x - center_x)**2 / a**2) + ((y - center_y)**2 / b**2)
        return ellipse_eq < 1

    def draw_dividers(self, ax):
        x_vals = np.linspace(-10, 10, 400)
        y_vals = self.s_curve_func(x_vals)
        ax.plot(x_vals, y_vals, color='red', linestyle='--')
        # ax.plot(x_vals, y_vals, color='red', linestyle='--', label='Lowered S-curve')

        # ellipse = Ellipse(self.ellipse_center, 2*self.ellipse_axes[0], 2*self.ellipse_axes[1], color='blue',
        # fill=False, linestyle='--', label='Larger Ellipse')
        ellipse = Ellipse(self.ellipse_center, 2 * self.ellipse_axes[0], 2 * self.ellipse_axes[1], color='blue',
                          fill=False, linestyle='--')
        ax.add_patch(ellipse)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        # ax.legend()


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


class TenCategorySplitter(GraphSplitter):
    """
    Split the graph into ten sectors within a central circle.
    Each sector is formed by splitting the circle into 10 equal parts of 36 degrees each.
    """

    def split_graph_and_generate_points(self):
        positions = self.generate_positions()
        labels = []
        for p in positions:
            angle = np.arctan2(p[1], p[0]) % (2 * np.pi)  # Normalize angle to be between 0 and 2*pi
            sector = int(angle / (2 * np.pi / 10)) + 1  # Divide the circle into 10 equal parts
            labels.append(sector)

        return positions, np.array(labels)

    def draw_dividers(self, ax):
        # Assume self.xy is the extent of the plot boundary
        boundary_x = self.xy
        boundary_y = self.xy

        for i in range(10):
            angle = 2 * np.pi * i / 10
            # Assuming the plot boundary is centered at the origin and equal in all directions
            if -np.pi / 4 <= angle < np.pi / 4 or 3 * np.pi / 4 <= angle < 5 * np.pi / 4:
                # Right or left boundary
                end_x = boundary_x if np.cos(angle) > 0 else -boundary_x
                end_y = end_x * np.tan(angle)
            else:
                # Top or bottom boundary
                end_y = boundary_y if np.sin(angle) > 0 else -boundary_y
                end_x = end_y / np.tan(angle)

            # Check if the computed point exceeds the boundary, and adjust
            if abs(end_y) > boundary_y:
                end_y = boundary_y if end_y > 0 else -boundary_y
                end_x = end_y / np.tan(angle)
            if abs(end_x) > boundary_x:
                end_x = boundary_x if end_x > 0 else -boundary_x
                end_y = end_x * np.tan(angle)

            ax.plot([0, end_x], [0, end_y], color=self.divider_color, linestyle='--')


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

