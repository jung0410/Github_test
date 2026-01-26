import os
from itertools import combinations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from lib.L1_Image_Conversion import *
# from lib.L2_Point_Detection import *
# from lib.L3_Point_Conversion import *
# from lib.L4_Zhangs_Calibration import *
from lib.L4_Pipeline_utility import *
# from lib.L6_ETC_visualization import *

def visualize_points(points):
    """
    Visualize 2D points using a scatter plot.

    Args:
        points: (N, 2) numpy array of (X, Y).
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], c='red', s=10, label='Points')
    plt.show()


def visualize_points_red_blue(points_red, points_blue):
    """
    Visualize two sets of points on the same plot.
    - Red (hollow circles): calibration points
    - Green (hollow triangles): ground-truth points

    Args:
        points_red: (N, 2) numpy array
        points_blue: (M, 2) numpy array
    """
    plt.figure(figsize=(10, 10))

    plt.scatter(
        points_red[:, 0], points_red[:, 1],
        edgecolors='red', facecolors='none', s=100,
        label='Calibration-point'
    )
    plt.scatter(
        points_blue[:, 0], points_blue[:, 1],
        edgecolors='green', facecolors='none', s=70, marker='^',
        label='GT-point'
    )

    plt.rcParams['font.family'] = 'Arial'
    plt.xlabel('X-axis (pixel)')
    plt.ylabel('Y-axis (pixel)')

    plt.xlim(0, 4000)
    plt.ylim(0, 3000)

    plt.gca().set_aspect(4000 / 3000)
    plt.legend(fontsize=18, frameon=False, loc='upper right')
    plt.autoscale(enable=False)
    plt.show()


def display_image(image, title="Image"):
    """
    Display a BGR image (OpenCV) using matplotlib (RGB).

    Args:
        image: BGR image (H, W, 3)
        title: plot title
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


def draw_points_on_image(image, points, color=(0, 0, 255), radius=1, thickness=-1):
    """
    Draw points on an image and show it.
    NOTE: Current implementation always uses green points (overrides `color`).

    Args:
        image: BGR image
        points: iterable of (x, y) points (float allowed)
        color: (B, G, R) - ignored (kept for compatibility)
        radius: circle radius
        thickness: circle thickness (-1 fills)

    Returns:
        image_rgb: RGB image with points drawn
    """
    # Force green for visibility (keeps original behavior).
    draw_color = (0, 255, 0)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for x, y in points:
        image_rgb = cv2.circle(
            image_rgb,
            (int(x), int(y)),
            radius,
            draw_color,
            thickness
        )

    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.title("image")
    plt.axis('off')
    plt.show()

    return image_rgb


def draw_points_on_large_image(image, points, color=(255, 0, 0), radius=2, thickness=-1, scale_factor=3):
    """
    Enlarge an image and draw points aligned to the enlarged coordinates.
    Points are scaled around the image center, then drawn on the enlarged image.

    NOTE: Current implementation draws single pixels (not circles) and uses green color.
          Also prints intermediate scaled coordinates.

    Args:
        image: BGR image
        points: (N, 2) numpy array-like points (float allowed)
        color: unused (kept for compatibility)
        radius: unused (kept for compatibility)
        thickness: unused (kept for compatibility)
        scale_factor: enlargement factor

    Returns:
        enlarged_image: RGB enlarged image with points drawn
    """
    enlarged_image = cv2.resize(
        image, None,
        fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_LANCZOS4
    )
    enlarged_image = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2RGB)

    def scale_points_from_center(pts, sf, image_shape):
        """
        Scale points around the image center.

        Args:
            pts: (N, 2)
            sf: scale factor
            image_shape: (H, W)

        Returns:
            new_positions: (N, 2) scaled point positions
        """
        H, W = image_shape
        center_y, center_x = H / 2, W / 2

        pts = np.asarray(pts, dtype=np.float64)
        moved = pts - np.array([center_x, center_y])
        scaled = np.round(moved * sf)


        new_positions = scaled + np.array([center_x, center_y])
        return new_positions

    H, W = enlarged_image.shape[:2]
    scaled_points = scale_points_from_center(points, scale_factor, (H, W))

    draw_color = (0, 255, 0)

    for x, y in scaled_points:
        x = int(x)
        y = int(y)

        # Guard against out-of-bounds indexing
        if 0 <= x < enlarged_image.shape[1] and 0 <= y < enlarged_image.shape[0]:
            enlarged_image[y, x] = draw_color

    plt.figure(figsize=(10, 8))
    plt.imshow(enlarged_image)
    plt.title("image")
    plt.axis('off')
    plt.show()

    return enlarged_image
