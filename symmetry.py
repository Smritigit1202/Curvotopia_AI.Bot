# Define the file path directly
file_path = '/content/occlusion2.csv'  # Adjust this path as needed


import numpy as np
import matplotlib.pyplot as plt

# Function to read CSV file
def read_csv(csv_path):
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    if np_path_XYs.ndim != 2 or np_path_XYs.shape[1] < 3:
        print("CSV file does not have the expected format.")
        return []

    path_XYs = []
    unique_paths = np.unique(np_path_XYs[:, 0])

    for i in unique_paths:
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        unique_xs = np.unique(npXYs[:, 0])
        XYs = []

        for j in unique_xs:
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)

        path_XYs.append(XYs)

    return path_XYs

# Function to plot points
def plot(points):
    if not points:
        print("No points to plot.")
        return

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(points):
        if not XYs:
            print(f"No data in path {i}.")
            continue
        for XY in XYs:
            if XY.shape[0] < 2:
                print("Not enough points to plot.")
                continue
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
    ax.set_aspect('equal')
    plt.show()

# Read the CSV file
path_XYs = read_csv(file_path)
if not path_XYs:
    raise ValueError("No data loaded. Check the CSV file.")

# Print the structure
print("Number of shapes in file:", len(path_XYs))
for i, XYs in enumerate(path_XYs):
    print(f"Path {i}: {len(XYs)} paths")
    for j, XY in enumerate(XYs):
        print(f"Path {i}, Subpath {j}: {XY.shape}")

# Plot all shapes from the CSV file
plot(path_XYs)


import cv2
import math
from matplotlib.patches import Ellipse

# Functions for calculations and processing
def calculate_centroid(points):
    pointsX = np.array(points[:, 0])
    pointsY = np.array(points[:, 1])
    medianX = np.median(pointsX)
    medianY = np.median(pointsY)
    return (medianX, medianY)

def calculate_angle(point, centroid):
    delta_y = point[1] - centroid[1]
    delta_x = point[0] - centroid[0]
    angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
    return angle

def calculate_angle_between_points(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

def identify_shape(count):
    if count < 4:
        return "Circle or ellipse"
    else:
        return "Polygon"

def find_contours(points):
    points = np.array(points, dtype=np.float32)
    points = points.reshape((-1, 1, 2))
    img = np.zeros((1000, 1000), dtype=np.uint8)
    min_x, min_y = np.min(points[:, 0, :], axis=0)
    max_x, max_y = np.max(points[:, 0, :], axis=0)
    scale = 1000 / max(max_x - min_x, max_y - min_y)
    points = (points - [min_x, min_y]) * scale

    for point in points:
        cv2.circle(img, tuple(np.round(point[0]).astype(int)), 1, 255, -1)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def plot_contours(contours):
    fig, ax = plt.subplots()
    for contour in contours:
        contour = contour.reshape(-1, 2)
        ax.plot(contour[:, 0], contour[:, 1], marker='o', linestyle='-')
    ax.set_aspect('equal')
    plt.show()

def convex_hull(points):
    points = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points)
    return hull

def plot_convex_hull(hull):
    plt.figure()
    plt.plot(hull[:, 0, 0], hull[:, 0, 1], marker='o')
    plt.title("Convex Hull")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

def fit_ellipse(points):
    points = np.array(points, dtype=np.float32)
    if len(points) >= 5:
        ellipse = cv2.fitEllipse(points)
        return ellipse
    else:
        return None


# Function to plot the fitted ellipse
def plot_ellipse(ellipse, centroid):
    fig, ax = plt.subplots()
    if ellipse:
        center, (MA, ma), angle = ellipse
        ellipse_patch = Ellipse(center, MA, ma, angle, edgecolor='r', facecolor='none')
        ax.add_patch(ellipse_patch)

        # Plot centroid
        ax.plot(center[0], center[1], 'ro', label='Centroid')
        ax.text(center[0], center[1], 'Centroid', horizontalalignment='right')

        # Set axis limits to ensure the ellipse is fully visible
        ax.set_xlim(center[0] - 1.2 * MA, center[0] + 1.2 * MA)
        ax.set_ylim(center[1] - 1.2 * ma, center[1] + 1.2 * ma)

        plt.title("Fitted Ellipse")
    else:
        print("Not enough points to fit an ellipse.")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_aspect('equal')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()



def process_and_plot_shape(XYs, index):
    for XY in XYs:
        if XY.shape[0] < 3:
            print(f"Not enough points to process shape {index}.")
            continue

        centroid = calculate_centroid(XY)
        print(f"Centroid of shape {index}: {centroid}")

        # Plot the shape
        plt.figure()
        plt.plot(XY[:, 0], XY[:, 1], marker='o')
        plt.title(f"Shape {index}")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.show()

        # Find and plot contours
        contours = find_contours(XY)
        if contours:
            plot_contours(contours)
        else:
            print(f"No contours found for shape {index}.")

        # Calculate and plot convex hull
        hull = convex_hull(XY)
        if hull.shape[0] > 2:
            plot_convex_hull(hull)
        else:
            print(f"Convex hull not valid for shape {index}.")

        # Fit and plot ellipse if possible
        ellipse = fit_ellipse(XY)
        plot_ellipse(ellipse, centroid)

        # Detect and plot critical points
        critical_threshold = 20  # Angle difference threshold to detect vertices
        angles = [calculate_angle(point, centroid) for point in XY]
        angles = np.array(angles)
        angles = (angles + 360) % 360
        anglesArgSort = np.argsort(angles)
        XY = XY[anglesArgSort]

        diff_angles = np.abs(np.diff(angles))
        critical_pts = []
        for i in range(len(diff_angles)):
            if diff_angles[i] >= critical_threshold:
                critical_pts.append(XY[i + 1])

        if critical_pts and not np.array_equal(critical_pts[0], critical_pts[-1]):
            critical_pts.append(critical_pts[0])

        critical_pts = np.array(critical_pts)

        print(f"Critical Points Count for shape {index}:", len(critical_pts))
        print(f"Critical Points for shape {index}:", critical_pts)

        shape_type = identify_shape(len(critical_pts))
        print(f"Detected Shape for shape {index}:", shape_type)

        if shape_type == "Circle or ellipse":
            dis = []
            for i in range(len(XY) - 1):
                x1, y1 = centroid
                x2, y2 = XY[i + 1]
                m = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dis.append(m)
            dis_array = np.array(dis)

            distdiff = []
            for i in range(len(dis_array) - 1):
                d = dis_array[i + 1] - dis_array[i]
                distdiff.append(d)

            if len(distdiff) > 0 and (sum(distdiff) / len(distdiff)) < 2:
                rad = sum(dis_array) / len(dis_array)
                fig, ax = plt.subplots()
                Drawcircle = plt.Circle(centroid, rad, fill=False)
                ax.add_patch(Drawcircle)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(centroid[0] - rad, centroid[0] + rad)
                ax.set_ylim(centroid[1] - rad, centroid[1] + rad)
                plt.title("Detected Circle")
                plt.show()

        else:
            print(f"Shape {index} is a Polygon")
            if len(critical_pts) >= 3:
                plt.figure()
                plt.plot([point[0] for point in critical_pts], [point[1] for point in critical_pts], 'o-', markersize=5)
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.title(f"Polygon of Shape {index}")
                plt.grid(True)
                plt.show()
                print(f"Critical points plotted successfully for shape {index}.")
            else:
                print(f"Not enough critical points to plot a polygon for shape {index}.")

# Process and plot each shape in the CSV file
for index, XYs in enumerate(path_XYs):
    process_and_plot_shape(XYs, index)


import os
import numpy as np
import cv2

def scale_points_to_image(points, img_size):
    """
    Scale points to fit within the image dimensions.
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    scale = img_size / max(max_x - min_x, max_y - min_y)
    scaled_points = (points - [min_x, min_y]) * scale
    return scaled_points, scale, min_x, min_y

def process_and_save_images(path_XYs, output_folder='/content/finished_images'):
    """
    Process each shape and save the finished images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for index, shape_points in enumerate(path_XYs):
        for j, XY in enumerate(shape_points):
            if XY.shape[0] < 3:
                continue  # Skip shapes that don't have enough points

            # Create a blank canvas
            img_size = 1000
            img = np.zeros((img_size, img_size), dtype=np.uint8)

            # Scale points
            scaled_points, _, _, _ = scale_points_to_image(XY, img_size)

            # Convert to integer coordinates
            scaled_points = np.round(scaled_points).astype(int)

            # Draw the shape on the canvas
            cv2.polylines(img, [scaled_points], isClosed=True, color=255, thickness=2)

            # Save the processed image
            output_path = f"{output_folder}/shape_{index}_subpath_{j}.png"
            cv2.imwrite(output_path, img)
            print(f"Saved finished image: {output_path}")

# Example usage
process_and_save_images(path_XYs)


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def combine_finished_images(image_folder='/content/finished_images', img_size=1000):
    """
    Combine finished images from the specified folder into a single image.
    """
    # Create a blank canvas
    combined_img = np.zeros((img_size, img_size), dtype=np.uint8)

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Add the current image to the combined image
        combined_img = np.maximum(combined_img, img)

    return combined_img

# Example usage
combined_image = combine_finished_images()

# Display the combined image
plt.figure(figsize=(8, 8))
plt.imshow(combined_image, cmap='gray')
plt.title("Combined Finished Shapes Image")
plt.axis('off')  # Turn off axis
plt.show()

# Save the combined image if needed
cv2.imwrite('/content/combined_finished_shapes_image.png', combined_image)
