import numpy as np

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


import matplotlib.pyplot as plt

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


path_XYs = read_csv(r"/content/isolated.csv")
if not path_XYs:
    raise ValueError("No data loaded. Check the CSV file.")

plot(path_XYs)

print(len(path_XYs))
for i, XYs in enumerate(path_XYs):
    print(f"Path {i}: {len(XYs)} paths")
    for j, XY in enumerate(XYs):
        print(f"Path {i}, Subpath {j}: {XY.shape}")


import pandas as pd

try:
    db = pd.read_csv(r"/content/isolated_sol.csv")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    db = pd.DataFrame()

if db.empty:
    raise ValueError("DataFrame is empty. Check the CSV file.")

print(db.shape)

if db.shape[1] < 2:
    raise ValueError("DataFrame does not have the expected format.")

col0 = db.iloc[:, 1]
print(col0.unique())


fig1Points = path_XYs[0][0]
if fig1Points.shape[0] < 3:
    raise ValueError("Not enough points to calculate centroid or plot.")

plot([[fig1Points]])

import numpy as np

pointsX = np.array(fig1Points[:, 0])
pointsY = np.array(fig1Points[:, 1])

if len(pointsX) == 0 or len(pointsY) == 0:
    raise ValueError("Points arrays are empty.")

medianX = np.median(pointsX)
medianY = np.median(pointsY)
centroid = (medianX, medianY)
print("Centroid:", centroid)

def calculate_angle(point, centroid):
    delta_y = point[1] - centroid[1]
    delta_x = point[0] - centroid[0]
    angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
    return angle

angles = [calculate_angle(point, centroid) for point in fig1Points]
angles = np.array(angles)
angles = (angles + 360) % 360
anglesArgSort = np.argsort(angles)
fig1Points = fig1Points[anglesArgSort]

plot([[fig1Points]])

anglesSorted = [angles[i] for i in anglesArgSort]
plt.plot(fig1Points[:, 0], fig1Points[:, 1], marker='o')
plt.title("Figpoints")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()


import math

def calculate_angle(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

if len(fig1Points) < 2:
    raise ValueError("Not enough points to calculate angles.")

angleSp = []
for i in range(len(fig1Points) - 1):
    x1, y1 = fig1Points[i]
    x2, y2 = fig1Points[i + 1]
    angleS = (calculate_angle(x1, y1, x2, y2) + 0)
    angleS = (angleS + 360) % 360
    angleSp.append(angleS)

angleSp = np.array(angleSp)

if len(angleSp) < 2:
    raise ValueError("Not enough angle data to plot.")

angleM = sorted(angleSp)
plt.plot([i for i in range(len(angleM))], angleM, marker='o')
plt.title("Sorted Angles")
plt.xlabel("x axis")
plt.ylabel("Angle")
plt.show()

plt.plot([i for i in range(len(angleSp))], angleSp, marker='o')
plt.title("Unsorted Angles")
plt.xlabel("x axis")
plt.ylabel("Angle")
plt.show()

diffAngles = np.abs(angleSp[1:] - angleSp[:-1])
print("Sorted differences between angles:", sorted(diffAngles))

anglediff = []
for i in range(len(angleM) - 1):
    x1 = angleSp[i]
    x2 = angleSp[i + 1]
    anglek = x2 - x1
    anglediff.append(anglek)

print("Angle differences:", anglediff)


import math
def calculate_distances(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.

    :param x1: X-coordinate of the first point.
    :param y1: Y-coordinate of the first point.
    :param x2: X-coordinate of the second point.
    :param y2: Y-coordinate of the second point.
    :return: The Euclidean distance between the two points.
    """
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

rad =0


def identify_shape(count):
    """
    Identify the shape based on count.

    :param count: An integer used to determine the shape.
    :return: Shape type as a string.
    """
    if count < 5:
        return "Circle or ellipse"
    else:
        return "Polygon"


diffAngles = np.abs(angleSp[1:] - angleSp[:-1])
critical_threshold = 25

criticle_pts = []
for i in range(len(diffAngles)):
    if diffAngles[i] >= critical_threshold:
        criticle_pts.append(fig1Points[i])

shape = identify_shape(len(criticle_pts))
print("Detected Shape:" , shape)

if criticle_pts and not np.array_equal(criticle_pts[0], criticle_pts[-1]):
    criticle_pts.append(criticle_pts[0])

criticle_pts = np.array(criticle_pts)


def process_and_plot(count1, centroid, fig1Points=None, criticle_pts=None):
    """
    Process the shape and plot based on the given data.

    :param count1: Shape identifier.
    :param centroid: Coordinates of the centroid for circle/ellipse.
    :param fig1Points: List of points for circle/ellipse.
    :param criticle_pts: List of critical points to plot.
    :return: A string message about the plot action taken.
    """
    shape_type = identify_shape(count1)
    global rad

    if shape_type == "Circle or ellipse" and fig1Points is not None:
        dis = []
        rad = 0
        for i in range(len(fig1Points) - 1):
            x1, y1 = centroid
            x2, y2 = fig1Points[i + 1]
            m = calculate_distances(x1, y1, x2, y2)
            dis.append(m)
        dis_array = np.array(dis)

        distdiff = []
        for i in range(len(dis_array) - 1):
            d = dis_array[i + 1] - dis_array[i]
            distdiff.append(d)

        if len(distdiff) > 0 and (sum(distdiff) / len(distdiff)) < 2:
            rad = sum(dis_array) / len(dis_array)
            fig, ax = plt.subplots()
            Drawcircle = plt.Circle(centroid, rad, fill=False, color='blue')
            ax.add_patch(Drawcircle)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(centroid[0] - rad, centroid[0] + rad)
            ax.set_ylim(centroid[1] - rad, centroid[1] + rad)

        else:
            return "Distances indicate that the shape is not a circle or ellipse."

        radius = rad

        angles = np.linspace(2 * np.pi, 0, 1000, endpoint=False)

        x_points = centroid[0] + radius * np.cos(angles)
        y_points = centroid[1] + radius * np.sin(angles)

        print(points)

        df = pd.DataFrame(points, columns=['X', 'Y'])

        df.to_csv('Answer.csv', index=False)
        df = pd.read_csv('Answer.csv')
        print(df)
        df = pd.read_csv('Answer.csv')

        x = df['X']
        y = df['Y']

        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of Answer.csv')
        plt.grid(True)
        plt.show()

    elif shape_type == "Polygon" and criticle_pts is not None:


        consecutive_distances = []
        for i in range(len(criticle_pts) - 1):
            point1 = criticle_pts[i]
            point2 = criticle_pts[i + 1]
            distance = np.linalg.norm(point2 - point1)
            consecutive_distances.append(distance)

        consecutive_distances_array = np.array(consecutive_distances)
        print(consecutive_distances_array)

        non_overlapping_points = []
        for i in range(len(criticle_pts) - 1):
            if consecutive_distances_array[i] > 10:
                non_overlapping_points.append(criticle_pts[i])

        non_overlapping_points_array = np.array(non_overlapping_points)
        print(non_overlapping_points_array)
        consecutive_distances1 = []
        for i in range(len(non_overlapping_points_array) - 1):
            point1 = non_overlapping_points_array[i]
            point2 = non_overlapping_points_array[i + 1]
            distance1 = np.linalg.norm(point2 - point1)
            consecutive_distances1.append(distance1)

        point1 = non_overlapping_points_array[-1]
        point2 = non_overlapping_points_array[0]
        distance1 = np.linalg.norm(point2 - point1)
        consecutive_distances1.append(distance1)

        consecutive_distances1_array = np.array(consecutive_distances1)
        print(consecutive_distances1_array)
        mean_distance = np.mean(consecutive_distances1_array)
        mean_distance

        def calculate_angle(A, B, C):
            """Calculates the angle ABC in degrees."""
            vector_AB = (B[0] - A[0], B[1] - A[1])
            vector_BC = (C[0] - B[0], C[1] - B[1])
            dot_product = vector_AB[0] * vector_BC[0] + vector_AB[1] * vector_BC[1]
            magnitude_AB = math.sqrt(vector_AB[0]*2 + vector_AB[1]*2)
            magnitude_BC = math.sqrt(vector_BC[0]*2 + vector_BC[1]*2)
            angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_BC))
            return math.degrees(angle_radians)

        anglesu = []

        for i in range(len(non_overlapping_points_array)):
            A = non_overlapping_points_array[i]
            B = non_overlapping_points_array[(i + 1) % len(non_overlapping_points_array)]
            C = non_overlapping_points_array[(i + 2) % len(non_overlapping_points_array)]
            angle = calculate_angle(A, B, C)
            anglesu.append(angle)

        print("Angles between all points:", anglesu)
        mean_angles = np.mean(anglesu)
        print(mean_angles)

        def adjust_coordinates(points_array, mean_distance, mean_angles):
            num_points = len(points_array)
            new_points = np.zeros_like(points_array)

            new_points[0] = points_array[0]

            if not isinstance(mean_angles, (list, np.ndarray)):
                mean_angles = np.array([mean_angles] * num_points)
            for i in range(1, num_points):

                cumulative_angle = np.sum(mean_angles[:i])  # Calculate cumulative angle up to current point
                angle_rad = np.deg2rad(cumulative_angle)

                new_x = new_points[i - 1][0] + mean_distance * np.cos(angle_rad)
                new_y = new_points[i - 1][1] + mean_distance * np.sin(angle_rad)

                new_points[i] = np.array([new_x, new_y])

            return new_points

        new_points = adjust_coordinates(non_overlapping_points_array, mean_distance, mean_angles)

        print("New adjusted coordinates:")
        print(new_points)
        plt.figure()
        plt.plot(new_points[:, 0], new_points[:, 1], 'o-', color='blue')
        plt.plot([new_points[-1, 0], new_points[0, 0]], [new_points[-1, 1], new_points[0, 1]], 'o-', color='blue')

        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title("Plot of Adjusted Points")
        plt.grid(True)
        plt.show()

        def append_first_as_last(array):
            """Appends the first element of the array to the end."""
            new_array = np.concatenate((array, [array[0]]), axis=0)
            return new_array

        new_points1 = append_first_as_last(new_points)
        print(new_points1)
        corners = new_points
        num_points = 200
        all_points = []

        for i in range(4):
            start_point = corners[i]
            end_point = corners[(i + 1) % 4]

            t = np.linspace(0, 1, num_points)
            line_points = start_point + t[:, np.newaxis] * (end_point - start_point)
            all_points.extend(line_points)

        all_points = np.array(all_points)
        plt.figure(figsize=(6, 6))
        plt.plot(all_points[:, 0], all_points[:, 1], marker='o', markersize=2, linestyle='-')
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title("Points adjusted")
        plt.grid(True)
        plt.show()

        points = all_points

        df = pd.DataFrame(all_points, columns=['X', 'Y'])

        df.to_csv('Answer.csv', index=False)
        x = df['X']
        y = df['Y']

        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of Answer.csv')
        plt.grid(True)
        plt.show()
        plt.plot(all_points[:, 0], all_points[:, 1], marker='o', linestyle='-')  # plot the points using matplotlib
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot symmetry')
        plt.grid(True)
        plt.show()
    else:
        return "Shape not recognized or insufficient data provided."

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def identify_shape(count):
    """
    Identify the shape based on count.

    :param count: An integer used to determine the shape.
    :return: Shape type as a string.
    """
    if count < 5:
        return "Circle or ellipse"
    else:
        return "Polygon"

def calculate_distances(x1, y1, x2, y2):
    """Calculate the distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(A, B, C):
    """Calculates the angle ABC in degrees."""
    vector_AB = (B[0] - A[0], B[1] - A[1])
    vector_BC = (C[0] - B[0], C[1] - B[1])
    dot_product = vector_AB[0] * vector_BC[0] + vector_AB[1] * vector_BC[1]
    magnitude_AB = math.sqrt(vector_AB[0]**2 + vector_AB[1]**2)
    magnitude_BC = math.sqrt(vector_BC[0]**2 + vector_BC[1]**2)
    angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_BC))
    return math.degrees(angle_radians)

def adjust_coordinates(points_array, mean_distance, mean_angles):
    num_points = len(points_array)
    new_points = np.zeros_like(points_array)
    new_points[0] = points_array[0]

    if not isinstance(mean_angles, (list, np.ndarray)):
        mean_angles = np.array([mean_angles] * num_points)

    for i in range(1, num_points):
        cumulative_angle = np.sum(mean_angles[:i])  # Calculate cumulative angle up to current point
        angle_rad = np.deg2rad(cumulative_angle)

        new_x = new_points[i-1][0] + mean_distance * np.cos(angle_rad)
        new_y = new_points[i-1][1] + mean_distance * np.sin(angle_rad)

        new_points[i] = np.array([new_x, new_y])

    return new_points

def process_and_plot(count1, centroid, fig1Points=None, criticle_pts=None):
    """
    Process the shape and plot based on the given data.

    :param count1: Shape identifier.
    :param centroid: Coordinates of the centroid for circle/ellipse.
    :param fig1Points: List of points for circle/ellipse.
    :param criticle_pts: List of critical points to plot.
    :return: A string message about the plot action taken.
    """
    shape_type = identify_shape(count1)
    global rad

    if shape_type == "Circle or ellipse" and fig1Points is not None:
        dis = []
        rad = 0
        for i in range(len(fig1Points) - 1):
            x1, y1 = centroid
            x2, y2 = fig1Points[i + 1]
            m = calculate_distances(x1, y1, x2, y2)
            dis.append(m)
        dis_array = np.array(dis)

        distdiff = []
        for i in range(len(dis_array) - 1):
            d = dis_array[i + 1] - dis_array[i]
            distdiff.append(d)

        if len(distdiff) > 0 and (sum(distdiff) / len(distdiff)) < 2:
            rad = sum(dis_array) / len(dis_array)
            fig, ax = plt.subplots()
            Drawcircle = plt.Circle(centroid, rad, fill=False, color='blue')
            ax.add_patch(Drawcircle)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(centroid[0] - rad, centroid[0] + rad)
            ax.set_ylim(centroid[1] - rad, centroid[1] + rad)

            radius = rad
            angles = np.linspace(2 * np.pi, 0, 1000, endpoint=False)
            x_points = centroid[0] + radius * np.cos(angles)
            y_points = centroid[1] + radius * np.sin(angles)

            points = np.column_stack((x_points, y_points))

            df = pd.DataFrame(points, columns=['X', 'Y'])
            df.to_csv('Answer.csv', index=False)
            df = pd.read_csv('Answer.csv')
            print(df)

            x = df['X']
            y = df['Y']

            plt.plot(x, y, marker='o', linestyle='-')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Plot of Answer.csv')
            plt.grid(True)
            plt.show()

        else:
            return "Distances indicate that the shape is not a circle or ellipse."

    elif shape_type == "Polygon" and criticle_pts is not None:
      if len(criticle_pts) >= 3:
            plt.figure()
            plt.plot([point[0] for point in criticle_pts], [point[1] for point in criticle_pts], 'o-', markersize=5)
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.title("Plot of Critical Points")
            plt.grid(True)
            plt.show()
            return "Critical points plotted successfully."
            consecutive_distances = []
            for i in range(len(criticle_pts) - 1):
              point1 = criticle_pts[i]
              point2 = criticle_pts[i + 1]
              distance = np.linalg.norm(point2 - point1)
              consecutive_distances.append(distance)
            consecutive_distances_array = np.array(consecutive_distances)
            print(consecutive_distances_array)
            non_overlapping_points = []
            for i in range(len(criticle_pts)-1):
              if consecutive_distances_array[i]>10:
                non_overlapping_points.append(criticle_pts[i])
            non_overlapping_points_array = np.array(non_overlapping_points)
            print(non_overlapping_points_array)
            consecutive_distances1 = []
            for i in range(len(non_overlapping_points_array) - 1):
              point1 = non_overlapping_points_array[i]
              point2 = non_overlapping_points_array[i + 1]
              distance1 = np.linalg.norm(point2 - point1)
              consecutive_distances1.append(distance1)
            point1 = non_overlapping_points_array[-1]
            point2 = non_overlapping_points_array[0]
            distance1 = np.linalg.norm(point2 - point1)
            consecutive_distances1.append(distance1)
            consecutive_distances1_array = np.array(consecutive_distances1)
            print(consecutive_distances1_array)
            mean_distance = np.mean(consecutive_distances1_array)
            mean_distance
            def calculate_angle(A, B, C):
              vector_AB = (B[0] - A[0], B[1] - A[1])
              vector_BC = (C[0] - B[0], C[1] - B[1])
              dot_product = vector_AB[0] * vector_BC[0] + vector_AB[1] * vector_BC[1]
              magnitude_AB = math.sqrt(vector_AB[0]**2 + vector_AB[1]**2)
              magnitude_BC = math.sqrt(vector_BC[0]**2 + vector_BC[1]**2)
              angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_BC))
              return math.degrees(angle_radians)
            anglesu = []
            for i in range(len(non_overlapping_points_array)):
              A = non_overlapping_points_array[i]
              B = non_overlapping_points_array[(i + 1) % len(non_overlapping_points_array)]
              C = non_overlapping_points_array[(i + 2) % len(non_overlapping_points_array)]
              angle = calculate_angle(A, B, C)
              anglesu.append(angle)
            print("Angles between all points:", anglesu)
            mean_angles = np.mean(anglesu)
            print(mean_angles)
            def adjust_coordinates(points_array, mean_distance, mean_angles):
              num_points = len(points_array)
              new_points = np.zeros_like(points_array)
              new_points[0] = points_array[0]
              if not isinstance(mean_angles, (list, np.ndarray)):
                mean_angles = np.array([mean_angles] * num_points)
              for i in range(1, num_points):
                cumulative_angle = np.sum(mean_angles[:i])
                angle_rad = np.deg2rad(cumulative_angle)
                new_x = new_points[i-1][0] + mean_distance * np.cos(angle_rad)
                new_y = new_points[i-1][1] + mean_distance * np.sin(angle_rad)
                new_points[i] = np.array([new_x, new_y])
              return new_points
            new_points = adjust_coordinates(non_overlapping_points_array, mean_distance, mean_angles)
            print("New adjusted coordinates:")
            print(new_points)
            plt.figure()
            plt.plot(new_points[:, 0], new_points[:, 1], 'o-', color='blue')
            plt.plot([new_points[-1, 0], new_points[0, 0]], [new_points[-1, 1], new_points[0, 1]], 'o-', color='blue')
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.title("Plot of Adjusted Points")
            plt.grid(True)
            plt.show()
            def append_first_as_last(array):
              new_array = np.concatenate((array, [array[0]]), axis=0)
              return new_array
            new_points1 = append_first_as_last(new_points)
            print(new_points1)
            corners = new_points
            num_points = 200
            points = []
            for i in range(4):
              start_point = corners[i]
              end_point = corners[(i + 1) % 4]
              t = np.linspace(0, 1, num_points)
              line_points = start_point + t[:, np.newaxis] * (end_point - start_point)
              points.extend(line_points)
            points = np.array(points)
            plt.figure(figsize=(6, 6))
            plt.plot(points[:, 0], points[:, 1], marker='o', markersize=2, linestyle='-')
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.title("Points adjusted")
            plt.grid(True)
            plt.show()
            points = points
            df = pd.DataFrame(points, columns=['X', 'Y'])
            df.to_csv('Answer.csv', index=False)
            df = pd.read_csv('Answer.csv')
            print(df)
            df = pd.read_csv('Answer.csv')
            x = df['X']
            y = df['Y']
            plt.plot(x, y, marker='o', linestyle='-')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Plot of Answer.csv')
            plt.grid(True)
            plt.show()
            plt.plot(points[:, 0], points[:, 1], marker='o', linestyle='-') #plot the points using matplotlib
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Plot symmetry')
            plt.grid(True)
            plt.show()





      else:
            return "Not enough critical points to plot a polygon."

    else:
        return "Shape not recognized or insufficient data provided."

result = process_and_plot(len(criticle_pts), centroid, fig1Points, criticle_pts)



import numpy as np
import matplotlib.pyplot as plt
points_array = np.array(points)


x = points_array[:, 0]
y = points_array[:, 1]


centroid_x = np.mean(x)
centroid_y = np.mean(y)

x_centered = x - centroid_x
y_centered = y - centroid_y

y_reflected = -y_centered
is_horizontally_symmetrical = np.allclose(np.sort(y_centered), np.sort(y_reflected))

x_reflected = -x_centered
is_vertically_symmetrical = np.allclose(np.sort(x_centered), np.sort(x_reflected))

rotated_x = (x_centered - y_centered) / np.sqrt(2)
rotated_y = (x_centered + y_centered) / np.sqrt(2)

rotated_y_reflected = -rotated_y
is_diagonally_symmetrical_45 = np.allclose(np.sort(rotated_y), np.sort(rotated_y_reflected))

rotated_x_anti = (x_centered + y_centered) / np.sqrt(2)
rotated_y_anti = (y_centered - x_centered) / np.sqrt(2)

rotated_y_anti_reflected = -rotated_y_anti
is_diagonally_symmetrical_anti_45 = np.allclose(np.sort(rotated_y_anti), np.sort(rotated_y_anti_reflected))

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

line_45_x = np.array([x_min, x_max])
line_45_y = centroid_y + (line_45_x - centroid_x)
line_anti_45_x = np.array([x_min, x_max])
line_anti_45_y = centroid_y - (line_anti_45_x - centroid_x)

print("Symmetrical: ", is_horizontally_symmetrical or is_vertically_symmetrical or is_diagonally_symmetrical_45 or is_diagonally_symmetrical_anti_45)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Shape Points')
plt.scatter(centroid_x, centroid_y, color='red', marker='*', s=100, label='Centroid')

if is_horizontally_symmetrical:
    plt.axhline(y=centroid_y, color='purple', linestyle='--', label='Horizontal Symmetry Line')
if is_vertically_symmetrical:
    plt.axvline(x=centroid_x, color='green', linestyle='--', label='Vertical Symmetry Line')
if is_diagonally_symmetrical_45:
    plt.plot(line_45_x, line_45_y, color='blue', linestyle='--', label='Diagonal Symmetry Line (45°)')
if is_diagonally_symmetrical_anti_45:
    plt.plot(line_anti_45_x, line_anti_45_y, color='orange', linestyle='--', label='Diagonal Symmetry Line (-45°)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Shape and Symmetry')
plt.legend()
plt.grid(True)
plt.show()