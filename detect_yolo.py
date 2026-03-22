import Metashape

from PySide2 import QtCore, QtWidgets
import pathlib, os, time, sys

import cv2
import numpy as np
import shutil
import csv
from shapely.errors import GEOSException
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import minimum_rotated_rectangle
from rtree import index
import math
import torch
import pandas as pd
from ultralytics import YOLO
from typing import List, Tuple, Union, Optional

DEVICES = []
print("sys.executable: ",sys.executable)
print("python version: ",sys.version)

if torch.backends.mps.is_available():
    device = "mps"
    print("\n✅ MPS (Metal) backend available — using Apple GPU!")
elif torch.cuda.is_available():
    device = "cuda"
    print("\n✅ CUDA available!")
    num_devices: int = torch.cuda.device_count()
    print(f"Number of devices: {num_devices}")

    for i in range(num_devices):
        DEVICES.append(i)
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    device = "cpu"
    print("\n⚠️ Using CPU — no hardware acceleration")


def get_utm_epsg_from_point(lon: float, lat: float) -> Optional[str]:
    if abs(lon) > 180 or abs(lat) > 90:
        return None

    zone = int((lon + 180) // 6) + 1
    base = 32600 if lat >= 0 else 32700
    return f"EPSG::{base + zone}"


def normalize_points(points_data):
    """
    Convert the input data to the numpy(N, 2) array.

    Supported formats:
      - [(x, y), (x, y), ...] → list of tuples/lists
      - [[x, y], [x, y], ...] → list of lists
      - [x1, y1, x2, y2, ...] → flat list

    Returns:
        np.ndarray of the form (N, 2), dtype=float64
    """
    if points_data is None:
        return np.empty((0, 2), dtype=np.float64)


    if isinstance(points_data, np.ndarray):
        if points_data.size == 0:
            return np.empty((0, 2), dtype=np.float64)
    else:

        try:
            if len(points_data) == 0:
                return np.empty((0, 2), dtype=np.float64)
        except TypeError:
            raise ValueError(f"Unsupported input data type: {type(points_data)}")

    first = points_data[0]

    if isinstance(first, (int, float, np.number)):
        n = len(points_data)
        if n % 2 != 0:
            raise ValueError(f"An odd number of coordinates ({n}) is a data error.")

        arr = np.asarray(points_data, dtype=np.float64)
        return arr.reshape(-1, 2)

    if hasattr(first, '__len__') and len(first) == 2:
        try:
            return np.array(points_data, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Couldn't convert to array (N, 2): {e}")


    raise ValueError(
        f"Unknown coordinate format. The first element is {first} (type {type(first)}). "
        "Expected: [(x, y), ...], [[x, y], ...] or [x, y, x, y, ...]"
    )


def convert_to_utm(input_data, source_crs, target_crs):
    points = normalize_points(input_data)
    if not points.any():
        return []

    lon0, lat0 = points[0]
    if abs(lon0) > 180 or abs(lat0) > 90:
        return [Metashape.Vector([x, y]) for x, y in points]

    vectors_list = [
        Metashape.CoordinateSystem.transform(
            Metashape.Vector([x, y]),
            source_crs,
            target_crs
        )
        for x, y in points
    ]
    return vectors_list


def vectors_to_bbox(vectors_data):
    """
    Converts the Metashape list.Vector in the bounding box [x_min, y_min, x_max, y_max].

    Args:
        vectors_data: list or array Vector or (x, y)

    Returns:
        [x_min, y_min, x_max, y_max] - list of float
    """
    if not vectors_data:
        raise ValueError("The list of vectors is empty")

    xs = []
    ys = []
    for v in vectors_data:
        if hasattr(v, 'x') and hasattr(v, 'y'):  # Metashape.Vector
            xs.append(v.x)
            ys.append(v.y)
        elif isinstance(v, (list, tuple)) and len(v) >= 2:  # (x, y)
            xs.append(v[0])
            ys.append(v[1])
        else:
            raise TypeError(f"Unsupported point type: {type(v)}")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return [x_min, y_min, x_max, y_max]


def pandas_append(df, row, ignore_index=False):
    """
    Append a row or DataFrame to an existing DataFrame with additional type handling
    and optional ignore_index functionality.

    This function facilitates appending various types of inputs (`pd.DataFrame`,
    `pd.Series`, `dict`) to an existing DataFrame with specific handling for
    empty DataFrames and `NaN` entries. It enables flexibility in merging data
    while maintaining the order of columns in the resulting DataFrame.

    :param df: The original DataFrame to which the row or DataFrame will be appended.
    :type df: pandas.DataFrame
    :param row: The row or DataFrame to append. It can be a pandas DataFrame, pandas
                Series, or dictionary.
    :type row: Union[pandas.DataFrame, pandas.Series, dict]
    :param ignore_index: Whether to ignore the index during concatenation. Defaults to False.
    :type ignore_index: bool
    :return: A new DataFrame with the `row` or DataFrame appended to the `df`.
    :rtype: pandas.DataFrame
    """

    if isinstance(row, pd.DataFrame):
        if not df.empty and not row.isna().all().all():  # Additional check for NA
            result = pd.concat([df, row], ignore_index=True)
        else:
            result = row if df.empty else df

    elif isinstance(row, pd.core.series.Series):
        result = pd.concat([df, row.to_frame().T], ignore_index=ignore_index)

    elif isinstance(row, dict):
        result = pd.concat([df, pd.DataFrame(row, index=[0], columns=df.columns)])

    else:
        raise RuntimeError("pandas_append: unsupported row type - {}".format(type(row)))

    return result


def getShapeVertices(shape):
    """
    Gets the vertices of the given shape.

    This function computes and returns a list of vertices for the specified shape. It retrieves marker positions for attached
    shapes or directly uses coordinate values for detached shapes. Transformations are applied to convert marker positions
    to the desired coordinate system when working with attached shapes.

    Parameters:
    shape: Metashape.Shape
        The shape from which vertices are to be extracted. The shape can either be attached or detached.

    Returns:
    list
        A list of vertex points representing the shape's geometry. The points are either transformed marker positions
        (for attached shapes) or coordinate values directly extracted from the shape (for detached shapes).

    Raises:
    Exception
        If the active chunk is null.
    Exception
        If any marker position is invalid within the given shape.
    """
    chunk = Metashape.app.document.chunk
    if chunk is None:
        raise Exception("Null chunk")

    T = chunk.transform.matrix
    result = []

    if shape.is_attached:
        assert (len(shape.geometry.coordinates) == 1)
        for key in shape.geometry.coordinates[0]:
            for marker in chunk.markers:
                if marker.key == key:
                    if not marker.position:
                        raise Exception("Invalid shape vertex")

                    point = T.mulp(marker.position)
                    point = Metashape.CoordinateSystem.transform(point, chunk.world_crs, chunk.shapes.crs)
                    result.append(point)
    else:
        assert (len(shape.geometry.coordinates) == 1)
        for coord in shape.geometry.coordinates[0]:
            result.append(coord)

    return result


def ensure_unique_directory(base_dir):
    """
    Generates a unique directory name by appending a numeric suffix to the provided base directory
    name if a directory with the same base name already exists. If the base directory does not exist,
    it is returned unchanged.

    Args:
        base_dir (str): The base directory name to check and ensure uniqueness for.

    Returns:
        str: A unique directory name. If the base directory does not exist, the same directory name is
        returned. If it does exist, a unique name with an appended numeric suffix is returned.
    """
    if not os.path.exists(base_dir):
        return base_dir

    counter = 1
    new_dir = f"{base_dir}_{counter}"  # Form initial name (with suffix `1`)
    while os.path.exists(new_dir):  # While directory exists, increment counter
        counter += 1
        new_dir = f"{base_dir}_{counter}"

    return new_dir


def remove_directory(directory_path):
    """
    Removes the specified directory and its contents if it exists. If the directory
    doesn't exist, the function does nothing.

    Args:
        directory_path (str): The path to the directory to remove.

    Raises:
        Exception: If an unexpected error occurs while attempting to remove the
        directory.
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    else:
        pass


def merge_contours_shapely(contours, iou_threshold=0.7):
    """
    Merges contours based on Intersection over Union (IoU) threshold using Shapely library.

    This function converts contours to polygons and iteratively merges them if their IoU
    (relative to the minimum area) exceeds the specified threshold. It handles invalid
    geometries and tracks original contour indices for each merged result.

    Parameters:
        contours (list): List of contours to merge. Each contour is a list of coordinate points.
        iou_threshold (float): IoU threshold for merging (default: 0.7). Polygons with
            intersection_area / min_area >= iou_threshold will be merged.

    Returns:
        tuple: A tuple containing:
            - result_contours (list): List of merged contours as numpy arrays
            - result_indices (list): List of lists, where each sublist contains the original
              indices of contours that were merged together
    """
    # Convert input contours to polygons
    polygons = []
    valid_indices = []  # Track which original indices are valid
    for i_c, contour in enumerate(contours):
        try:
            poly = Polygon(contour).buffer(0)  # Fix geometry using buffer(0)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
                valid_indices.append(i_c)
            else:
                print(f"Skipping invalid geometry: {contour}")
        except GEOSException as e:
            print(f"Error creating polygon: {e}, skipping this geometry.")

    # Track which original indices belong to each polygon
    polygon_indices = [[idx] for idx in valid_indices]

    # Iterative merging
    while len(polygons) > 1:  # While there is more than one polygon, try to merge
        has_changes = False
        merged_polygons = []
        merged_indices = []
        visited = [False] * len(polygons)  # Array to mark merged polygons

        # Index polygons for optimal search
        idx = index.Index()
        for pos, poly in enumerate(polygons):
            if not poly.is_empty and poly.is_valid:  # Skip empty/invalid polygons
                idx.insert(pos, poly.bounds)

        for pos, poly in enumerate(polygons):
            if visited[pos] or poly.is_empty or not poly.is_valid:  # If already processed or polygon is invalid
                continue

            merge_queue = [pos]
            candidate_merged_poly = poly
            candidate_merged_indices = polygon_indices[pos].copy()
            visited[pos] = True

            # Iterate through all neighboring polygons
            while merge_queue:
                current_idx = merge_queue.pop()
                current_poly = polygons[current_idx]

                for merge_index in idx.intersection(current_poly.bounds):  # Check intersecting polygons
                    if visited[merge_index] or merge_index == current_idx:
                        continue

                    neighbor_poly = polygons[merge_index]
                    try:
                        # Check intersection and calculate IoU (relative to minimum area)
                        if current_poly.intersects(neighbor_poly):
                            intersection_area = current_poly.intersection(neighbor_poly).area
                            min_area = min(current_poly.area, neighbor_poly.area)

                            if intersection_area / min_area >= iou_threshold:
                                # Merge current polygon with neighbor
                                candidate_merged_poly = unary_union([candidate_merged_poly, neighbor_poly]).buffer(0)

                                if candidate_merged_poly.is_valid and not candidate_merged_poly.is_empty:
                                    merge_queue.append(merge_index)
                                    visited[merge_index] = True
                                    candidate_merged_indices.extend(polygon_indices[merge_index])
                                    has_changes = True
                    except GEOSException as e:
                        print(f"Intersection check failed: {e}")
                        continue

            # Add merged polygon to list
            if candidate_merged_poly.is_valid and not candidate_merged_poly.is_empty:
                merged_polygons.append(candidate_merged_poly)
                merged_indices.append(candidate_merged_indices)

        # If nothing changed during iteration, merging is complete
        if not has_changes:
            break

        # Update polygon list after merging
        polygons = merged_polygons
        polygon_indices = merged_indices

    # Convert polygons back to contours
    result_contours = []
    result_indices = []
    for i_p, poly in enumerate(polygons):
        if isinstance(poly, Polygon):
            result_contours.append(np.array(poly.exterior.coords))
            result_indices.append(polygon_indices[i_p])
        elif isinstance(poly, MultiPolygon):
            multi_poly_contours = []
            for sub_poly in poly.geoms:
                if sub_poly.is_valid and not sub_poly.is_empty:  # Skip invalid or empty
                    multi_poly_contours.append(np.array(sub_poly.exterior.coords))

            # Calculate area of each contour
            areas = []
            for contour in multi_poly_contours:
                if len(contour) >= 4:  # Ensure contour has at least 4 points
                    areas.append(cv2.contourArea(np.array(contour, dtype=np.float32)))

            # Find index of contour with largest area
            if areas:
                largest_contour_index = np.argmax(areas)
                # Select contour with largest area
                largest_contour = multi_poly_contours[largest_contour_index]
                result_contours.append(largest_contour)
                result_indices.append(polygon_indices[i_p])

    return result_contours, result_indices


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) for two boxes: box1 and box2.

    Parameters:
        box1 (list): Coordinates of the first box [x1, y1, x2, y2].
        box2 (list): Coordinates of the second box [x1, y1, x2, y2].

    Returns:
        float: IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union_area = box1_area + box2_area - intersection_area

    # IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou


def is_inside(box1, box2, threshold=0.9):
    """
    Checks if box1 is inside box2 with a given overlap percentage.

    Parameters:
        box1 (list): Coordinates of the first box [x1, y1, x2, y2].
        box2 (list): Coordinates of the second box [x1, y1, x2, y2].
        threshold (float): Overlap threshold (from 0 to 1), 1.0 for complete intersection, 0.9 for 90%.

    Returns:
        bool: True if box1 is inside box2 by 'threshold', otherwise False.
    """
    # Calculate intersection coordinates
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

    # Calculate area of first box (box1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    # Check overlap condition
    return (intersect_area / box1_area) >= threshold


def merge_boxes(box1, box2):
    """
    Merges two boxes into one that encompasses both.

    Parameters:
        box1 (list): Coordinates of the first box [x1, y1, x2, y2].
        box2 (list): Coordinates of the second box [x1, y1, x2, y2].

    Returns:
        list: Coordinates of the merged box [x1, y1, x2, y2].
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]


def merge_all_boxes(boxes, iou_threshold=0.9):
    """
    Merges all intersecting or fully nested boxes until no more merging is possible.

    Parameters:
        boxes (list): List of boxes to merge, each box is [x1, y1, x2, y2].
        iou_threshold (float): IoU threshold for merging (default: 0.9).

    Returns:
        tuple: A tuple containing:
            - merged_boxes (list): List of merged boxes
            - box_indices (list): List of lists, where each sublist contains the original
              indices of boxes that were merged together
    """
    merged_boxes = boxes.copy()  # Create copy of original list
    # Track which original indices belong to each box
    box_indices = [[i_b] for i_b in range(len(boxes))]

    while True:
        new_boxes = []
        new_indices = []
        skip_indices = set()  # Indices that were added/merged in this iteration
        merged = False  # Flag to control whether merging occurred

        for i_mb in range(len(merged_boxes)):
            if i_mb in skip_indices:
                continue  # Skip already processed boxes

            current_box = merged_boxes[i_mb]  # Current box we're trying to merge
            current_indices = box_indices[i_mb].copy()

            for j_mb in range(len(merged_boxes)):
                if i_mb == j_mb or j_mb in skip_indices:
                    continue  # Skip itself or already processed box

                # Check merging conditions:
                iou = calculate_iou(current_box, merged_boxes[j_mb])
                if (
                        iou >= iou_threshold or
                        is_inside(current_box, merged_boxes[j_mb], iou_threshold) or
                        is_inside(merged_boxes[j_mb], current_box, iou_threshold)
                ):
                    # Merge two boxes
                    current_box = merge_boxes(current_box, merged_boxes[j_mb])
                    current_indices.extend(box_indices[j_mb])
                    skip_indices.add(j_mb)  # Mark that this box was processed
                    merged = True  # Indicate that merging occurred

            # Save merged box
            new_boxes.append(current_box)
            new_indices.append(current_indices)
            skip_indices.add(i_mb)

        # If no merging occurred, finish
        if not merged:
            break

        # Update box list for next iteration
        merged_boxes = new_boxes
        box_indices = new_indices

    return merged_boxes, box_indices


def group_shapes_by_label(input_data) -> dict:
    """
    Groups shape data by their labels.

    Parameters:
        input_data (list): List of dictionaries, each containing shape data with a 'label' key.

    Returns:
        dict: Dictionary where keys are labels and values are lists of shape data
              belonging to that label.
    """
    groups = {}
    for d in input_data:
        label = d['label']
        if label not in groups:
            groups[label] = []
        groups[label].append(d)

    return groups


def extract_coords_metadata_shapes(input_shapes) -> list:
    """
    Extracts coordinates and metadata from shape objects or processes existing dictionaries.

    This function handles two types of input:
    1. Shape objects with geometry attributes - extracts coordinates, label, and score
    2. Dictionaries already containing coordinates, label, and score - returns as is

    Parameters:
        input_shapes (list): List of shape objects or dictionaries containing shape data.

    Returns:
        list: List of dictionaries, each containing:
              - 'coords': List of coordinate points
              - 'label': Shape label/class name
              - 'score': Confidence score (0.0 if not available)
    """
    shapes_data = []

    if input_shapes:
        if hasattr(input_shapes[0], 'geometry'):
            # Extract coordinates and metadata from shape objects
            for shape in input_shapes:
                coords = np.array(getShapeVertices(shape)).tolist()
                label = shape.label
                try:
                    score = float(shape.attributes["Score"])
                except (KeyError, ValueError):
                    score = 0.0
                shapes_data.append({
                    'coords': coords,
                    'label': label,
                    'score': score
                })
        else:
            # Already dictionaries with coordinates, label, and score
            shapes_data = input_shapes

    return shapes_data


class MainWindowDetect(QtWidgets.QDialog):

    def __init__(self, parent):
        """
        Initializes the YOLO object detection dialog window.

        Sets up the detection parameters, loads settings from the application configuration,
        initializes the GUI, and prepares the working environment for object detection on
        orthomosaic images.

        Parameters:
            parent: Parent widget for this dialog.

        Instance Variables:
            stopped (bool): Flag to control process interruption.
            model: YOLO model instance (initialized later).
            classes: List of detection class names.
            force_small_patch_size (bool): Whether to force small patch size.
            working_dir (str): Directory for storing detection results.
            load_model_path (str): Path to the YOLO model file.
            max_image_size (int): Maximum image size for processing (default: 640).
            isDebugMode (bool): Whether debug mode is enabled.
            detection_score_threshold (float): Confidence threshold for detections (default: 0.90).
            iou_threshold (float): IoU threshold for merging overlapping detections (default: 0.6).
            preferred_patch_size (int): Preferred size for image patches (default: 640 pixels).
            preferred_resolution (float): Preferred resolution in meters/pixel (default: 0.005 m/pix).
        """
        QtWidgets.QDialog.__init__(self, parent)

        self.TARGET_UTM_CRS = None

        self.stopped = False
        self.model = None
        self.classes = None
        self.force_small_patch_size = True
        self.expected_layer_name_train_zones = "zone"
        self.expected_layer_name_train_data = "data"
        self.layer_name_detection_data = ""

        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent / "objects_detection")
        else:
            self.working_dir = ""

        self.load_model_path = self.read_model_path_from_settings()
        max_image_size = Metashape.app.settings.value("scripts/yolo/max_image_size")
        self.max_image_size = int(max_image_size) if max_image_size else 640
        self.cleanup_working_dir = False
        self.isDebugMode = False
        self.detect_on_user_layer_enabled = False
        self.preferred_patch_size = 640  # 640 pixels
        self.preferred_resolution = 0.005  # 0.5 cm/pix

        # Load detection_score_threshold from settings
        score_threshold = Metashape.app.settings.value("scripts/yolo/score-threshold")
        self.detection_score_threshold = float(score_threshold) if score_threshold else 0.90

        # Load iou_threshold from settings
        iou_threshold = Metashape.app.settings.value("scripts/yolo/iou-threshold")
        self.iou_threshold = float(iou_threshold) if iou_threshold else 0.6

        self.prefer_original_resolution = False

        self.setWindowTitle("YOLO objects detection on orthomosaic")
        self.chunk = Metashape.app.document.chunk

        self.create_gui()
        self.exec()

    def stop(self):
        """
        Stops the running process by setting the stopped state to True.

        This method updates the state of the `stopped` attribute, effectively causing any
        process depending on this attribute to halt its execution. It provides a mechanism
        to control and terminate activities gracefully.

        Attributes:
            - stopped (bool): Represents whether the process is stopped. Initially set to
                            False and updated to True when this method is called.
        """
        self.stopped = True

    def check_stopped(self):
        """
        Checks whether the 'stopped' attribute is True and raises an exception if it is.

        This method is used to determine if a stop signal has been triggered. If the
        'stopped' attribute evaluates to True, it raises an InterruptedError, which
        signals that the operation should be halted.

        Raises:
            InterruptedError: If the 'stopped' attribute is True, indicating a
            stop request.
        """
        if self.stopped:
            raise InterruptedError("Stop was pressed")

    def show_question_dialog(self, title, text, informative_text, buttons, default_button_index=0):
        """
        Displays a universal question dialog with custom buttons.

        This method creates and displays a QMessageBox with a question icon, allowing
        the user to choose between multiple options. It's a reusable utility method
        for showing dialogs throughout the application.

        Parameters:
            title (str): The window title of the message box.
            text (str): The main text displayed in the message box.
            informative_text (str): Additional informative text displayed below the main text.
            buttons (list of tuple): A list of tuples where each tuple contains:
                - button_text (str): The text to display on the button
                - button_role (QtWidgets.QMessageBox.ButtonRole): The role of the button
            default_button_index (int, optional): The index of the button to set as default.
                Defaults to 0 (first button).

        Returns:
            int: The index of the clicked button in the buttons list.

        Example:
             buttons = [
            ... ("Use Existing", QtWidgets.QMessageBox.YesRole),
            ... ("Create New", QtWidgets.QMessageBox.NoRole)
            ... ]
             choice = self.show_question_dialog(
            ... "Title", "Main text", "Info text", buttons, default_button_index=0
            ... )
             if choice == 0:
            ... print("User chose first option")
        """
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Question)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setInformativeText(informative_text)

        # Add all buttons and store references
        button_widgets = []
        for button_text, button_role in buttons:
            btn = msg_box.addButton(button_text, button_role)
            button_widgets.append(btn)

        # Set default button
        if 0 <= default_button_index < len(button_widgets):
            msg_box.setDefaultButton(button_widgets[default_button_index])

        # Execute dialog
        msg_box.exec_()

        # Find which button was clicked and return its index
        clicked_button = msg_box.clickedButton()
        for i_bw, btn in enumerate(button_widgets):
            if btn == clicked_button:
                return i_bw

        # Should not reach here, but return 0 as fallback
        return 0

    def run_detect(self):
        """
        Runs the detection process and controls the workflow of specific tasks such as loading
        parameters, preparing the environment, creating a neural network, and exporting results.

        This method also tracks processing time, manages user interactions via enabling/disabling
        appropriate UI buttons, and handles post-processing cleanup. The method uses unified
        detection that processes either selected zones or the entire orthomosaic based on user choice.

        """
        try:
            self.stopped = False
            self.btnDetect.setEnabled(False)
            self.btnStop.setEnabled(True)

            time_start = time.time()

            self.load_params()
            self.prepare()

            print("Script started...")

            self.create_neural_network()
            self.export_orthomosaic()

            if self.chunk.shapes is None:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            print(f"Source CRS: {self.chunk.orthomosaic.crs}")
            target_crs = get_utm_epsg_from_point(self.chunk.orthomosaic.left, self.chunk.orthomosaic.top)
            if target_crs:
                self.TARGET_UTM_CRS = Metashape.CoordinateSystem(target_crs)

            # Determine whether to use zones or entire orthomosaic
            detectZonesLayer = self.layers[self.detectZonesLayer.currentIndex()]
            use_zones = (detectZonesLayer != self.noDataChoice)

            # Call unified detection method
            self.detect_unified(use_zones=use_zones)

            results_time_total = time.time() - time_start
            self.show_results_dialog(results_time_total)

        except Exception as ex:
            if self.stopped:
                Metashape.app.messageBox("Processing was stopped.")
            else:
                Metashape.app.messageBox("Something gone wrong.\n"
                                         "Please check the console.")

                raise ex
        finally:
            if self.cleanup_working_dir:
                # shutil.rmtree(self.working_dir, ignore_errors=True)
                pass
            self.reject()

        print("Script finished.")
        return True

    def prepare(self):
        """
        Prepares the working directory and its subdirectories for the project. Ensures the necessary
        directory structure is created and sets configurations for multiprocessing if the operating system
        is Windows.

        Raises
        ------
        Exception
            If the working directory is not specified.
        """

        import multiprocessing as mp

        if self.working_dir == "":
            raise Exception("You should specify working directory (or save .psx project)")

        os.makedirs(self.working_dir, exist_ok=True)
        print("Working dir: {}".format(self.working_dir))

        self.dir_tiles = self.working_dir + "/tiles/"
        self.dir_detection_results = self.working_dir + "/detection/"
        self.dir_subtiles_results = self.dir_detection_results + "inner/"

        for subdir in [self.dir_tiles, self.dir_detection_results, self.dir_subtiles_results]:
            os.makedirs(subdir, exist_ok=True)

        if os.name == 'nt':  # if Windows
            mp.set_executable(os.path.join(sys.exec_prefix, 'python.exe'))
            print(f"multiprocessing set_executable: {os.path.join(sys.exec_prefix, 'python.exe')}")

    def create_neural_network(self):
        """
        Loads and initializes a neural network model using the Ultralytics YOLO framework. This function
        either loads a neural network from a pre-specified path or raises an error in the
        absence of a specified model.

        Parameters: None

        Raises:
            FileExistsError: If no model path is specified upon invocation.
        """
        print("Neural network loading...")

        if self.load_model_path:
            print("Using the neural network loaded from '{}'...".format(self.load_model_path))
            Metashape.app.settings.setValue("scripts/yolo/model_load_path", self.load_model_path)

            self.model = YOLO(self.load_model_path)
            self.classes = self.model.names

        else:
            self.model = YOLO("yolo11x-seg.pt")
            self.classes = self.model.names
            print("Using the neural network loaded defaults `yolo11x-seg.pt`")
            # raise FileExistsError("No neural network was specified")

    def export_orthomosaic(self):
        """
        Prepares and exports an orthomosaic image to a set of tiles, then processes
        the tiles to map their paths and spatial transformation data. It ensures
        the tiles are stored with proper metadata required for further processing.
        Handles resolution preferences and manages data accordingly.

        Raises:
            Exception: If no tiles are found in the specified directory.

        """

        print("Preparing orthomosaic...")

        kwargs = {}
        if not self.prefer_original_resolution:  # and (self.chunk.orthomosaic.resolution < self.preferred_resolution * 0.90):
            kwargs["resolution"] = self.preferred_resolution
        else:
            print("no resolution downscaling required")

        tiles = os.listdir(self.dir_tiles)
        if tiles:
            # Tiles already exist, ask user what to do
            buttons = [
                ("Use Existing Tiles", QtWidgets.QMessageBox.YesRole),
                ("Delete and Create New", QtWidgets.QMessageBox.NoRole)
            ]
            choice = self.show_question_dialog(
                title="Existing Tiles Found",
                text="Tiles already exist in the directory.",
                informative_text="Do you want to use existing tiles or delete them and create new ones?",
                buttons=buttons,
                default_button_index=0
            )

            if choice == 1:  # "Delete and Create New" was clicked
                # Delete existing tiles
                print("Deleting existing tiles...")
                for tile in tiles:
                    tile_path = os.path.join(self.dir_tiles, tile)
                    try:
                        os.remove(tile_path)
                    except Exception as e:
                        print(f"Error deleting tile {tile}: {e}")

                # Create new tiles
                self.chunk.exportRaster(path=self.dir_tiles + "tile.jpg",
                                        source_data=Metashape.OrthomosaicData,
                                        image_format=Metashape.ImageFormat.ImageFormatJPEG,
                                        save_alpha=False,
                                        white_background=True,
                                        save_world=True,
                                        split_in_blocks=True,
                                        block_width=self.patch_size,
                                        block_height=self.patch_size,
                                        **kwargs)
            else:
                print("Using existing tiles...")
        else:
            # No tiles exist, create them
            self.chunk.exportRaster(path=self.dir_tiles + "tile.jpg",
                                    source_data=Metashape.OrthomosaicData,
                                    image_format=Metashape.ImageFormat.ImageFormatJPEG,
                                    save_alpha=False,
                                    white_background=True,
                                    save_world=True,
                                    split_in_blocks=True,
                                    block_width=self.patch_size,
                                    block_height=self.patch_size,
                                    **kwargs)

        tiles = os.listdir(self.dir_tiles)
        if not tiles:
            raise Exception("No tiles found in the directory.")

        app = QtWidgets.QApplication.instance()

        self.tiles_paths = {}
        self.tiles_to_world = {}

        # Предварительно группируем тайлы по координатам
        tile_dict = {}

        for i_t, tile in enumerate(tiles):
            if not tile.startswith("tile-"):
                continue

            # Извлекаем координаты один раз
            base_name = tile.split(".")[0]  # "tile-123-456"
            parts = base_name.split("-")
            if len(parts) != 3:
                continue  # защита от некорректных имён
            _, tile_x_str, tile_y_str = parts
            try:
                tile_x, tile_y = int(tile_x_str), int(tile_y_str)
            except ValueError:
                continue  # некорректные координаты

            key = (tile_x, tile_y)

            if key not in tile_dict:
                tile_dict[key] = {"jpg": None, "world": None}

            if tile.endswith((".jpg",)):
                tile_dict[key]["jpg"] = os.path.join(self.dir_tiles, tile)
            elif tile.endswith((".jgw", ".pgw")):
                tile_dict[key]["world"] = os.path.join(self.dir_tiles, tile)

            self.detectionPBar.setValue(int((100 * i_t + 1) / len(tiles)))
            self.check_stopped()
            app.processEvents()

        # Теперь обрабатываем только полные пары
        i_item = 1
        for (tile_x, tile_y), files in tile_dict.items():
            jpg_path = files["jpg"]
            world_path = files["world"]

            if jpg_path is None or world_path is None:
                continue  # пропускаем неполные пары

            self.tiles_paths[(tile_x, tile_y)] = jpg_path

            with open(world_path, "r") as f:
                matrix2x3 = np.array([float(line.strip()) for line in f], dtype=np.float64).reshape(3, 2).T
            self.tiles_to_world[(tile_x, tile_y)] = matrix2x3

            self.detectionPBar.setValue(int((100 * i_item + 1) / len(tile_dict.items())))
            self.check_stopped()
            app.processEvents()
            i_item += 1

        # Verification
        assert len(self.tiles_paths) == len(self.tiles_to_world)
        assert set(self.tiles_paths.keys()) == set(self.tiles_to_world.keys())

        # Calculate boundaries in one pass
        xs, ys = zip(*self.tiles_paths.keys())  # unpack coordinates
        self.tile_min_x, self.tile_max_x = min(xs), max(xs)
        self.tile_min_y, self.tile_max_y = min(ys), max(ys)

        print(f"{len(self.tiles_paths)} tiles, tile_x in [{self.tile_min_x}; {self.tile_max_x}], "
              f"tile_y in [{self.tile_min_y}; {self.tile_max_y}]")

    def read_part(self, res_from, res_to):
        """
        Reads a specified region of an image by combining multiple tiles into a single output.

        This method processes a specified rectangular region of an image, defined by the
        coordinates `res_from` and `res_to`, by stitching together multiple smaller tiles.
        The output is a complete section of the image with padding applied where necessary
        to fill missing parts.

        Parameters:
            res_from (numpy.ndarray): A 2-element array specifying the top-left corner
                (x, y) of the region to be read.
            res_to (numpy.ndarray): A 2-element array specifying the bottom-right corner
                (x, y) of the region to be read.

        Returns:
            numpy.ndarray: A 3D array representing the specified region of the image in RGB
                format, with missing areas filled with white.

        Raises:
            AssertionError: If the size of the requested region is smaller than the patch
                size or if an invalid region range is specified.
        """

        res_size = res_to - res_from
        assert np.all(res_size >= [self.patch_size, self.patch_size])
        res = np.zeros((res_size[1], res_size[0], 3), np.uint8)
        res[:, :, :] = 255

        tile_xy_from = np.int32(res_from // self.patch_size)
        tile_xy_upto = np.int32((res_to - 1) // self.patch_size)
        assert np.all(tile_xy_from <= tile_xy_upto)
        for tile_x in range(tile_xy_from[0], tile_xy_upto[0] + 1):
            for tile_y in range(tile_xy_from[1], tile_xy_upto[1] + 1):
                if (tile_x, tile_y) not in self.tiles_paths:
                    continue
                part = cv2.imread(self.tiles_paths[tile_x, tile_y])
                part = cv2.copyMakeBorder(part, 0, self.patch_size - part.shape[0], 0, self.patch_size - part.shape[1],
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
                part_from = np.int32([tile_x, tile_y]) * self.patch_size - res_from
                part_to = part_from + self.patch_size

                res_inner_from = np.int32([max(0, part_from[0]), max(0, part_from[1])])
                res_inner_to = np.int32([min(part_to[0], res_size[0]), min(part_to[1], res_size[1])])

                part_inner_from = res_inner_from - part_from
                part_inner_to = part_inner_from + res_inner_to - res_inner_from

                res[res_inner_from[1]:res_inner_to[1], res_inner_from[0]:res_inner_to[0], :] = part[part_inner_from[1]:
                                                                                                    part_inner_to[1],
                part_inner_from[0]:
                part_inner_to[0], :]

        return res

    @staticmethod
    def add_pixel_shift(to_world, dx, dy):
        """
        Adds a pixel shift to a given transformation matrix by updating its
        translation components based on the specified shifts in the x and y
        directions. This operation modifies only the translation components
        of the matrix while keeping other transformations intact.

        Args:
            to_world (numpy.ndarray): The transformation matrix to modify.
                Expected to be a 3x3 matrix representing an affine transformation
                in homogeneous coordinates.
            dx (float): The horizontal pixel shift to apply to the transformation.
            dy (float): The vertical pixel shift to apply to the transformation.

        Returns:
            numpy.ndarray: A new transformation matrix with the updated translation
            components that include the specified pixel shifts.
        """
        to_world = to_world.copy()
        to_world[0, 2] = to_world[0, :] @ [dx, dy, 1]
        to_world[1, 2] = to_world[1, :] @ [dx, dy, 1]
        return to_world

    @staticmethod
    def invert_matrix_2x3(to_world):
        """
        Invert a 2x3 transformation matrix.

        This static method takes a 2x3 transformation matrix and calculates its inverse.
        The input matrix is extended into a 3x3 matrix by appending a row of [0, 0, 1]
        to make it suitable for inversion. The result is a 2x3 matrix obtained by
        removing the last row after inversion.

        Parameters:
            to_world: numpy.ndarray
                A 2x3 transformation matrix to be inverted.

        Returns:
            numpy.ndarray
                The inverted 2x3 transformation matrix.

        Raises:
            AssertionError
                If the extended 3x3 matrix does not fulfill the expected
                constraints after inversion.
        """

        to_world33 = np.vstack([to_world, [0, 0, 1]])
        from_world = np.linalg.inv(to_world33)

        assert (from_world[2, 0] == from_world[2, 1] == 0)
        assert (from_world[2, 2] == 1)
        from_world = from_world[:2, :]

        return from_world

    @staticmethod
    def read_model_path_from_settings():
        """
        Reads the model path from the application settings.

        This static method retrieves the model load path specified in the application's settings.
        If no path is found, it defaults to an empty string.

        Returns:
            str: The model load path from the application's settings or an empty string if no path is set.
        """
        load_path = Metashape.app.settings.value("scripts/yolo/model_load_path")
        if load_path is None:
            load_path = ""
        return load_path

    def _get_debug_save_flags(self):
        """
        Returns save flags for debug mode.

        Returns:
            tuple: (save, save_conf, save_txt, save_crop)
        """
        if self.isDebugMode:
            return True, True, True, True
        return False, False, False, False

    def _run_prediction(self, subtile):
        """
        Executes YOLO prediction on a subtile.

        Parameters:
            subtile: Image tile for prediction

        Returns:
            Prediction result from the model
        """
        save, save_conf, save_txt, save_crop = self._get_debug_save_flags()

        with torch.no_grad():
            prediction = self.model.predict(
                subtile,
                imgsz=640,
                device=DEVICES if device == "cuda" else device,
                conf=self.detection_score_threshold,
                iou=0.45,
                project=self.dir_subtiles_results,
                save=save,
                save_conf=save_conf,
                save_txt=save_txt,
                save_crop=save_crop,
                half=True
            )

        return prediction[0].cpu()

    def _process_mask_contour(self, mask, orig_h, orig_w, x_offset, y_offset):
        """
        Processes mask data and extracts the largest contour with coordinate offset.

        Parameters:
            mask: Normalized mask coordinates
            orig_h: Original image height
            orig_w: Original image width
            x_offset: X coordinate offset to apply
            y_offset: Y coordinate offset to apply

        Returns:
            list: Shifted contour points or empty list if no valid contour
        """
        # Convert normalized coordinates to pixels
        pixel_coords = np.array([[int(y * orig_h), int(x * orig_w)] for y, x in mask], dtype=np.int32)

        # Create binary mask
        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [pixel_coords], color=255)

        # Get contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find contour with largest area
            areas = [cv2.contourArea(contour) for contour in contours]
            largest_contour_index = np.argmax(areas)
            largest_contour = contours[largest_contour_index]

            # Shift contour point coordinates
            shifted_contour = [[x + x_offset, y + y_offset] for [[x, y]] in largest_contour]
            return shifted_contour

        return []

    def _compute_box_coordinates(self, data, to_world_key='zone_to_world', use_offset=True):
        """
        Computes box coordinates from prediction data without creating shapes.

        Parameters:
            data: DataFrame with prediction data
            to_world_key: Key/attribute name for transformation matrix
            use_offset: Whether to use 0.5 pixel offset (False for zones, True for tiles)

        Returns:
            list: List of dictionaries containing:
                - 'coords': coordinate lists (corners) for each box
                - 'label': class label
                - 'score': confidence score
        """
        result = []
        for row in data.itertuples():
            xmin, ymin, xmax, ymax = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
            label, score = row.label, row.score

            # Get transformation matrix
            if to_world_key == 'zone_to_world':
                to_world = row.zone_to_world
            else:
                to_world = getattr(row, to_world_key, None)

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                if use_offset:
                    transformed = to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                else:
                    transformed = to_world @ np.array([x, y, 1]).reshape(3, 1)
                p = Metashape.Vector([transformed[0, 0], transformed[1, 0]])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            result.append({
                'coords': corners,
                'label': label,
                'score': score
            })

        return result

    def _add_box_shapes_to_layer(self, data, shapes_group, to_world_key='zone_to_world', use_offset=True):
        """
        Unified method to add box shapes from prediction data.

        Parameters:
            data: DataFrame with prediction data
            shapes_group: Group to add shapes to
            to_world_key: Key/attribute name for transformation matrix
            use_offset: Whether to use 0.5 pixel offset (False for zones, True for tiles)

        Returns:
            list: Created shape objects
        """
        shapes = []
        for row in data.itertuples():
            xmin, ymin, xmax, ymax = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)
            label, score = row.label, row.score

            # Get transformation matrix
            if to_world_key == 'zone_to_world':
                to_world = row.zone_to_world
            else:
                to_world = getattr(row, to_world_key, None)

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                if use_offset:
                    transformed = to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                else:
                    transformed = to_world @ np.array([x, y, 1]).reshape(3, 1)
                p = Metashape.Vector([transformed[0, 0], transformed[1, 0]])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)
            shape.label = self.classes[int(label)]
            shape.attributes["Score"] = str(score)
            shapes.append({
                'coords': corners,
                'label': label,
                'score': score
            })

        return shapes

    def _compute_mask_coordinates(self, data, to_world_key='zone_to_world', use_offset=True):
        """
        Computes mask coordinates from prediction data without creating shapes.

        Parameters:
            data: DataFrame with prediction data containing mask column
            to_world_key: Key/attribute name for transformation matrix
            use_offset: Whether to use 0.5 pixel offset

        Returns:
            list: List of dictionaries containing:
                - 'coords': coordinate lists (corners) for each mask polygon
                - 'label': class label
                - 'score': confidence score
        """
        result = []
        for row in data.itertuples():
            mask = row.mask
            label, score = row.label, row.score

            # Get transformation matrix
            if to_world_key == 'zone_to_world':
                to_world = row.zone_to_world
            else:
                to_world = getattr(row, to_world_key, None)

            corners = []
            for coord in mask:
                x, y = coord
                if use_offset:
                    transformed = to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                else:
                    transformed = to_world @ np.array([x, y, 1]).reshape(3, 1)
                p = Metashape.Vector([transformed[0, 0], transformed[1, 0]])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            if len(corners) >= 3:
                result.append({
                    'coords': corners,
                    'label': label,
                    'score': score
                })
            else:
                print(f"Invalid polygon with less than 3 coordinates: {corners}")

        return result

    def _add_mask_shapes_to_layer(self, data, shapes_group, to_world_key='zone_to_world', use_offset=True):
        """
        Unified method to add mask shapes from prediction data.

        Parameters:
            data: DataFrame with prediction data containing mask column
            shapes_group: Group to add shapes to
            to_world_key: Key/attribute name for transformation matrix
            use_offset: Whether to use 0.5 pixel offset

        Returns:
            list: Created shape objects
        """
        shapes = []
        for row in data.itertuples():
            mask = row.mask
            label, score = row.label, row.score

            # Get transformation matrix
            if to_world_key == 'zone_to_world':
                to_world = row.zone_to_world
            else:
                to_world = getattr(row, to_world_key, None)

            corners = []
            for coord in mask:
                x, y = coord
                if use_offset:
                    transformed = to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                else:
                    transformed = to_world @ np.array([x, y, 1]).reshape(3, 1)
                p = Metashape.Vector([transformed[0, 0], transformed[1, 0]])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            if len(corners) >= 3:
                shape = self.chunk.shapes.addShape()
                shape.group = shapes_group
                shape.geometry = Metashape.Geometry.Polygon(corners)
                shape.label = self.classes[int(label)]
                shape.attributes["Score"] = str(score)
                shapes.append({
                    'coords': corners,
                    'label': label,
                    'score': score
                })
            else:
                print(f"Invalid polygon with less than 3 coordinates: {corners}")

        return shapes

    def _merge_and_apply_shapes(self, box_shapes, mask_shapes):
        """
        Merges overlapping shapes and applies them to the chunk.

        Parameters:
            box_shapes: List of box shape objects OR list of coordinate lists
            mask_shapes: List of mask shape objects OR list of coordinate lists
        """
        print(f"Processing overlapping contours...")

        start_time = time.time()

        # Handle both shape objects and coordinate lists
        # If first element is a shape object, extract coordinates and metadata
        box_data = extract_coords_metadata_shapes(box_shapes)
        mask_data = extract_coords_metadata_shapes(mask_shapes)

        # Group shapes by label
        box_groups = group_shapes_by_label(box_data)
        mask_groups = group_shapes_by_label(mask_data)

        # Process each label group separately
        merged_mask_results = []
        for label, masks in mask_groups.items():
            coords_list = [mask['coords'] for mask in masks]
            scores = [mask['score'] for mask in masks]

            merged_shapes, merged_indices = merge_contours_shapely(coords_list,
                                                                   iou_threshold=self.iouThresholdSpinBox.value())

            # Calculate average score for each merged shape based on its contributing inputs
            for merged_shape, contributing_indices in zip(merged_shapes, merged_indices):
                # Get scores of all shapes that were merged into this one
                contributing_scores = [scores[idx] for idx in contributing_indices]
                avg_score = sum(contributing_scores) / len(contributing_scores) if contributing_scores else 0.0

                try:
                    if self.TARGET_UTM_CRS:
                        shape_coords = convert_to_utm(merged_shape, source_crs=self.chunk.orthomosaic.crs,
                                                      target_crs=self.TARGET_UTM_CRS)
                    else:
                        shape_coords = merged_shape

                    poly = Polygon(shape_coords)
                    shape_area = poly.area
                    shape_centroid = (poly.centroid.x, poly.centroid.y)

                    # Get minimum rotated rectangle
                    min_rect = minimum_rotated_rectangle(poly)

                    if min_rect.is_empty:
                        shape_width = 0.0
                        shape_length = 0.0

                    else:
                        # min_rect is a Polygon with 5 points (closed rectangle)
                        coords = list(min_rect.exterior.coords)

                        # Take first three points: A, B, C
                        # AB and BC are adjacent sides

                        def dist(p1, p2):
                            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

                        side1 = dist(coords[0], coords[1])
                        side2 = dist(coords[1], coords[2])

                        # Ensure length >= width
                        shape_length = max(side1, side2)
                        shape_width = min(side1, side2)


                except Exception as e:
                    print(f"Error calculating area or dimensions for merged shape: {e}")
                    shape_area = 0.0
                    shape_centroid = None
                    shape_width = 0.0
                    shape_length = 0.0

                merged_mask_results.append({
                    'shape': merged_shape,
                    'label': label,
                    'score': avg_score,
                    'area': shape_area,
                    'centroid': shape_centroid,
                    'width': shape_width,
                    'length': shape_length
                })

        # Process boxes
        merged_box_results = []
        for label, boxes in box_groups.items():
            # Extract coordinates in xyxy format
            coords_boxes_xyxy = []
            scores = []

            for box in boxes:
                coords = box['coords']
                scores.append(box['score'])
                coords_boxes_xyxy.append([
                    min(point[0] for point in coords),  # x_min
                    min(point[1] for point in coords),  # y_min
                    max(point[0] for point in coords),  # x_max
                    max(point[1] for point in coords)  # y_max
                ])

            merged_boxes, merged_indices = merge_all_boxes(coords_boxes_xyxy,
                                                           iou_threshold=self.iouThresholdSpinBox.value())

            # Calculate average score for each merged box based on its contributing inputs
            for merged_box, contributing_indices in zip(merged_boxes, merged_indices):
                # Get scores of all boxes that were merged into this one
                contributing_scores = [scores[idx] for idx in contributing_indices]
                avg_score = sum(contributing_scores) / len(contributing_scores) if contributing_scores else 0.0

                if self.TARGET_UTM_CRS:
                    shape_coords = convert_to_utm(merged_box, source_crs=self.chunk.orthomosaic.crs,
                                                  target_crs=self.TARGET_UTM_CRS)
                    norm_bbox = vectors_to_bbox(shape_coords)
                else:
                    norm_bbox = merged_box

                # Calculate area of the merged box (format: [x_min, y_min, x_max, y_max])
                x_min, y_min, x_max, y_max = norm_bbox
                width = min(x_max - x_min, y_max - y_min)
                length = max(x_max - x_min, y_max - y_min)
                centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2]

                box_area = width * length

                merged_box_results.append({
                    'box': merged_box,
                    'label': label,
                    'score': avg_score,
                    'area': box_area,
                    'centroid': centroid,
                    'width': width,
                    'length': length
                })

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing {len(mask_data)} contours, result {len(merged_mask_results)} contours, "
              f"duration {elapsed_time / 60:.2f} minutes")

        # Apply merged shapes
        if merged_mask_results:
            union_outline_label = f"union_outline_detected ({100.0 * self.orthomosaic_resolution:.2f} cm/pix, " \
                                  f"size img: {self.max_image_size}, confidence: {self.detection_score_threshold})"
            self.apply_union_mask_shapes(self.chunk, merged_mask_results, union_outline_label)

        if merged_box_results:
            union_boxes_label = f"union_boxes_detected ({100.0 * self.orthomosaic_resolution:.2f} cm/pix, " \
                                f"size img: {self.max_image_size}, confidence: {self.detection_score_threshold})"
            self.apply_union_boxes_shapes(self.chunk, merged_box_results, union_boxes_label)

    def draw_boxes_zone_tiles(self, tiles_data, use_zones=False):
        """
        Draws boxes on zone tiles based on provided tile data and adds them as polygon shapes to the
        project's chunk.

        This function processes tile data containing bounding box coordinates and transforms them from
        zone-based coordinates to the appropriate coordinate system used within the Metashape project.
        The resultant polygons are labeled and grouped under a new shapes group.

        Args:
            tiles_data (list[dict]): A list of dictionaries where each dictionary represents a tile's
                                      data. Each dictionary must contain:
                                      - 'x_tile': int, X coordinate of the tile's bottom-left corner.
                                      - 'y_tile': int, Y coordinate of the tile's bottom-left corner.
                                      - 'x_max': int, X coordinate of the tile's top-right corner.
                                      - 'y_max': int, Y coordinate of the tile's top-right corner.
                                      - 'label': str, Label for the polygon shape.
                                      - 'zone_to_world': numpy.ndarray, Transformation matrix from zone
                                        to world coordinates.
            use_zones (bool): If True, coordinates are zone-local; if False, coordinates are absolute
                             and zone_to_world is tile-specific (already accounts for tile position).
        """

        shapes_group = self.chunk.shapes.addGroup()
        shapes_group.label = "Tiles Boxes"
        shapes_group.show_labels = False

        for row in tiles_data:
            xmin = int(row["x_tile"])
            ymin = int(row["y_tile"])
            xmax = int(row["x_max"])
            ymax = int(row["y_max"])
            label = row["label"]
            zone_to_world = row["zone_to_world"]

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                if use_zones:
                    # For zones mode: coordinates are zone-local, use them directly with offset
                    transformed = zone_to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                else:
                    # For orthomosaic mode: coordinates are absolute, but zone_to_world is tile-specific
                    # We need to convert absolute coords to tile-local coords (0,0 at tile origin)
                    tile_local_x = x - xmin
                    tile_local_y = y - ymin
                    transformed = zone_to_world @ np.array([tile_local_x + 0.5, tile_local_y + 0.5, 1]).reshape(3, 1)

                p = Metashape.Vector([transformed[0, 0], transformed[1, 0]])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)
            shape.label = label

    def get_tiles_from_zones(self):
        """
        Extracts and processes tiles from zones based on detected shapes and spatial relationships.

        This method identifies tiles from zones defined by shapes, applying coordinate transformations
        to determine the zones' presence and placement within the orthomosaic project. It further calculates
        dimensions, verifies permissible sizes, and processes tiles accordingly while noting tiles outside
        the orthomosaic or with excessive white pixels.

        Parameters
        ----------
        self : class
            An instance of the class containing detected zones, tiles' spatial metadata, and necessary
            attributes for tile extraction and processing.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries where each dictionary represents a processed tile. Each dictionary
            contains the tile's data, position information, its label, and a transformation matrix.
        """

        app = QtWidgets.QApplication.instance()

        all_tiles_zones = []
        zones_on_ortho = []

        for zone_i, shape in enumerate(self.detected_zones):
            shape_vertices = getShapeVertices(shape)
            zone_from_world = None
            zone_from_world_best = None
            zone_to_world = None

            for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
                for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                    if (tile_x, tile_y) not in self.tiles_paths:
                        continue
                    to_world = self.tiles_to_world[tile_x, tile_y]
                    from_world = self.invert_matrix_2x3(to_world)
                    for v_p in shape_vertices:
                        p = Metashape.CoordinateSystem.transform(v_p, self.chunk.shapes.crs,
                                                                 self.chunk.orthomosaic.crs)
                        p_in_tile = from_world @ [p.x, p.y, 1]
                        distance2_to_tile_center = np.linalg.norm(
                            p_in_tile - [self.patch_size / 2, self.patch_size / 2])
                        if zone_from_world_best is None or distance2_to_tile_center < zone_from_world_best:
                            zone_from_world_best = distance2_to_tile_center
                            zone_from_world = self.invert_matrix_2x3(
                                self.add_pixel_shift(to_world, -tile_x * self.patch_size, -tile_y * self.patch_size))
                            zone_to_world = self.add_pixel_shift(to_world, -tile_x * self.patch_size,
                                                                 -tile_y * self.patch_size)

            zone_from = None
            zone_to = None
            for v_p in shape_vertices:
                p = Metashape.CoordinateSystem.transform(v_p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))
                if zone_from is None:
                    zone_from = p_in_ortho
                if zone_to is None:
                    zone_to = p_in_ortho
                zone_from = np.minimum(zone_from, p_in_ortho)
                zone_to = np.maximum(zone_to, p_in_ortho)
            train_size = zone_to - zone_from
            train_size_m = np.int32(np.round(train_size * self.orthomosaic_resolution))
            if np.any(train_size < self.patch_size):
                print("Zone #{} {}x{} pixels ({}x{} meters) is too small - each side should be at least {} meters"
                      .format(zone_i + 1, train_size[0], train_size[1], train_size_m[0], train_size_m[1],
                              self.patch_size * self.orthomosaic_resolution), file=sys.stderr)
                zones_on_ortho.append(None)
            else:
                print("Zone #{}: {}x{} orthomosaic pixels, {}x{} meters".format(zone_i + 1, train_size[0],
                                                                                train_size[1], train_size_m[0],
                                                                                train_size_m[1]))

                self.check_stopped()

                border = self.patch_inner_border
                inner_path_size = self.patch_size - 2 * border

                zone_size = zone_to - zone_from
                assert np.all(zone_size >= self.patch_size)
                nx_tiles, ny_tiles = np.int32((zone_size - 2 * border + inner_path_size - 1) // inner_path_size)
                assert nx_tiles >= 1 and ny_tiles >= 1
                xy_step = np.int32(np.round((zone_size + [nx_tiles, ny_tiles] - 1) // [nx_tiles, ny_tiles]))

                out_of_orthomosaic_train_tile = 0
                total_steps = nx_tiles * ny_tiles

                for x_tile in range(0, nx_tiles):
                    for y_tile in range(0, ny_tiles):

                        current_step = x_tile * ny_tiles + y_tile + 1
                        progress = (current_step * 100) / total_steps
                        self.detectionPBar.setValue(int(progress))
                        app.processEvents()
                        self.check_stopped()

                        tile_to = zone_from + self.patch_size + xy_step * [x_tile, y_tile]
                        if x_tile == nx_tiles - 1 and y_tile == ny_tiles - 1:
                            assert np.all(tile_to >= zone_to)
                        tile_to = np.minimum(tile_to, zone_to)
                        tile_from = tile_to - self.patch_size
                        if x_tile == 0 and y_tile == 0:
                            assert np.all(tile_from == zone_from)
                        assert np.all(tile_from >= zone_from)

                        tile = self.read_part(tile_from, tile_to)
                        assert tile.shape == (self.patch_size, self.patch_size, 3)

                        white_pixels_fraction = np.sum(np.all(tile == 255, axis=-1)) / (
                                tile.shape[0] * tile.shape[1])
                        if np.all(tile == 255) or white_pixels_fraction >= 0.90:
                            out_of_orthomosaic_train_tile += 1
                            continue

                        label_tile = f"{(zone_i + 1)}-{x_tile}-{y_tile}"

                        all_tiles_zones.append({"tile": tile, "x_tile": tile_from[0], "y_tile": tile_from[1],
                                                "x_max": tile_to[0], "y_max": tile_to[1], "label": label_tile,
                                                "zone_to_world": zone_to_world})

        print(f"Tiles: {len(all_tiles_zones)}")
        return all_tiles_zones

    def get_tiles_from_orthomosaic(self):
        """
        Extracts and processes tiles from the entire orthomosaic with overlap.

        Returns tiles in the same format as get_tiles_from_zones() for unified processing.
        Implements tile overlap similar to get_tiles_from_zones() using patch_inner_border.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries where each dictionary represents a processed tile.
        """
        app = QtWidgets.QApplication.instance()
        all_tiles = []

        # Calculate the orthomosaic dimensions based on tile range
        ortho_from = np.array([self.tile_min_x * self.patch_size, self.tile_min_y * self.patch_size])
        ortho_to = np.array([(self.tile_max_x + 1) * self.patch_size, (self.tile_max_y + 1) * self.patch_size])
        ortho_size = ortho_to - ortho_from

        # Use border for overlap, same as in get_tiles_from_zones
        border = self.patch_inner_border
        inner_patch_size = self.patch_size - 2 * border

        # Calculate number of tiles with overlap
        # Same logic as in get_tiles_from_zones
        assert np.all(ortho_size >= self.patch_size)
        nx_tiles = int((ortho_size[0] - 2 * border + inner_patch_size - 1) // inner_patch_size)
        ny_tiles = int((ortho_size[1] - 2 * border + inner_patch_size - 1) // inner_patch_size)
        assert nx_tiles >= 1 and ny_tiles >= 1

        # Calculate step between tiles to cover the entire orthomosaic
        xy_step = np.int32(np.round((ortho_size + [nx_tiles, ny_tiles] - 1) // [nx_tiles, ny_tiles]))

        # Find a reference tile to get the transformation matrix
        reference_to_world = None
        reference_tile_x = None
        reference_tile_y = None
        for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
            for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                if (tile_x, tile_y) in self.tiles_paths:
                    reference_to_world = self.tiles_to_world[tile_x, tile_y]
                    reference_tile_x = tile_x
                    reference_tile_y = tile_y
                    break
            if reference_to_world is not None:
                break

        if reference_to_world is None:
            print("No tiles found in orthomosaic!")
            return all_tiles

        out_of_orthomosaic_tiles = 0
        total_steps = nx_tiles * ny_tiles

        # Generate overlapping tiles
        for x_tile in range(nx_tiles):
            for y_tile in range(ny_tiles):

                current_step = x_tile * ny_tiles + y_tile + 1
                progress = (current_step * 100) / total_steps
                self.detectionPBar.setValue(int(progress))
                app.processEvents()
                self.check_stopped()

                # Calculate tile boundaries
                tile_to = ortho_from + self.patch_size + xy_step * [x_tile, y_tile]
                if x_tile == nx_tiles - 1 and y_tile == ny_tiles - 1:
                    assert np.all(tile_to >= ortho_to)
                tile_to = np.minimum(tile_to, ortho_to)
                tile_from = tile_to - self.patch_size
                if x_tile == 0 and y_tile == 0:
                    assert np.all(tile_from == ortho_from)
                assert np.all(tile_from >= ortho_from)

                # Read the tile part from orthomosaic
                tile = self.read_part(tile_from, tile_to)
                assert tile.shape == (self.patch_size, self.patch_size, 3)

                # Check if tile is mostly white (outside orthomosaic)
                white_pixels_fraction = np.sum(np.all(tile == 255, axis=-1)) / (
                        tile.shape[0] * tile.shape[1])
                if np.all(tile == 255) or white_pixels_fraction >= 0.90:
                    out_of_orthomosaic_tiles += 1
                    continue

                # Calculate transformation matrix for this specific tile
                # Similar to the old detect() method where each tile has its own transformation
                # tile_from is already in absolute coordinates, so we just need to shift from reference tile
                tile_to_world = self.add_pixel_shift(reference_to_world,
                                                     tile_from[0] - reference_tile_x * self.patch_size,
                                                     tile_from[1] - reference_tile_y * self.patch_size)

                label_tile = f"ortho_{x_tile}_{y_tile}"

                all_tiles.append({
                    "tile": tile,
                    "x_tile": tile_from[0],
                    "y_tile": tile_from[1],
                    "x_max": tile_to[0],
                    "y_max": tile_to[1],
                    "label": label_tile,
                    "zone_to_world": tile_to_world
                })

        print(f"Tiles from orthomosaic: {len(all_tiles)} (skipped {out_of_orthomosaic_tiles} white tiles)")
        return all_tiles

    def detect_unified(self, use_zones=False):
        """
        Unified detection method that processes either selected zones or entire orthomosaic.

        Parameters
        ----------
        use_zones : bool, optional
            If True, performs prediction on selected zones only.
            If False, performs prediction on entire orthomosaic.
            Default is False.
        """
        app = QtWidgets.QApplication.instance()

        if use_zones:
            print("Detection selected zones...")
            if not hasattr(self, 'detected_zones') or not self.detected_zones:
                print("No zones detected!")
                return
            print(f"Zones: {len(self.detected_zones)}")
        else:
            print("Detection on entire orthomosaic...")

        if not self.model:
            print("No init model!")
            return

        print(f"Classes: {self.classes}")
        print(f"Size tile: {self.max_image_size}")

        self.txtDetectionPBar.setText("Prepare progress:")

        # Get tiles based on mode
        if use_zones:
            tiles_data = self.get_tiles_from_zones()
        else:
            tiles_data = self.get_tiles_from_orthomosaic()

        if self.isDebugMode:
            self.draw_boxes_zone_tiles(tiles_data, use_zones=use_zones)

        # Initialize predictions DataFrame
        subtile_inner_preds = pd.DataFrame(
            columns=['zone_to_world', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'mask'])

        # Process each tile
        for tile_index, tile_data in enumerate(tiles_data):
            self.txtDetectionPBar.setText(f"Detection progress: ({tile_index + 1} of {len(tiles_data)})")
            self.detectionPBar.setValue(int((tile_index + 1) * 100 / len(tiles_data)))

            subtile = tile_data["tile"]
            x_tile = tile_data["x_tile"]
            y_tile = tile_data["y_tile"]
            zone_to_world = tile_data["zone_to_world"]

            # Execute prediction (using unified method)
            subtile_prediction = self._run_prediction(subtile)

            Metashape.app.update()
            app.processEvents()
            self.check_stopped()

            if subtile_prediction.boxes is not None:
                # Get image dimensions
                original_shape = subtile_prediction.orig_shape
                orig_h, orig_w = original_shape[:2]

                masks = None
                if subtile_prediction.masks is not None:
                    masks = subtile_prediction.masks.xyn

                boxes = subtile_prediction.boxes.xyxyn

                for idx, bbox in enumerate(boxes):
                    box = subtile_prediction.boxes[idx]
                    score = box.conf.numpy()[0]
                    b_class = box.cls.numpy()
                    label = b_class[0]

                    xmin, ymin, xmax, ymax = bbox
                    xmin, ymin, xmax, ymax = int(xmin * orig_w), int(ymin * orig_h), int(xmax * orig_w), int(
                        ymax * orig_h)

                    # For zones mode: zone_to_world transforms from zone-local coords, so we need absolute coords
                    # For orthomosaic mode: tile_to_world transforms from tile-local coords (0,0 at tile origin)
                    if use_zones:
                        # Add tile position offset to get zone-local coordinates
                        xmin, xmax = map(lambda x: x_tile + x, [xmin, xmax])
                        ymin, ymax = map(lambda y: y_tile + y, [ymin, ymax])
                        x_offset, y_offset = x_tile, y_tile
                    else:
                        # Keep tile-local coordinates (no offset needed)
                        # tile_to_world already accounts for tile position in orthomosaic
                        x_offset, y_offset = 0, 0

                    row = {"zone_to_world": zone_to_world, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                           "label": label, "score": score, 'mask': []}

                    if masks is not None and len(masks) > 0:
                        # Process mask (using unified method)
                        mask = masks[idx]
                        mask_contour = self._process_mask_contour(mask, orig_h, orig_w, x_offset, y_offset)
                        row['mask'] = mask_contour

                    subtile_inner_preds = pandas_append(subtile_inner_preds, pd.DataFrame([row]),
                                                        ignore_index=True)

        # Create shapes for processing (regardless of debug mode)
        boxes_shapes = []
        outline_shapes = []

        if self.isDebugMode:
            # In debug mode, create new layers for results
            box_detected_label = f"box_detected ({100.0 * self.orthomosaic_resolution:.2f} cm/pix, " \
                                 f"size img: {self.max_image_size}, confidence: {self.detection_score_threshold})"

            detected_shapes_layer = self.chunk.shapes.addGroup()
            detected_shapes_layer.label = box_detected_label
            detected_shapes_layer.show_labels = False

            # Use unified method to add boxes
            boxes_shapes = self._add_box_shapes_to_layer(subtile_inner_preds, detected_shapes_layer,
                                                         to_world_key='zone_to_world', use_offset=True)

            if subtile_inner_preds['mask'].notna().any():
                detected_mask_label = self.layer_name_detection_data + f"outline_detected ({100.0 * self.orthomosaic_resolution:.2f} cm/pix, " \
                                                                       f"size img: {self.max_image_size}, confidence: {self.detection_score_threshold})"
                detected_mask_shapes_layer = self.chunk.shapes.addGroup()
                detected_mask_shapes_layer.label = detected_mask_label
                detected_mask_shapes_layer.show_labels = False
                # Use unified method to add masks
                outline_shapes = self._add_mask_shapes_to_layer(subtile_inner_preds, detected_mask_shapes_layer,
                                                                to_world_key='zone_to_world', use_offset=True)
        else:
            # In normal mode, compute coordinates directly without creating temporary layers
            boxes_shapes = self._compute_box_coordinates(subtile_inner_preds,
                                                         to_world_key='zone_to_world', use_offset=True)

            if subtile_inner_preds['mask'].notna().any():
                outline_shapes = self._compute_mask_coordinates(subtile_inner_preds,
                                                                to_world_key='zone_to_world', use_offset=True)

        # Use unified method to merge and apply shapes
        self._merge_and_apply_shapes(boxes_shapes, outline_shapes)

        Metashape.app.update()

    def apply_union_mask_shapes(self, chunk, new_shapes, detected_label):
        """
        Applies a union mask to the given shapes in a 3D chunk model, organizing the new
        shapes into a labeled group within the chunk. This method processes a collection
        of geometric shapes, creates new polygonal shapes from their corner coordinates,
        and assigns them to a labeled group with specific display settings.

        Parameters:
            chunk (Metashape.Chunk): The 3D chunk object where shapes will be modified and
                grouped.
            new_shapes (list): A list of dictionaries containing 'shape' (coordinates),
                'label' (class label), and 'score' (confidence score) for each shape.
            detected_label (str): Label to be assigned to the created shapes group for
                organizational purposes, helping to identify added shapes.
        """
        shapes_layer = self.chunk.shapes.addGroup()
        shapes_layer.label = detected_label
        shapes_layer.show_labels = True  # Enable labels to show class and score

        # Prepare list to collect CSV data
        csv_data = []

        for shape_data in new_shapes:

            new_shape = chunk.shapes.addShape()
            new_shape.group = shapes_layer

            # Get shape coordinates
            corners = shape_data['shape']

            # Create polygon geometry
            new_shape.geometry = Metashape.Geometry.Polygon([Metashape.Vector([x, y]) for x, y in corners])

            # Set label and score if available
            label = shape_data['label']
            score = shape_data['score']
            area = shape_data['area']
            centroid = shape_data['centroid']
            width = shape_data['width']
            length = shape_data['length']

            if self.isDebugMode:
                print(
                    f"label: {label}, score: {score}, area: {area}, centroid: {centroid}, width: {width}, length: {length}")

            new_shape.attributes["Width (m)"] = f"{width:.3f}"
            new_shape.attributes["Length (m)"] = f"{length:.3f}"

            if centroid is not None:
                new_shape.attributes["Centroid (x,y)"] = f"{centroid}"

            if area is not None:
                new_shape.attributes["Area 2D (m²)"] = f"{area:.2f}"

            # Determine class name
            class_name = 'Unknown'
            if label is not None:
                if isinstance(label, (int, float)) and 0 <= int(label) < len(self.classes):
                    class_name = self.classes[int(label)]
                else:
                    class_name = str(label)

                new_shape.label = class_name

            # Add score as attribute
            if score is not None:
                new_shape.attributes["Score (avg)"] = f"{score:.2f}"

            # Collect data for CSV export
            csv_row = {
                'Label': class_name,
                'Score (avg)': f"{score:.2f}" if score is not None else 'N/A',
                'Area 2D (m²)': f"{area:.2f}" if area is not None else 'N/A',
                'Centroid': f"{centroid}" if centroid is not None else 'N/A',
                'Width (m)': f"{width:.3f}",
                'Length (m)': f"{length:.3f}"
            }
            csv_data.append(csv_row)

        # Export collected data to CSV file
        if csv_data:
            # Create filename with timestamp and detected_label
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Sanitize detected_label for filename
            # safe_label = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in detected_label)
            csv_filename = f"detection_results_{timestamp}.csv"
            csv_filepath = os.path.join(self.dir_detection_results, csv_filename)

            # Write CSV file
            with open(csv_filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['Label', 'Score (avg)', 'Area 2D (m\u00b2)', 'Centroid', 'Width (m)', 'Length (m)']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(csv_data)

            print(f"Detection results exported to: {csv_filepath}")

    def apply_union_boxes_shapes(self, chunk, new_shapes, detected_label):
        """
        Applies union boxes' shapes to a given chunk and assigns them to a new layer.

        This method works by creating a new shapes group to represent the bounding boxes.
        It assigns the provided label to the new shapes layer and populates it with
        polygonal shapes defined by the input list of dictionaries containing box coordinates,
        class labels, and confidence scores. Each shape is transformed from the orthomosaic
        coordinate reference system to the shapes coordinate reference system before being
        added to the group.

        Parameters:
        chunk (Metashape.Chunk): The target chunk to which the shapes are added.
        new_shapes (list): A list of dictionaries containing 'box' (coordinates in xmin, ymin, xmax, ymax format),
            'label' (class label), and 'score' (confidence score) for each shape.
        detected_label (str): The label to assign to the newly created shapes layer.
        """
        shapes_layer = self.chunk.shapes.addGroup()
        shapes_layer.label = detected_label
        shapes_layer.show_labels = True  # Enable labels to show class and score

        for shape_data in new_shapes:
            box = shape_data['box']
            label = shape_data['label']
            score = shape_data['score']
            centroid = shape_data['centroid']
            width = shape_data['width']
            length = shape_data['length']
            area = shape_data['area']

            xmin, ymin, xmax, ymax = box
            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                p = Metashape.Vector([x, y])
                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            new_shape = chunk.shapes.addShape()
            new_shape.group = shapes_layer
            new_shape.geometry = Metashape.Geometry.Polygon(corners)

            new_shape.attributes["Width (m)"] = f"{width:.3f}"
            new_shape.attributes["Length (m)"] = f"{length:.3f}"

            if centroid is not None:
                new_shape.attributes["Centroid (x,y)"] = f"{centroid}"

            if area is not None:
                new_shape.attributes["Area 2D (m²)"] = f"{area:.2f}"

            # Set label and score if available
            if label is not None:
                if isinstance(label, (int, float)) and 0 <= int(label) < len(self.classes):
                    class_name = self.classes[int(label)]
                else:
                    class_name = str(label)

                new_shape.label = class_name

            # Add score as attribute
            if score is not None:
                new_shape.attributes["Score (avg)"] = f"{score:.2f}"

    def show_results_dialog(self, results_time_total):
        """
        Displays a dialog showing the total processing time.

        Parameters:
            results_time_total (float): Total processing time in seconds.
        """
        message = "Finished in {:.2f} sec:\n".format(results_time_total)

        print(message)
        Metashape.app.messageBox(message)

    def create_gui(self):
        """
        Creates and initializes the graphical user interface for the detection dialog.

        This method sets up all UI components including input fields, buttons, labels,
        progress bars, and organizes them in a structured layout. It also connects
        signals to appropriate slots for user interaction handling.
        """
        self.labelDetectZonesLayer = QtWidgets.QLabel("Layer zones:")
        self.detectZonesLayer = QtWidgets.QComboBox()
        self.noDataChoice = (None, "No additional (use as is)", True)
        self.layers = [self.noDataChoice]

        slow_shape_layers_enumerating_but_with_number_of_shapes = False
        if slow_shape_layers_enumerating_but_with_number_of_shapes:
            print("Enumerating all shape layers...")
            shapes_enumerating_start = time.time()
            self.layersDict = {}
            self.layersSize = {}
            shapes = self.chunk.shapes

            for shape in shapes:
                layer = shape.group
                if layer.key not in self.layersDict:
                    self.layersDict[layer.key] = (layer.key, layer.label, layer.enabled)
                    self.layersSize[layer.key] = 1
                else:
                    self.layersSize[layer.key] += 1

            print("Found {} shapes layers in {:.2f} sec:".format(len(self.layersDict),
                                                                 time.time() - shapes_enumerating_start))
            for key in sorted(self.layersDict.keys()):
                key, label, enabled = self.layersDict[key]
                size = self.layersSize[key]

                print("Shape layer: {} shapes, key={}, label={}".format(size, key, label))

                if label == '':
                    label = 'Layer'

                label = label + " ({} shapes)".format(size)
                self.layers.append((key, label, enabled))

            self.layersDict = None
            self.layersSize = None

        else:
            if self.chunk.shapes is None:
                print("No shapes")
            else:
                for layer in self.chunk.shapes.groups:
                    key, label, enabled = layer.key, layer.label, layer.enabled
                    if not enabled:
                        continue

                    print("Shape layer: key={}, label={}, enabled={}".format(key, label, enabled))

                    if label == '':
                        label = 'Layer'

                    self.layers.append((key, label, layer.enabled))

        for key, label, enabled in self.layers:
            self.detectZonesLayer.addItem(label)

        self.detectZonesLayer.setCurrentIndex(0)

        for i, (key, label, enabled) in enumerate(self.layers):

            if label.lower().startswith(self.expected_layer_name_train_zones.lower()):
                self.detectZonesLayer.setCurrentIndex(i)

        self.chkUse5mmResolution = QtWidgets.QCheckBox("Process with 0.50 cm/pix resolution")
        self.chkUse5mmResolution.setToolTip(
            "Process with downsampling to 0.50 cm/pix instad of original orthomosaic resolution.")
        self.chkUse5mmResolution.setChecked(not self.prefer_original_resolution)

        self.groupBoxGeneral = QtWidgets.QGroupBox("General")
        generalLayout = QtWidgets.QGridLayout()

        self.labelWorkingDir = QtWidgets.QLabel()
        self.labelWorkingDir.setText("Working dir:")
        self.workingDirLineEdit = QtWidgets.QLineEdit()
        self.workingDirLineEdit.setText(self.working_dir)
        self.workingDirLineEdit.setPlaceholderText("Path to dir for intermediate data")
        self.workingDirLineEdit.setToolTip("Path to dir for intermediate data")
        self.btnWorkingDir = QtWidgets.QPushButton("...")
        self.btnWorkingDir.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnWorkingDir, QtCore.SIGNAL("clicked()"), lambda: self.choose_working_dir())
        generalLayout.addWidget(self.labelWorkingDir, 0, 0)
        generalLayout.addWidget(self.workingDirLineEdit, 0, 1)
        generalLayout.addWidget(self.btnWorkingDir, 0, 2)

        generalLayout.addWidget(self.chkUse5mmResolution, 1, 1)

        self.debugModeCbox = QtWidgets.QCheckBox("Debug mode")
        generalLayout.addWidget(self.debugModeCbox, 4, 1, 1, 2)

        self.maxSizeImageSpinBox = QtWidgets.QSpinBox(self)
        self.maxSizeImageSpinBox.setMaximumWidth(150)
        self.maxSizeImageSpinBox.setMinimum(256)
        self.maxSizeImageSpinBox.setMaximum(2048)
        self.maxSizeImageSpinBox.setSingleStep(256)
        self.maxSizeImageSpinBox.setValue(self.max_image_size)
        self.maxSizeImageLabel = QtWidgets.QLabel("Max size tiles:")
        generalLayout.addWidget(self.maxSizeImageLabel, 5, 0)
        generalLayout.addWidget(self.maxSizeImageSpinBox, 5, 1, 1, 2)

        generalLayout.addWidget(self.labelDetectZonesLayer, 7, 0)
        generalLayout.addWidget(self.detectZonesLayer, 7, 1, 1, 2)

        self.groupBoxGeneral.setLayout(generalLayout)
        # Создаем таб-панель
        self.tabWidget = QtWidgets.QTabWidget()

        self.modelLoadPathLabel = QtWidgets.QLabel()
        self.modelLoadPathLabel.setText("Load model from:")
        self.modelLoadPathLineEdit = QtWidgets.QLineEdit()
        self.modelLoadPathLineEdit.setText(self.load_model_path)
        self.modelLoadPathLineEdit.setPlaceholderText(
            "File with previously saved neural network model (resolution must be the same)")
        self.modelLoadPathLineEdit.setToolTip(
            "File with previously saved neural network model (resolution must be the same)")
        self.btnModelLoadPath = QtWidgets.QPushButton("...")
        self.btnModelLoadPath.setFixedSize(25, 25)
        QtCore.QObject.connect(self.btnModelLoadPath, QtCore.SIGNAL("clicked()"), lambda: self.choose_model_load_path())
        generalLayout.addWidget(self.modelLoadPathLabel, 8, 0)
        generalLayout.addWidget(self.modelLoadPathLineEdit, 8, 1)
        generalLayout.addWidget(self.btnModelLoadPath, 8, 2)

        self.tabModelPrediction = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabModelPrediction, "Model detection")
        detectionLayout = QtWidgets.QGridLayout()

        # detection_score_threshold
        self.scoreThresholdLabel = QtWidgets.QLabel("Confidence Threshold:")
        self.scoreThresholdLabel.setFixedWidth(130)
        self.scoreThresholdLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.scoreThresholdSpinBox = CustomDoubleSpinBox()
        self.scoreThresholdSpinBox.setMaximumWidth(150)
        self.scoreThresholdSpinBox.setRange(0, 1)
        self.scoreThresholdSpinBox.setSingleStep(0.0001)
        self.scoreThresholdSpinBox.setDecimals(5)
        self.scoreThresholdSpinBox.setValue(self.detection_score_threshold)
        self.scoreThresholdSpinBox.valueChanged.connect(self.save_score_threshold)

        detectionLayout.addWidget(self.scoreThresholdLabel, 0, 0)
        detectionLayout.addWidget(self.scoreThresholdSpinBox, 0, 1, 1, 2)

        # iou_threshold
        self.iouThresholdLabel = QtWidgets.QLabel("IOU Threshold:")
        self.iouThresholdLabel.setFixedWidth(130)
        self.iouThresholdLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.iouThresholdSpinBox = CustomDoubleSpinBox()
        self.iouThresholdSpinBox.setMaximumWidth(150)
        self.iouThresholdSpinBox.setRange(0, 1)
        self.iouThresholdSpinBox.setSingleStep(0.01)
        self.iouThresholdSpinBox.setDecimals(2)
        self.iouThresholdSpinBox.setValue(self.iou_threshold)
        self.iouThresholdSpinBox.valueChanged.connect(self.save_iou_threshold)

        detectionLayout.addWidget(self.iouThresholdLabel, 1, 0)
        detectionLayout.addWidget(self.iouThresholdSpinBox, 1, 1, 1, 2)

        self.btnDetect = QtWidgets.QPushButton("Detection")
        self.btnDetect.setMaximumWidth(100)
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        self.btnStop.setMaximumWidth(100)

        layout = QtWidgets.QGridLayout()
        row = 0

        layout.addWidget(self.groupBoxGeneral, row, 0, 1, 3)
        row += 1

        self.tabModelPrediction.setLayout(detectionLayout)
        layout.addWidget(self.tabWidget, row, 0, 1, 3)
        row += 1

        self.txtInfoPBar = QtWidgets.QLabel()
        self.txtInfoPBar.setText("")
        layout.addWidget(self.txtInfoPBar, row, 1, 1, 3)
        row += 1

        self.txtDetectionPBar = QtWidgets.QLabel()
        self.txtDetectionPBar.setText("Progress:")
        self.detectionPBar = QtWidgets.QProgressBar()
        self.detectionPBar.setTextVisible(True)
        layout.addWidget(self.txtDetectionPBar, row, 0)
        layout.addWidget(self.detectionPBar, row, 1, 1, 2)
        row += 1

        layout.addWidget(self.btnDetect, row, 1)
        layout.addWidget(self.btnStop, row, 3)
        row += 1

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnDetect, QtCore.SIGNAL("clicked()"), lambda: self.run_detect())
        QtCore.QObject.connect(self.btnStop, QtCore.SIGNAL("clicked()"), lambda: self.stop())

        self.debugModeCbox.stateChanged.connect(self.change_debug_mode)

    def change_debug_mode(self, value):
        """
        Handles changes to the debug mode checkbox state.

        Parameters:
            value (bool): The new checkbox state (True for checked, False for unchecked).
        """
        self.isDebugMode = value
        print(f"Debug mode: {'On' if value else 'Off'}")

    def save_score_threshold(self, value):
        """Save score threshold value to settings when changed."""
        Metashape.app.settings.setValue("scripts/yolo/score-threshold", str(value))
        print(f"Score threshold saved: {value}")

    def save_iou_threshold(self, value):
        """Save IOU threshold value to settings when changed."""
        Metashape.app.settings.setValue("scripts/yolo/iou-threshold", str(value))
        print(f"IOU threshold saved: {value}")

    def choose_working_dir(self):
        """
        Opens a directory selection dialog and updates the working directory field.

        This method allows the user to select a working directory through a file dialog
        and updates the corresponding line edit widget with the selected path.
        """
        working_dir = Metashape.app.getExistingDirectory()
        self.workingDirLineEdit.setText(working_dir)

    def choose_model_save_path(self):
        """
        Opens a file dialog to select a path for saving a trained model.

        This method retrieves the previously used model directory from settings,
        displays a save file dialog to the user, and updates the model save path
        line edit widget with the selected file path.
        """
        models_dir = ""
        load_path = Metashape.app.settings.value("scripts/yolo/model_load_path")
        if load_path is not None:
            models_dir = str(pathlib.Path(load_path).parent)

        save_path = Metashape.app.getSaveFileName("Trained model save path", models_dir,
                                                  "Model Files (*.model *.pth *.pt);;All Files (*)")
        if len(save_path) <= 0:
            return

        self.modelSavePathLineEdit.setText(save_path)

    def choose_model_load_path(self):
        """
        Opens a file dialog to select a trained model file for loading.

        This method displays an open file dialog filtered for model files,
        and updates the model load path line edit widget with the selected file path.
        """
        load_path = Metashape.app.getOpenFileName("Trained model load path", "",
                                                  "Model Files (*.model *.pth *.pt);;All Files (*)")
        self.modelLoadPathLineEdit.setText(load_path)

    def load_params(self):
        """
        Loads and validates parameters from the GUI for the detection process.

        This method retrieves user-configured parameters from the GUI widgets,
        validates them, calculates derived values (like patch size and resolution),
        and prepares the detection zones based on the selected layer.

        Raises:
            Exception: If orthomosaic resolution exceeds 10 cm/pix.
        """

        app = QtWidgets.QApplication.instance()
        self.prefer_original_resolution = not self.chkUse5mmResolution.isChecked()

        self.max_image_size = self.maxSizeImageSpinBox.value()
        self.preferred_patch_size = self.max_image_size

        Metashape.app.settings.setValue("scripts/yolo/max_image_size", str(self.max_image_size))

        if not self.prefer_original_resolution:
            self.orthomosaic_resolution = self.preferred_resolution
            self.patch_size = self.preferred_patch_size
        else:
            self.orthomosaic_resolution = self.chunk.orthomosaic.resolution

            if self.orthomosaic_resolution > 0.105:
                raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")
            if self.force_small_patch_size:
                patch_size_multiplier = 1
            else:
                patch_size_multiplier = max(1, min(4, self.preferred_resolution / self.orthomosaic_resolution))

            self.patch_size = round(self.preferred_patch_size * patch_size_multiplier)

        self.patch_inner_border = self.patch_size // 8

        print("Using resolution {} m/pix with patch {}x{}".format(self.orthomosaic_resolution, self.patch_size,
                                                                  self.patch_size))
        self.working_dir = self.workingDirLineEdit.text()
        self.load_model_path = self.modelLoadPathLineEdit.text()
        self.detection_score_threshold = self.scoreThresholdSpinBox.value()
        self.iou_threshold = self.iouThresholdSpinBox.value()

        # Save threshold values to settings
        Metashape.app.settings.setValue("scripts/yolo/score-threshold", str(self.detection_score_threshold))
        Metashape.app.settings.setValue("scripts/yolo/iou-threshold", str(self.iou_threshold))

        detectZonesLayer = self.layers[self.detectZonesLayer.currentIndex()]

        if detectZonesLayer == self.noDataChoice:
            self.detect_on_user_layer_enabled = False
            print("Layer detect zones disabled")
        else:
            self.detect_on_user_layer_enabled = True
            print("Detecting expected on key={} layer zones".format(detectZonesLayer[0]))

        print("Loading shapes...")

        loading_shapes_start = time.time()
        shapes = self.chunk.shapes
        self.detected_zones = []

        print(f"All shapes chunk: {len(shapes)}")

        # Get layer key
        _zones_key = detectZonesLayer[0]

        for i_sp, shape in enumerate(shapes):
            i_sp += 1
            self.detectionPBar.setValue(int((100 * i_sp + 1) / len(shapes)))
            app.processEvents()
            self.check_stopped()

            if not shape.group.enabled:
                continue

            if shape.group.key == _zones_key:
                self.detected_zones.append(shape)

        print("{} zones  loaded in {:.2f} sec".format(len(self.detected_zones),
                                                      time.time() - loading_shapes_start))


class CustomDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """
    Custom implementation of QDoubleSpinBox to modify text representation.

    This class provides a customized text representation for the numerical
    values of a QDoubleSpinBox. It removes trailing zeroes and unnecessary
    decimal points from the displayed value for cleaner visualization.

    Attributes
    ----------
    Inherited from QDoubleSpinBox.
    """

    def textFromValue(self, value):
        import re
        text = super(CustomDoubleSpinBox, self).textFromValue(value)
        return re.sub(r'0*$', '', re.sub(r'\.0*$', '', text))




