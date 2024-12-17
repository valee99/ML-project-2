import os
import json
import numpy as np
import tifffile as tiff
import argparse
import cv2
from tqdm import tqdm
import warnings
from os.path import exists, join, dirname


def get_geojson_files(geojson_dir: str) -> set:
    """
    Get set of geoJSON file names without extensions.

    Arguments:
        geojson_dir: a string denoting the path to the geojson files
    """
    return {
        os.path.splitext(f)[0]
        for f in os.listdir(geojson_dir)
        if f.endswith(".geojson")
    }


def process_slice(
    slice_idx: int,
    geojson_data: dict,
    class_mapping: dict,
    img_height: int,
    img_width: int,
    min_surface: int,
    max_surface: int,
    task: str,
) -> str:
    """
    Creates the labels of a slice of an image from the geojson datan the type of task at hand and the given filters of the surface of the labels

    Arguments:
        slice_idx: the index of the current slice in the image
        geojson_data: a dictionary containing the geojson data for the current image
        class_mapping: a dictionary containing the mapping from class names to class value
        img_height: a scalar denoting the height of an image in pixel
        img_width: a scalar denoting the width of an image in pixel
        min_surface: a scalar denoting the minimum bounding box area authorized in the labels
        max_surface: a scalar denoting the maximum bounding box area authorized in the labels
        task: a string denoting the type of task (object detection "box" or instance segmentation "seg")

    Returns:
        slice_labels: a string containing the labels of a given slice of a given image
    """
    # Retrieve the features located on the chosen slice
    slice_features = [
        feature
        for feature in geojson_data["features"]
        if feature["geometry"]["plane"]["t"] == slice_idx
    ]

    # Process the labels only if the slice has some features
    if slice_features != []:

        slice_labels = []
        # Iterate through the found labels in the geojson data
        for feature in slice_features:
            # Get the class and the coordinates of the annotated object
            classification = feature["properties"]["classification"]["name"]
            class_id = class_mapping.get(classification, class_mapping["Non-Living"])
            coordinates = feature["geometry"]["coordinates"][0]

            x_coords = [point[0] for point in coordinates]
            y_coords = [point[1] for point in coordinates]

            # Get the minimum, maximum and center coordinates along x and y
            min_x, max_x = int(min(x_coords) - 39), int(max(x_coords) - 39)
            min_y, max_y = int(min(y_coords) - 95), int(max(y_coords) - 95)

            x_center = (min_x + max_x) / 2 / img_width
            y_center = (min_y + max_y) / 2 / img_height
            width = (max_x - min_x) / img_width
            height = (max_y - min_y) / img_height
            # Apply the filtering on the bounding box area
            if max_surface > (max_x - min_x) * (max_y - min_y) >= min_surface:
                # Add the label following the box format for object detection : class_id x_center y_center width height
                if task == "box":
                    slice_labels.append(
                        f"{class_id} {x_center} {y_center} {width} {height}"
                    )
                # Add the label following the segmentation format for instance segmentation : class_id x_0 y_0 x_1 y_1 ... x_n y_n
                elif task == "seg":
                    line = (
                        str(class_id)
                        + " "
                        + " ".join(
                            [
                                f"{(point[0] - 39)/img_width} {(point[1] - 95)/img_height}"  # 39 and 95 are cropping values to remove the white padding
                                for point in coordinates
                            ]
                        )
                    )
                    slice_labels.append(line)
                else:
                    raise ("Wrong task specified. Only box and seg available")
        # Combine all labels in a single string with a label per line
        if slice_labels != []:
            slice_labels = "\n".join(slice_labels)
        else:
            slice_labels = ""

    else:

        slice_labels = ""

    return slice_labels


def adjust_contrast(slice_array: np.array, min_range: int, max_range: int) -> np.array:
    """
    Adjusts the contrast of the image to a given range, expects grayscale image

    Arguments:
        slice_array: a numpy array of a slice of a grayscale image
        min_range: a scalar denoting the minimum pixel value accepted from the original image
        max_range: a scalar denoting the maximum pixel value accepted from the original image

    Returns:
        adjusted_slice: a numpy array of a slice of a grayscale image after contrast adjustment
    """
    cliped_array = np.clip(slice_array, min_range, max_range)
    adjusted_slice = 255 * ((cliped_array - min_range) / (max_range - min_range))
    return adjusted_slice


def preprocess_image(
    slice_array: np.array,
    kernel_size: tuple = (4, 4),
    kernel_size_blur: tuple = (5, 5),
    sigma: int = 1,
    iterations: int = 1,
    thresh: int = 0,
    process_iter: int = 1,
) -> np.array:
    """
    Preprocesses a slice by applying Gaussian Blur, Binarization, Dilation and Erosion to remove noise

    Arguments:
        slice_array: a numpy array of a slice of a grayscale image
        kernel_size: a tuple of scalar denoting the size of the kernel used when applying Dilation and Erosion
        kernel_size_blur: a tuple of scalar denoting the size of the kernel used when applying the Gaussian Blur
        sigma: a scalar denoting the standard deviation used for the gaussian kernel in the Gaussian Blur
        iteratons: a scalar denoting the number of iterations to apply at once for Dilation and Erosion
        thresh: a scalar denoting the threshold used for binarization
        process_iter: a scalar denoting the number of iterations of preprocessing to apply

    Returns:
        preprocess_array: a numpy array of a slice after preprocessing
    """
    preprocess_array = slice_array.copy()
    kernel_blur = np.ones(kernel_size_blur, np.uint8)
    kernel = np.ones(kernel_size, np.uint8)
    for _ in range(process_iter):
        blurred_array = cv2.GaussianBlur(
            preprocess_array.astype(np.uint8), kernel_blur, sigma
        )
        _, binarized_array = cv2.threshold(
            blurred_array.astype(np.uint8),
            thresh,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        dilated = cv2.dilate(
            binarized_array.astype(np.uint8), kernel, iterations=iterations
        )
        cleaned_array = cv2.erode(
            dilated.astype(np.uint8), kernel, iterations=iterations
        )
        preprocess_array = np.where(cleaned_array == 255, 255, preprocess_array)
    return preprocess_array


def process_geojson_file(
    file_name: str,
    tif_dir: str,
    geojson_dir: str,
    output_label_dir: str,
    class_mapping: dict,
    min_range: int,
    max_range: int,
    min_surface: int,
    max_surface: int,
    preprocess: bool,
    save_img: bool,
    task: str,
):
    """
    Processes a single geoJSON file and its corresponding .tif file

    Arguments:
        file_name: a string denoting the name of the case to process to retrieve the correct tif and geojson files
        tif_dir: a string denoting the path to the directory with the tif images
        geojson_dir: a string denoting the path to the directory with the geojson files
        output_label_dir: a string denoting the path to the directory where the labels and slices will be saved
        class_mapping: a dictionary containing the mapping from class names to class value
        min_range: a scalar denoting the minimum pixel value accepted from the original image for contrast adjustment
        max_range: a scalar denoting the maximum pixel value accepted from the original image for contrast adjustment
        min_surface: a scalar denoting the minimum bounding box area authorized in the labels
        max_surface: a scalar denoting the maximum bounding box area authorized in the labels
        preprocess: a boolean denoting if the slices need to be preprocessed
        save_img: a boolean denoting if the images need to be saved
        task: a string denoting the type of task (object detection "box" or instance segmentation "seg")
    """
    tif_path = join(tif_dir, file_name + ".tif")
    geojson_path = join(geojson_dir, file_name + ".geojson")

    # Check if the corresponding .tif file exists
    if not os.path.exists(tif_path):
        raise ValueError(f"`{tif_path}` file not found.")

    # Load geoJSON data
    with open(geojson_path, "r") as file:
        geojson_data = json.load(file)

    # Open the .tif file and process each frame specified in the geoJSON
    img_stack = tiff.imread(tif_path)
    for slice_idx, slice_array in enumerate(img_stack):
        # Crop the image to remove white padding from the original image
        cropped_array = slice_array[95:-137, 39:-57]
        img_height, img_width = cropped_array.shape
        # Get the labels in the expected format
        slice_labels = process_slice(
            slice_idx,
            geojson_data,
            class_mapping,
            img_height,
            img_width,
            min_surface,
            max_surface,
            task,
        )
        # Adjust the contrast and preprocess the slice
        adjust_slice_array = adjust_contrast(cropped_array, min_range, max_range)
        if preprocess:
            preprocessed_array = preprocess_image(
                adjust_slice_array, (4, 4), (5, 5), 1, 1, 0, 1
            )
        else:
            preprocessed_array = adjust_slice_array.astype(np.uint8)

        # Create the final path for the slice and the corresponding labels
        image_path = join(
            dirname(output_label_dir),
            "images",
            f"{file_name}_ctrst-{min_range}-{max_range}_slice-{slice_idx}.jpg",
        )
        label_path = join(
            output_label_dir,
            f"{file_name}_ctrst-{min_range}-{max_range}_slice-{slice_idx}.txt",
        )

        # Save the slice and the corresponding labels
        if save_img:
            cv2.imwrite(image_path, preprocessed_array)
        if slice_labels != "":
            with open(label_path, "w") as label_file:
                label_file.write(slice_labels)


def main(
    tif_dir: str,
    geojson_dir: str,
    output_label_dir: str,
    class_mapping: dict,
    min_range: int,
    max_range: int,
    split_value_surface: int,
    preprocess: bool,
    task: str,
):
    """
    Main function loading all the images and creating the labels for each of their slices

    Arguments:
        tif_dir: a string denoting the path to the directory with the tif images
        geojson_dir: a string denoting the path to the directory with the geojson files
        output_label_dir: a string denoting the path to the directory where the labels and slices will be saved
        class_mapping: a dictionary containing the mapping from class names to class value
        min_range: a scalar denoting the minimum pixel value accepted from the original image for contrast adjustment
        max_range: a scalar denoting the maximum pixel value accepted from the original image for contrast adjustment
        split_value_surface: a scalar denoting the bounding box area on which the labels will be split between "small" and "big" objects
        preprocess: a boolean denoting if the slices need to be preprocessed
        task: a string denoting the type of task (object detection "box" or instance segmentation "seg")
    """
    if task not in ["box", "seg"]:
        raise ValueError(f"Wrong task specified : {task}. Only box and seg available.")
    if split_value_surface < 0:
        raise ValueError("Surface limits can't be negative.")
    if split_value_surface == 0:
        raise ValueError(
            "split_value_surface equals 0 so the small objects labels would be empty."
        )
    if min_range < 0 or max_range > 255:
        warnings.warn(
            f"({min_range},{max_range}) range provided for contrast adjustment while the pixel values range is (0,255). Nothing will happen.",
            UserWarning,
        )
    if not exists(tif_dir):
        raise ValueError(f"{tif_dir} provided for TIF directory path does not exist")
    if not exists(geojson_dir):
        raise ValueError(
            f"{geojson_dir} provided for GeoJSON directory path does not exist"
        )
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(join(output_label_dir, "small_labels"), exist_ok=True)
    os.makedirs(join(output_label_dir, "big_labels"), exist_ok=True)
    os.makedirs(join(output_label_dir, "images"), exist_ok=True)
    geojson_files = get_geojson_files(geojson_dir)
    for file_name in tqdm(geojson_files):
        process_geojson_file(
            file_name,
            tif_dir,
            geojson_dir,
            join(output_label_dir, "big_labels"),
            class_mapping,
            min_range,
            max_range,
            split_value_surface,
            np.inf,
            preprocess,
            True,
            task,
        )
        process_geojson_file(
            file_name,
            tif_dir,
            geojson_dir,
            join(output_label_dir, "small_labels"),
            class_mapping,
            min_range,
            max_range,
            0,
            split_value_surface,
            preprocess,
            False,
            task,
        )
    print(f"Data saved at {output_label_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_geojson", type=str, default="./data/data_raw/geojson_file"
    )
    parser.add_argument("--path_images", type=str, default="./data/data_raw/images")
    parser.add_argument("--path_output", type=str, default="./data/data_labeled")
    parser.add_argument("--min_contrast", type=int, default=0)
    parser.add_argument("--max_contrast", type=int, default=255)
    parser.add_argument("--split_value_surface", type=int, default=200)
    parser.add_argument("--task", type=str, default="seg")
    parser.add_argument("--preprocess", action="store_true", default=False)
    args = parser.parse_args()

    class_mapping = {"Living": 0, "Non-Living": 1, "Bubble": 2}

    path_output_labeled_dir = f"{args.path_output}/ctrst-{args.min_contrast}-{args.max_contrast}_srfc-{args.split_value_surface}_prcs-{int(args.preprocess)}_{args.task}"

    main(
        args.path_images,
        args.path_geojson,
        path_output_labeled_dir,
        class_mapping,
        args.min_contrast,
        args.max_contrast,
        args.split_value_surface,
        args.preprocess,
        args.task,
    )
