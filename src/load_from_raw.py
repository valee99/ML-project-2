import os
import json
import numpy as np
import tifffile as tiff
from PIL import Image
import argparse


def get_geojson_files(geojson_dir: str) -> set:
    """
    Get set of geoJSON file names without extensions.
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
) -> str:

    slice_features = [
        feature
        for feature in geojson_data["features"]
        if feature["geometry"]["plane"]["t"] == slice_idx
    ]

    if slice_features != []:

        slice_labels = []
        for feature in slice_features:
            classification = feature["properties"]["classification"]["name"]
            class_id = class_mapping.get(classification, class_mapping["Non-Living"])
            coordinates = feature["geometry"]["coordinates"][0]

            x_coords = [point[0] for point in coordinates]
            y_coords = [point[1] for point in coordinates]
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))

            x_center = (min_x + max_x) / 2 / img_width
            y_center = (min_y + max_y) / 2 / img_height
            width = (max_x - min_x) / img_width
            height = (max_y - min_y) / img_height
            slice_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
        slice_labels = "\n".join(slice_labels)

    else:

        slice_labels = ""

    return slice_labels


def adjust_contrast(
    slice_array: np.ndarray, min_range: int, max_range: int
) -> np.ndarray:

    slice_min, slice_max = np.min(slice_array), np.max(slice_array)
    normalized_slice = (slice_array - slice_min) / (slice_max - slice_min)
    adjusted_slice_array = min_range + normalized_slice * (max_range - min_range)
    return adjusted_slice_array


def process_geojson_file(
    file_name: str,
    tif_dir: str,
    geojson_dir: str,
    output_label_dir: str,
    class_mapping: dict,
    min_range: int,
    max_range: int,
):
    """
    Process a single geoJSON file and its corresponding .tif file.
    """
    tif_path = os.path.join(tif_dir, file_name + ".tif")
    geojson_path = os.path.join(geojson_dir, file_name + ".geojson")

    # Check if the corresponding .tif file exists
    if not os.path.exists(tif_path):
        raise (f"Error: `{tif_path}` file not found.")

    # Load geoJSON data
    with open(geojson_path, "r") as file:
        geojson_data = json.load(file)

    # Open the .tif file and process each frame specified in the geoJSON
    img_stack = tiff.imread(tif_path)
    for slice_idx, slice_array in enumerate(img_stack):
        img_height, img_width = slice_array.shape
        slice_labels = process_slice(
            slice_idx, geojson_data, class_mapping, img_height, img_width
        )
        adjust_slice_array = adjust_contrast(slice_array, min_range, max_range)
        adjusted_image = Image.fromarray(adjust_slice_array.astype(np.uint8))

        image_path = os.path.join(
            output_label_dir,
            f"{file_name}_ctrst-{min_range}-{max_range}_slice-{slice_idx}.jpg",
        )
        label_path = os.path.join(
            output_label_dir,
            f"{file_name}_ctrst-{min_range}-{max_range}_slice-{slice_idx}.txt",
        )

        adjusted_image.save(image_path)
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
):

    os.makedirs(output_label_dir, exist_ok=True)
    geojson_files = get_geojson_files(geojson_dir)
    for file_name in geojson_files:
        process_geojson_file(
            file_name,
            tif_dir,
            geojson_dir,
            output_label_dir,
            class_mapping,
            min_range,
            max_range,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for processing files.")
    parser.add_argument(
        "--path_geojson", type=str, default="./data/data_raw/geojson_file"
    )
    parser.add_argument("--path_images", type=str, default="./data/data_raw/images")
    parser.add_argument("--path_output", type=str, default="./data/data_labeled")
    parser.add_argument("--min_contrast", type=int, default=210)
    parser.add_argument("--max_contrast", type=int, default=255)
    args = parser.parse_args()

    CLASS_MAPPING = {"Living": 0, "Non-Living": 1, "Bubble": 2}

    PATH_OUTPUT_LABELED_DIR = (
        f"{args.path_output}/ctrst-{args.min_contrast}-{args.max_contrast}"
    )

    main(
        args.path_images,
        args.path_geojson,
        PATH_OUTPUT_LABELED_DIR,
        CLASS_MAPPING,
        args.min_contrast,
        args.max_contrast,
    )
