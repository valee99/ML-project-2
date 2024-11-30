import os
import json
import numpy as np
import tifffile as tiff
import argparse
import cv2
from tqdm import tqdm


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
    min_surface: int,
    max_surface: int,
    task: str,
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

            min_x, max_x = int(min(x_coords) - 39), int(max(x_coords) - 39)
            min_y, max_y = int(min(y_coords) - 95), int(max(y_coords) - 95)

            x_center = (min_x + max_x) / 2 / img_width
            y_center = (min_y + max_y) / 2 / img_height
            width = (max_x - min_x) / img_width
            height = (max_y - min_y) / img_height
            if max_surface > (max_x - min_x) * (max_y - min_y) >= min_surface:
                if task == "box":
                    slice_labels.append(
                        f"{class_id} {x_center} {y_center} {width} {height}"
                    )
                elif task == "seg":
                    line = (
                        str(class_id)
                        + " "
                        + " ".join(
                            [
                                f"{(point[0] - 39)/img_width} {(point[1] - 95)/img_height}"
                                for point in coordinates
                            ]
                        )
                    )
                    slice_labels.append(line)
                else:
                    print("Wrong task specified. Only box and seg available")
        if slice_labels != []:
            slice_labels = "\n".join(slice_labels)
        else:
            slice_labels = ""

    else:

        slice_labels = ""

    return slice_labels


def adjust_contrast(
    slice_array: np.ndarray, min_range: int, max_range: int
) -> np.ndarray:

    cliped_array = np.clip(slice_array, min_range, max_range)
    adjusted_slice = 255 * ((cliped_array - min_range) / (max_range - min_range))
    return adjusted_slice


def preprocess_image(
    slice_array: np.array,
    kernel_size: tuple = (4, 4),
    sigma: int = 1,
    iterations: int = 1,
    thresh: int = 0,
    process_iter: int = 1,
) -> np.array:
    preprocess_array = slice_array.copy()
    for _ in range(process_iter):
        blurred_array = cv2.GaussianBlur(
            preprocess_array.astype(np.uint8), (5, 5), sigma
        )
        _, binarized_array = cv2.threshold(
            blurred_array.astype(np.uint8),
            thresh,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        kernel = np.ones(kernel_size, np.uint8)
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
    task: str,
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
        cropped_array = slice_array[95:-137, 39:-57]
        img_height, img_width = cropped_array.shape
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
        adjust_slice_array = adjust_contrast(cropped_array, min_range, max_range)
        if preprocess:
            preprocessed_array = preprocess_image(
                adjust_slice_array, (4, 4), 1, 1, 0, 1
            )
        else:
            preprocessed_array = adjust_slice_array.astype(np.uint8)

        image_path = os.path.join(
            output_label_dir,
            f"{file_name}_ctrst-{min_range}-{max_range}_slice-{slice_idx}.jpg",
        )
        label_path = os.path.join(
            output_label_dir,
            f"{file_name}_ctrst-{min_range}-{max_range}_slice-{slice_idx}.txt",
        )

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
    min_surface: int,
    max_surface: int,
    preprocess: bool,
    task: str,
):

    os.makedirs(output_label_dir, exist_ok=True)
    geojson_files = get_geojson_files(geojson_dir)
    for file_name in tqdm(geojson_files):
        process_geojson_file(
            file_name,
            tif_dir,
            geojson_dir,
            output_label_dir,
            class_mapping,
            min_range,
            max_range,
            min_surface,
            max_surface,
            preprocess,
            task,
        )
    print(f"Data saved at {output_label_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for processing files.")
    parser.add_argument(
        "--path_geojson", type=str, default="./data/data_raw/geojson_file"
    )
    parser.add_argument("--path_images", type=str, default="./data/data_raw/images")
    parser.add_argument("--path_output", type=str, default="./data/data_labeled")
    parser.add_argument("--min_contrast", type=int, default=0)
    parser.add_argument("--max_contrast", type=int, default=255)
    parser.add_argument("--min_surface", type=int, default=0)
    parser.add_argument("--max_surface", type=int, default=1e12)
    parser.add_argument("--task", type=str, default="seg")
    parser.add_argument("--preprocess", action="store_true", default=False)
    args = parser.parse_args()

    CLASS_MAPPING = {"Living": 0, "Non-Living": 1, "Bubble": 2}

    PATH_OUTPUT_LABELED_DIR = f"{args.path_output}/ctrst-{args.min_contrast}-{args.max_contrast}_srfc-{args.min_surface}-{args.max_surface}_prcs-{int(args.preprocess)}_{args.task}"

    main(
        args.path_images,
        args.path_geojson,
        PATH_OUTPUT_LABELED_DIR,
        CLASS_MAPPING,
        args.min_contrast,
        args.max_contrast,
        args.min_surface,
        args.max_surface,
        args.preprocess,
        args.task,
    )
