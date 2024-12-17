import os
import random
import shutil
import argparse
from glob import glob
from os.path import basename, join, exists
import random
import yaml
from tqdm import tqdm
import warnings


def split_files(path_labeled_dir: str, splits: dict, all_slices: bool) -> dict:
    """Splits the files to create train/val/test split. The dataset is split on the images to avoid having different slices of an image across different splits.
    The split method also takes into account the labels of small and big objects to make sure the models will be trained on same images and slices.

    Arguments:
        path_labeled_dir: a string denoting the path to the directory with images and labels
        splits: a directory storing the chosen splits
        all_slices: a boolean denoting if the test split should contain all the slices of the chosen images, useful for visual evaluation

    Returns:
        files_split: a directory storing the file names for each split
    """
    # Retrieve the path to the images and labels for small and big objects
    images_path = glob(join(path_labeled_dir, "images", "*.jpg"))
    labels_big_path = glob(join(path_labeled_dir, "big_labels", "*.txt"))
    labels_small_path = glob(join(path_labeled_dir, "small_labels", "*.txt"))

    # Retrieve the individual images with small and big objects labels
    images_with_big_labels = list(
        set([basename(path).split("_")[0] for path in labels_big_path])
    )
    images_with_small_labels = list(
        set([basename(path).split("_")[0] for path in labels_small_path])
    )

    # Select if the split will be done on the small or big objects dataset
    if len(images_with_big_labels) > len(images_with_small_labels):
        n_images_train = int(splits["train"] * len(images_with_big_labels))
        n_images_val = int(splits["val"] * len(images_with_big_labels))
        images_list = images_with_big_labels
    else:
        n_images_train = int(splits["train"] * len(images_with_small_labels))
        n_images_val = int(splits["val"] * len(images_with_small_labels))
        images_list = images_with_small_labels

    # Setting seed for reproducibility
    random.seed(42)
    random.shuffle(sorted(images_list))

    # Split the images across the train/val/test splits
    train_images = images_list[:n_images_train]
    val_images = images_list[n_images_train : n_images_train + n_images_val]
    test_images = images_list[n_images_train + n_images_val :]

    print("Images used for training : ", train_images)
    print("Images used for validation : ", val_images)
    print("Images used for testing : ", test_images)

    # Retrieve the file names from the chosen images
    train_files = [
        basename(path).split(".")[0]
        for path in labels_big_path
        if basename(path).split("_")[0] in train_images
    ] + [
        basename(path).split(".")[0]
        for path in labels_small_path
        if basename(path).split("_")[0] in train_images
    ]
    val_files = [
        basename(path).split(".")[0]
        for path in labels_big_path
        if basename(path).split("_")[0] in val_images
    ] + [
        basename(path).split(".")[0]
        for path in labels_small_path
        if basename(path).split("_")[0] in val_images
    ]
    test_files = [
        basename(path).split(".")[0]
        for path in labels_big_path
        if basename(path).split("_")[0] in test_images
    ] + [
        basename(path).split(".")[0]
        for path in labels_small_path
        if basename(path).split("_")[0] in test_images
    ]

    # If the the split is empty if a small test share is given, make sure to have at least one image taken from the validation split
    if test_images == [] and splits["test"] > 0:
        test_images.append(val_images.pop())

    # Add all the slices from the chosen images in the test split if all_slices if True
    if all_slices:
        test_files = [
            basename(path).split(".")[0]
            for path in images_path
            if basename(path).split(".")[0].split("_")[0] in test_images
        ]

    return {"train": train_files, "val": val_files, "test": test_files}


def move_files(
    files: dict, path_labeled_dir: str, path_split_dir: str, label_type: str
):
    """Copies the files from the original directory to the split directory

    Arguments:
        files: a directory storing the file names for each split
        path_labeled_dir: a string denoting the path to the original directory
        path_split_dir: a string denoting the path to the split directory
        label_type: a string denoting if the files are from the small or big labels
    """
    for split, files_list in tqdm(files.items()):

        for file_name in files_list:

            image_file = join(path_labeled_dir, "images", file_name + ".jpg")
            split_image_file = join(path_split_dir, split, "images", file_name + ".jpg")

            label_file = join(path_labeled_dir, label_type, file_name + ".txt")
            split_label_file = join(path_split_dir, split, "labels", file_name + ".txt")
            if split != "test":
                if os.path.exists(label_file):
                    shutil.copy(image_file, split_image_file)
                    shutil.copy(label_file, split_label_file)
            else:
                shutil.copy(image_file, split_image_file)
                if os.path.exists(label_file):
                    shutil.copy(label_file, split_label_file)


def create_yaml(dataset_name: str):
    """Creates the YAML file that will be used for the YOLO model to define the data augmentation and the path to the files

    Arguments:
        dataset_name: a string denoting the name of the dataset
    """
    content = {
        "path": dataset_name,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 3,
        "names": {0: "Living", 1: "Non-Living", 2: "Bubble"},
    }

    # Setting all the data augmentations as they will be done by our script later
    augmentations = {
        "mosaic": False,
        "mixup": False,
        "hsv_h": 0,
        "hsv_s": 0,
        "hsv_v": 0,
        "fliplr": 0,
        "flipud": 0,
        "rotate": 0,
        "scale": 0,
        "translate": 0,
        "cutout": False,
    }

    with open(f"configs/{dataset_name}.yaml", "w") as yaml_file:
        yaml_file.write(
            "# https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#21-create-datasetyaml\n\n"
        )
        yaml_file.write("# Paths\n")
        yaml.dump(content, yaml_file, default_flow_style=False, sort_keys=False)
        yaml_file.write("\naugmentation:")
        yaml.dump(augmentations, yaml_file, default_flow_style=False, sort_keys=False)


def main(
    path_labeled_dir: str,
    path_split_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    all_slices: bool,
):
    """Main function creating the files splits, moving them to the split directory and creating the YAML file for the "big" and "small" datasets

    Arguments:
        path_labeled_dir: a string denoting the path to the directory with images and labels
        path_split_dir: a string denoting the path to the split directory
        train_ratio: a scalar denoting the train ratio
        val_ratio: a scalar denoting the test ratio
        test_ratio: a scalar denoting the val ratio
        all_slices: a boolean denoting if the test split should contain all the slices of the chosen images, useful for visual evaluation
    """
    if not exists(path_labeled_dir):
        raise ValueError(
            f"{path_labeled_dir} provided for label directory path does not exist"
        )
    if not exists(path_split_dir):
        raise ValueError(
            f"{path_split_dir} provided for split directory path does not exist"
        )
    if train_ratio <= 0:
        raise ValueError(
            f"{train_ratio} provided for train_ratio. This value must be strictly positive."
        )

    splits = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    files = split_files(path_labeled_dir, splits, all_slices)

    for label_type in ["big_labels", "small_labels"]:
        dataset_path = join(
            path_split_dir, basename(path_labeled_dir) + "_" + label_type
        )
        os.makedirs(dataset_path, exist_ok=True)

        for split, ratio in splits.items():
            if ratio > 0:
                os.makedirs(join(dataset_path, split, "images"), exist_ok=True)
                os.makedirs(join(dataset_path, split, "labels"), exist_ok=True)

        create_yaml(basename(path_labeled_dir) + "_" + label_type)
        print(f"YAML file created !")

        move_files(files, path_labeled_dir, dataset_path, label_type)
        print(
            f"All files divided in train/val/test splits for dataset {basename(path_labeled_dir)+"_"+label_type}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_labeled",
        type=str,
    )
    parser.add_argument("--path_split", type=str, default="./data/data_split")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--all_slices", action="store_true", default=False)
    args = parser.parse_args()

    main(
        args.path_labeled,
        args.path_split,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.all_slices,
    )
