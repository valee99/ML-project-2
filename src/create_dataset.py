import os
import random
import shutil
import argparse
from glob import glob
from os.path import basename
import random
import yaml
from tqdm import tqdm


def split_files(path_labeled_dir: str, splits: dict, all_slices: bool) -> dict:

    images_path = glob(os.path.join(path_labeled_dir, "*.jpg"))
    labels_path = glob(os.path.join(path_labeled_dir, "*.txt"))

    files_with_labels = [basename(path).split(".")[0] for path in labels_path]

    # Setting seed for reproducibility
    random.seed(42)
    random.shuffle(files_with_labels)

    n_files = len(files_with_labels)
    train_files = files_with_labels[: int(n_files * splits["train"])]
    val_files = files_with_labels[
        int(n_files * splits["train"]) : int(n_files * splits["train"])
        + int(n_files * splits["val"])
    ]
    test_files = files_with_labels[
        int(n_files * splits["train"]) + int(n_files * splits["val"]) :
    ]

    if all_slices:
        img_test = list(set([file_path.split("_")[0] for file_path in test_files]))
        test_files = [
            basename(path).split(".")[0]
            for path in images_path
            if basename(path).split(".")[0].split("_")[0] in img_test
        ]

    return {"train": train_files, "val": val_files, "test": test_files}


def move_files(files: dict, path_labeled_dir: str, path_split_dir: str):

    for split, files_list in tqdm(files.items()):

        for file_name in files_list:

            image_file = os.path.join(path_labeled_dir, file_name + ".jpg")
            split_image_file = os.path.join(
                path_split_dir, split, "images", file_name + ".jpg"
            )
            shutil.copy(image_file, split_image_file)

            label_file = os.path.join(path_labeled_dir, file_name + ".txt")
            split_label_file = os.path.join(
                path_split_dir, split, "labels", file_name + ".txt"
            )

            if os.path.exists(label_file):
                shutil.copy(label_file, split_label_file)


def create_yaml(dataset_name: str):
    content = {
        "path": dataset_name,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 3,
        "names": {0: "Living", 1: "Non-Living", 2: "Bubble"},
    }

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

    dataset_path = os.path.join(path_split_dir, basename(path_labeled_dir))
    os.makedirs(dataset_path, exist_ok=True)

    splits = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    for split, ratio in splits.items():
        if ratio > 0:
            os.makedirs(os.path.join(dataset_path, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_path, split, "labels"), exist_ok=True)

    files = split_files(path_labeled_dir, splits, all_slices)

    move_files(files, path_labeled_dir, dataset_path)
    print(
        f"All files divided in train/val/test splits for dataset {basename(path_labeled_dir)}"
    )

    create_yaml(basename(path_labeled_dir))
    print(f"YAML file created !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for processing files.")
    parser.add_argument(
        "--path_labeled", type=str, default="./data/data_labeled/ctrst-210-255"
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
