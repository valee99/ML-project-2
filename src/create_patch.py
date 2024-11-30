import cv2
import numpy as np
import os
from os.path import join, basename
from glob import glob
import argparse
from tqdm import tqdm


def filter_labels(
    labels: list,
    img_height: int,
    img_width: int,
    x_min_patch: int,
    x_max_patch: int,
    y_min_patch: int,
    y_max_patch: int,
    task: str,
):

    patch_labels = []

    patch_width = x_max_patch - x_min_patch
    patch_height = y_max_patch - y_min_patch

    for label in labels:
        if task == "box":
            points = list(map(float, label.split()))
            x_top = min(
                int(points[1] * img_width + (points[3] * img_width) // 2), img_width
            )
            y_left = max(int(points[2] * img_height - (points[4] * img_height) // 2), 0)
            x_bot = max(int(points[1] * img_width - (points[3] * img_width) // 2), 0)
            y_right = min(
                int(points[2] * img_height + (points[4] * img_height) // 2), img_height
            )

            if (
                (x_top < x_max_patch)
                and (x_bot >= x_min_patch)
                and (y_right < y_max_patch)
                and (y_left >= y_min_patch)
            ):
                class_id = int(points[0])
                x_center = (x_bot + x_top) / 2 / patch_width
                y_center = (y_left + y_right) / 2 / patch_height
                width = (x_top - x_bot) / patch_width
                height = (y_right - y_left) / patch_height
                patch_labels.append(
                    f"{class_id} {x_center} {y_center} {width} {height}"
                )

        elif task == "seg":
            points = list(map(float, label.split()[1:]))

            class_id = int(label[0])
            x_coords = []
            y_coords = []
            for i in range(len(points), 2):
                x_coords.append(img_width * points[i])
                y_coords.append(img_height * points[i + 1])

            x_top = max(x_coords)
            x_bot = min(x_coords)
            y_right = max(y_coords)
            y_left = max(y_coords)

            if (
                (x_top < x_max_patch)
                and (x_bot >= x_min_patch)
                and (y_right < y_max_patch)
                and (y_left >= y_min_patch)
            ):
                line = (
                    str(class_id)
                    + " "
                    + " ".join(
                        [
                            f"{x/patch_width} {y/patch_height}"
                            for x, y in zip(x_coords, y_coords)
                        ]
                    )
                )
                patch_labels.append(line)

    if patch_labels != []:
        return "\n".join(patch_labels)

    else:
        return ""


def update_patches(
    patches_list: list,
    patches_labels_list: list,
    img_array: np.array,
    labels: list,
    img_height: int,
    img_width: int,
    y_min_patch: int,
    y_max_patch: int,
    x_min_patch: int,
    x_max_patch: int,
    task: str,
):
    patches_list.append(img_array[y_min_patch:y_max_patch, x_min_patch:x_max_patch])

    patches_labels = filter_labels(
        labels,
        img_height,
        img_width,
        x_min_patch,
        x_max_patch,
        y_min_patch,
        y_max_patch,
        task,
    )
    patches_labels_list.append(patches_labels)


def get_patches(
    n_rows_patch: int,
    n_cols_patch: int,
    img_array: np.array,
    path_label: str,
    split: str,
    task: str,
):
    reg_patches = []
    reg_patches_labels = []

    inter_patches = []
    inter_patches_labels = []

    hor_border_patches = []
    hor_patches_labels = []

    ver_border_patches = []
    ver_patches_labels = []

    with open(path_label, "r") as label_file:
        labels = label_file.readlines()

    img_height, img_width = img_array.shape
    img_height_patch, img_width_patch = (
        img_height // n_rows_patch,
        img_width // n_cols_patch,
    )

    for i in range(n_rows_patch):
        for j in range(n_cols_patch):

            x_min_patch, x_max_patch = j * img_width_patch, (j + 1) * img_width_patch
            y_min_patch, y_max_patch = i * img_height_patch, (i + 1) * img_height_patch

            y_half_step = int(0.5 * img_height_patch)
            x_half_step = int(0.5 * img_width_patch)

            if split in ["train", "val"]:

                update_patches(
                    reg_patches,
                    reg_patches_labels,
                    img_array,
                    labels,
                    img_height,
                    img_width,
                    y_min_patch,
                    y_max_patch,
                    x_min_patch,
                    x_max_patch,
                    task,
                )

            elif split == "test":

                if i < n_rows_patch - 1 and j < n_cols_patch - 1:

                    update_patches(
                        inter_patches,
                        inter_patches_labels,
                        img_array,
                        labels,
                        img_height,
                        img_width,
                        y_min_patch + y_half_step,
                        y_max_patch + y_half_step,
                        x_min_patch + x_half_step,
                        x_max_patch + x_half_step,
                        task,
                    )

                if (i == 0 or i == n_rows_patch - 1) and j < n_cols_patch - 1:

                    update_patches(
                        hor_border_patches,
                        hor_patches_labels,
                        img_array,
                        labels,
                        img_height,
                        img_width,
                        y_min_patch,
                        y_max_patch,
                        x_min_patch + x_half_step,
                        x_max_patch + x_half_step,
                        task,
                    )

                if (j == 0 or j == n_cols_patch - 1) and i < n_rows_patch - 1:

                    update_patches(
                        inter_patches,
                        inter_patches_labels,
                        img_array,
                        labels,
                        img_height,
                        img_width,
                        y_min_patch + y_half_step,
                        y_max_patch + y_half_step,
                        x_min_patch,
                        x_max_patch,
                        task,
                    )

    if split in ["train", "val"]:
        return reg_patches, reg_patches_labels

    else:
        test_patches = inter_patches + hor_border_patches + ver_border_patches
        test_labels = inter_patches_labels + hor_patches_labels + ver_patches_labels

        return test_patches, test_labels


def save_patches(
    patches: list, labels: list, image_name: str, path_dataset: str, split: str
):

    n_patches = len(patches)

    for patch_idx, patch in enumerate(patches):

        label = labels[patch_idx]
        if label == "":
            continue

        label_path = join(
            path_dataset,
            split,
            "labels",
            image_name + f"_patch-{str(patch_idx)}-{str(n_patches)}.txt",
        )
        with open(label_path, "w") as label_file:
            label_file.write(label)

        patch_path = join(
            path_dataset,
            split,
            "images",
            image_name + f"_patch-{str(patch_idx)}-{str(n_patches)}.jpg",
        )
        cv2.imwrite(patch_path, patch)


def main(n_rows_patch: int, n_cols_patch: int, path_dataset: str, task: str):

    for split in ["train", "val", "test"]:

        images_path = glob(join(path_dataset), split, "images", "*.jpg")

        for img_path in tqdm(images_path):

            image_name = basename(img_path).split(".")[0]
            image_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            label_path = img_path.replace("images", "labels").replace("jpg", "txt")

            patches, labels = get_patches(
                n_rows_patch, n_cols_patch, image_array, label_path, split, task
            )

            save_patches(patches, labels, image_name, path_dataset, split)

            os.remove(img_path)
            os.remove(label_path)

        print(f"All patches created for {split} split !")

    print(f"All patches created and saved for dataset {basename(path_dataset)} !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for processing files.")
    parser.add_argument("--path_dataset", type=str)
    parser.add_argument("--task", type=str, default="seg")
    parser.add_argument("--n_rows_patch", type=int, default=7)
    parser.add_argument("--n_cols_patch", type=int, default=7)
    args = parser.parse_args()

    main(args.n_rows_patch + 1, args.n_cols_patch + 1, args.path_dataset, args.task)
