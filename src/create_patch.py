import cv2
import numpy as np
import os
from os.path import join, basename, exists
from glob import glob
import argparse
from tqdm import tqdm
import warnings


def filter_labels(
    labels: list,
    img_height: int,
    img_width: int,
    x_min_patch: int,
    x_max_patch: int,
    y_min_patch: int,
    y_max_patch: int,
    task: str,
) -> str:
    """Filters the labels to keep the annotated objects entirely in the patch

    Arguments:
        labels: a list of string with the annotated objects labels
        img_height: a scalar denoting the height of the image in pixels
        img_width: a scalar denoting the width of the image in pixels
        x_min_path: a scalar denoting the minimum x coordinate of the patch
        x_max_patch: a scalar denoting the maximum x coordinate of the patch
        y_min_patch: a scalar denoting the minimum y coordinate of the patch
        y_max_patch: a scalar denoting the maximum y coordinate of the patch
        task: a string denoting the chosen task between object detection "box" or instance segmentation "seg"

    Returns:
        filtered_labels: a string containing the labels of the patch
    """
    patch_labels = []

    # Get the width and height of the patch
    patch_width = x_max_patch - x_min_patch
    patch_height = y_max_patch - y_min_patch

    # Iterate through the labels to check if they are entirely in the patch
    for label in labels:
        if task == "box":
            # Retrieve the center coordinates, width and height of the bounding box
            points = list(map(float, label.split()))
            # Retrieve the maximum and minimum coordinates of the bounding boxe along axis x and y
            x_top = min(
                int(points[1] * img_width + (points[3] * img_width) // 2), img_width
            )
            y_left = max(int(points[2] * img_height - (points[4] * img_height) // 2), 0)
            x_bot = max(int(points[1] * img_width - (points[3] * img_width) // 2), 0)
            y_right = min(
                int(points[2] * img_height + (points[4] * img_height) // 2), img_height
            )
            # Check that the bounding box is fully inside the patch
            if (
                (x_top < x_max_patch)
                and (x_bot >= x_min_patch)
                and (y_right < y_max_patch)
                and (y_left >= y_min_patch)
            ):
                # Update the label to get the center, width and height of the bouding box normalized to the dimension of the patch
                class_id = int(points[0])
                x_center = (x_bot + x_top) / 2 / patch_width
                y_center = (y_left + y_right) / 2 / patch_height
                width = (x_top - x_bot) / patch_width
                height = (y_right - y_left) / patch_height
                patch_labels.append(
                    f"{class_id} {x_center} {y_center} {width} {height}"
                )

        elif task == "seg":
            # Retrieve the center coordinates of the segmentation mask
            points = list(map(float, label.split()[1:]))

            class_id = int(label[0])
            x_coords = [
                int(img_width * points[i]) for i in range(len(points)) if i % 2 == 0
            ]
            y_coords = [
                int(img_height * points[i]) for i in range(len(points)) if i % 2 == 1
            ]

            # Retrieve the maximum and minimum coordinates of the bounding boxe along axis x and y
            x_top = max(x_coords)
            x_bot = min(x_coords)
            y_right = max(y_coords)
            y_left = min(y_coords)

            # Check that the bounding box is fully inside the patch
            if (
                (x_top < x_max_patch)
                and (x_bot >= x_min_patch)
                and (y_right < y_max_patch)
                and (y_left >= y_min_patch)
            ):
                # Update the label to get the x and y coordinates of the segmentation mask normalized to the dimension of the patch
                line = (
                    str(class_id)
                    + " "
                    + " ".join(
                        [
                            f"{(x - x_min_patch)/patch_width} {(y - y_min_patch)/patch_height}"
                            for x, y in zip(x_coords, y_coords)
                        ]
                    )
                )
                patch_labels.append(line)

    # Combine the labels of all the annotated objects after filtering
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
    """Updates the list of patch and list of labels for the current path

    Arguments:
        patches_list: a list of numpy array of shape (patch_height, patch_width)
        patches_labels_list: a list of strings containing the labels of each patch
        img_array: a numpy array of shape (img_height, img_width)
        labels: a list of strings with all labels of the image
        img_height: a scalar denoting the height of the image in pixels
        img_width: a scalar denoting the width of the image in pixels
        x_min_path: a scalar denoting the minimum x coordinate of the patch
        x_max_patch: a scalar denoting the maximum x coordinate of the patch
        y_min_patch: a scalar denoting the minimum y coordinate of the patch
        y_max_patch: a scalar denoting the maximum y coordinate of the patch
        task: a string denoting the chosen task between object detection "box" or instance segmentation "seg"
    """
    # Adding the patch of the image to the patch list
    patches_list.append(img_array[y_min_patch:y_max_patch, x_min_patch:x_max_patch])

    # Get the labels of the patch
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
    # Add the labels of the patch to the patch labels list
    patches_labels_list.append(patches_labels)


def get_patches(
    n_rows_patch: int,
    n_cols_patch: int,
    img_array: np.array,
    path_label: str,
    split: str,
    task: str,
) -> tuple[list, list]:
    """Creates the patches and their labels for a given image

    Arguments:
        n_rows_patch: a scalar denoting the number of rows of patches
        n_cols_patch: a scalar denoting the number of columns of patches
        img_array: a numpy array of shape (image_height, image_width)
        path_label: a string denoting the path to the label file of the image
        split: a string denoting the split in which the image is
        task: a string denoting the chosen task between object detection "box" or instance segmentation "seg"

    Returns:
        patches: a list containing the patches
        labels: a list containing the labels of the patches
    """
    reg_patches = []
    reg_patches_labels = []

    inter_patches = []
    inter_patches_labels = []

    hor_border_patches = []
    hor_patches_labels = []

    ver_border_patches = []
    ver_patches_labels = []

    # Open the labels of the image
    if exists(path_label):
        with open(path_label, "r") as label_file:
            labels = label_file.readlines()

    else:
        labels = []

    # Get the dimensions of the image and the patches
    img_height, img_width = img_array.shape
    img_height_patch, img_width_patch = (
        img_height // n_rows_patch,
        img_width // n_cols_patch,
    )

    # Iterate through the rows and columns of patches to create
    for i in range(n_rows_patch):
        for j in range(n_cols_patch):

            # Get the minimum and maximum coordinates of the patch along the x and y axis
            x_min_patch, x_max_patch = j * img_width_patch, (j + 1) * img_width_patch
            y_min_patch, y_max_patch = i * img_height_patch, (i + 1) * img_height_patch

            # Set the half step along x and y axis to create patches on the edges between regular patches
            y_half_step = int(0.5 * img_height_patch)
            x_half_step = int(0.5 * img_width_patch)

            # Get the regular patch for row i and column j
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

            # If the image is part of the test split, patches on the edges of regular patches are needed to be able to segment the objects on the edges
            if split == "test":

                # Create inter patches shifted of a half step along x and y to cover the edges if it is not the last row or column
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

                # Create a patch to cover the edges between the patches of the first and last rows that are not covered by the inter patches because of the half step shift
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

                # Create a patch to cover the edges between the patches of the first and last columns that are not covered by the inter patches because of the half step shift
                if (j == 0 or j == n_cols_patch - 1) and i < n_rows_patch - 1:

                    update_patches(
                        ver_border_patches,
                        ver_patches_labels,
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

    # Return only the regular patches and their labels for the train and val splits
    if split in ["train", "val"]:
        return reg_patches, reg_patches_labels

    # Concatente the list of all patches and labels and return them
    else:
        test_patches = (
            reg_patches + inter_patches + hor_border_patches + ver_border_patches
        )
        test_labels = (
            reg_patches_labels
            + inter_patches_labels
            + hor_patches_labels
            + ver_patches_labels
        )

        return test_patches, test_labels


def save_patches(
    patches: list,
    labels: list,
    image_name: str,
    path_dataset: str,
    split: str,
    all_patches: bool,
):
    """Saves the patches and their labels

    Arguments:
        patches: a list containing the patches
        labels: a list containing the labels of the patches
        image_name: a string denoting the name of the image
        path_dataset: a string denoting the path to the dataset directory
        split: a string denoting the split in which the image is
        all_patches: a boolean denoting if all patches need to be saved
    """
    # Get the number of patches
    n_patches = len(patches)

    # Iterate through the patches
    for patch_idx, patch in enumerate(patches):

        # Get the label of the patch
        label = labels[patch_idx]
        # If the label is empty and that not all patches need to be saved, skip this patch
        if label == "" and not all_patches:
            continue

        # Create label path by adding the index of the patch and the total number of patch
        label_path = join(
            path_dataset,
            split,
            "labels",
            f"{image_name}_patch-{str(patch_idx+1)}-{str(n_patches)}.txt",
        )
        # Save patch label
        with open(label_path, "w") as label_file:
            label_file.write(label)

        # Create image path by adding the index of the patch and the total number of patch
        patch_path = join(
            path_dataset,
            split,
            "images",
            f"{image_name}_patch-{str(patch_idx+1)}-{str(n_patches)}.jpg",
        )
        # Save patch image
        cv2.imwrite(patch_path, patch)


def main(
    n_rows_patch: int,
    n_cols_patch: int,
    path_dataset: str,
    task: str,
    all_patches: bool,
):
    """Main function to create and save the patches and their labels

    Arguments:
        n_rows_patch: a scalar denoting the number of rows of patches
        n_cols_patch: a scalar denoting the number of columns of patches
        path_dataset: a string denoting the path to the dataset directory
        task: a string denoting the chosen task between object detection "box" or instance segmentation "seg"
        all_patches: a boolean denoting if all patches need to be saved
    """
    if task not in ["box", "seg"]:
        raise ValueError(f"Wrong task specified : {task}. Only box and seg available.")
    if not exists(path_dataset):
        raise ValueError(
            f"{path_dataset} provided for dataset directory path does not exist"
        )
    if n_rows_patch <= 0 or n_cols_patch <= 0:
        raise ValueError(
            f"{(n_rows_patch,n_cols_patch)} provided for number of rows and columns of patch. A strictly positive number of rows and columns is needed"
        )
    if n_rows_patch == 1 and n_cols_patch == 1:
        warnings.warn(
            "Only 1 row and 1 column of patch won't change the original image",
            UserWarning,
        )
    # Iterate through the splits
    for split in ["train", "val", "test"]:

        # Set the boolean variable to save all patches in train/val splits based on the argument all_patches
        if split in ["train", "val"]:
            save_all_patches = all_patches
        else:
            save_all_patches = True

        # Get the images path
        images_path = glob(join(path_dataset, split, "images", "*.jpg"))

        # Iterate through the images
        for img_path in tqdm(images_path):

            # Get the name of the image and open it
            image_name = basename(img_path).split(".")[0]
            image_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Get the path to the label of the image
            label_path = img_path.replace("images", "labels").replace("jpg", "txt")

            # Get the patches and their labels of the image
            patches, labels = get_patches(
                n_rows_patch, n_cols_patch, image_array, label_path, split, task
            )

            # Save the patches and their labels
            save_patches(
                patches, labels, image_name, path_dataset, split, save_all_patches
            )

            # Remove the original images to keep only the patches in the dataset
            os.remove(img_path)

            # Remove the original labels to keep only the patches labels in the dataset
            if exists(label_path):
                os.remove(label_path)

        print(f"All patches created for {split} split !")

    print(f"All patches created and saved for dataset {basename(path_dataset)} !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", type=str)
    parser.add_argument("--task", type=str, default="seg")
    parser.add_argument("--n_rows_patch", type=int, default=8)
    parser.add_argument("--n_cols_patch", type=int, default=8)
    parser.add_argument("--all_patches", action="store_true", default=False)
    args = parser.parse_args()

    main(
        args.n_rows_patch,
        args.n_cols_patch,
        args.path_dataset,
        args.task,
        args.all_patches,
    )
