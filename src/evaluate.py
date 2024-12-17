import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from os.path import join, basename, exists
from matplotlib.colors import Normalize
import gc
import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy.ndimage import label, binary_fill_holes
from predict import (
    get_masks,
    combine_mask_patches,
    select_slices_for_output,
    combine_mask_models,
)
import argparse
import ultralytics
import tifffile as tiff


def visualise(
    img_name: str,
    segmented: bool,
    path_dataset_patch_images: str,
    path_dataset_full_images: str,
    path_model_full: str,
    path_model_patch: str,
    use_patch: bool,
    n_rows_patch: int = 0,
    n_cols_patch: int = 0,
):
    """
    Visualizes image slices and their corresponding binary masks with adjustable display settings

    Args:
        img_name: a string denoting the name of the image to process
        segmented: a boolean denoting if the segmented image should be displayed
        path_dataset_patch_images: a string denoting the path to the the patch images set if used
        path_dataset_full_images: a string denoting the path to the the full images set
        path_model_full: a string denoting the path to the the full model
        path_model_patch: a string denoting the path to the the patch model if used
        use_patch: a boolean denoting if a patch model is used
        n_rows_patch: a scalar denoting the number of rows of patches if used
        n_cols_patch: a scalar denoting the number of columns of patches if used

    Returns:
        interactive_plot: a widgets.Widget interactive plot for visualizing images and masks
        img_mask: a list of binary masks for each slice
        img: a list of full images corresponding to the slices
    """
    # Get the number of each kind of patch if a patch model is used
    if use_patch:
        n_reg_patches = n_rows_patch * n_cols_patch
        n_inter_patches = (n_rows_patch - 1) * (n_cols_patch - 1)
        n_hor_patches = 2 * (n_cols_patch - 1)
        n_ver_patches = 2 * (n_rows_patch - 1)
        n_patches = {
            "reg": n_reg_patches,
            "inter": n_inter_patches,
            "hor": n_hor_patches,
            "ver": n_ver_patches,
        }

    img_mask = []
    big_model_img_mask = []
    small_model_img_mask = []
    img = []

    # Get the paths to the slices of the image and sort them by slice index
    path_slices_full_image = glob(join(path_dataset_full_images, img_name + "*.jpg"))

    path_slices_full_image_sorted = [0] * len(path_slices_full_image)
    for img_path in path_slices_full_image:
        slice_idx = basename(img_path).split(".")[0].split("-")[-1]
        path_slices_full_image_sorted[int(slice_idx)] = img_path

    for slice_idx, slice_path in tqdm(enumerate(path_slices_full_image_sorted)):

        # Get the paths to the patches of the slice and sort them by patch index if a patch model is used
        if use_patch:
            path_patches_slice = glob(
                join(
                    path_dataset_patch_images,
                    basename(slice_path).split(".")[0] + "_*.jpg",
                )
            )
            path_patches_sorted = [0] * len(path_patches_slice)
            for img_path in path_patches_slice:
                patch_idx = basename(img_path).split(".")[0].split("-")[-2]
                path_patches_sorted[int(patch_idx) - 1] = img_path

        # Open the slice image
        full_image = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

        # Get the dimension of the image and the patches
        image_height, image_width = full_image.shape
        if use_patch:
            patch_height, patch_width = (
                image_height // n_rows_patch,
                image_width // n_cols_patch,
            )

        # Get the masks for the slice and fill the holes
        image_mask = get_masks(
            path_model_full, [slice_path], image_height, image_width
        )[0]
        image_mask = binary_fill_holes(image_mask).astype(int)
        big_model_img_mask.append(image_mask)
        # Get the mask for the patches, combine them and fill the holes
        if use_patch:
            patches_masks = get_masks(
                path_model_patch, path_patches_sorted, patch_height, patch_width
            )
            combined_patches_masks = combine_mask_patches(
                patches_masks,
                n_patches,
                n_rows_patch,
                n_cols_patch,
                (image_height, image_width),
            )
            combined_patches_masks = binary_fill_holes(combined_patches_masks).astype(
                int
            )
            small_model_img_mask.append(combined_patches_masks)
            # Combined the masks of the slice and the patches
            slice_mask = combine_mask_models(image_mask, combined_patches_masks)
            img_mask.append(slice_mask)
            del patches_masks
        else:
            img_mask.append(image_mask)

        img.append(full_image)
        gc.collect()

    all_slices_big_model = np.array(big_model_img_mask)
    all_images = np.array(img)
    # Select the best slice for each segmented object and aggregate the masks and segmented images
    if use_patch:
        all_slices_small_model = np.array(small_model_img_mask)
        combined_slices_big_model, combined_images_big_model = select_slices_for_output(
            all_slices_big_model, all_images
        )
        combined_slices_small_model, combined_images_small_model = (
            select_slices_for_output(all_slices_small_model, all_images)
        )

        combined_slices = np.where(
            combined_slices_small_model > 0,
            combined_slices_small_model,
            combined_slices_big_model,
        )
        combined_images = np.where(
            combined_slices_small_model > 0,
            combined_images_small_model,
            combined_images_big_model,
        )
    else:
        combined_slices, combined_images = select_slices_for_output(
            all_slices_big_model, all_images
        )

    def display_boxes(start_slice: int, min_contrast: int, max_contrast: int):
        plt.ioff()

        if segmented:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            norm = Normalize(vmin=1, vmax=101)

            axes[0].imshow(
                img[start_slice], cmap="gray", vmin=min_contrast, vmax=max_contrast
            )
            axes[0].imshow(
                np.where(img_mask[start_slice] == 0, np.nan, start_slice + 1),
                cmap="viridis",
                alpha=0.5,
                norm=norm,
            )
            axes[0].axis("off")
            axes[1].imshow(
                np.where(img_mask[start_slice] == 1, img[start_slice], 255),
                cmap="gray",
                vmin=min_contrast,
                vmax=max_contrast,
            )
            axes[1].axis("off")
        else:
            fig, axes = plt.subplots(1, 1, figsize=(20, 10))
            norm = Normalize(vmin=1, vmax=101)
            axes.imshow(
                img[start_slice], cmap="gray", vmin=min_contrast, vmax=max_contrast
            )
            axes.imshow(
                np.where(img_mask[start_slice] == 0, np.nan, start_slice + 1),
                cmap="viridis",
                alpha=0.5,
                norm=norm,
            )
            axes.axis("off")

        plt.tight_layout()
        plt.show()

    slice_slider = widgets.IntSlider(
        value=0, min=0, max=100, step=1, description="Slice:"
    )

    contrast_min_slider = widgets.IntSlider(
        value=0, min=0, max=254, step=1, description="Min Contrast:"
    )

    contrast_max_slider = widgets.IntSlider(
        value=255, min=1, max=255, step=1, description="Max Contrast:"
    )

    interactive_plot = widgets.interactive(
        display_boxes,
        start_slice=slice_slider,
        min_contrast=contrast_min_slider,
        max_contrast=contrast_max_slider,
    )

    return interactive_plot, img_mask, img, combined_slices, combined_images


def get_mask_from_gt(labeled_folder_path: str, img_name: str) -> np.array:
    """Gets the mask from the ground truth file

    Arguments:
        labeled_folder_path: a string denoting the path to the folder with the labels
        img_name: a string denoting the name of the image

    Returns:
        mask_gt: a binary numpy array of shape (image_height, image_weight) with the masks
    """
    slices = glob(join(labeled_folder_path, "images", f"{img_name}*.jpg"))
    mask_gt = np.zeros((1008, 1520))  # HARDCODED

    for slice_path in slices:
        small_label_path = slice_path.replace("images", "small_labels").replace(
            ".jpg", ".txt"
        )
        big_label_path = slice_path.replace("images", "big_labels").replace(
            ".jpg", ".txt"
        )

        if exists(small_label_path):
            with open(small_label_path, "r") as label_file:
                labels = label_file.readlines()

            for line in labels:

                # Get the coordinates of the mask
                points = list(map(float, line.split()[1:]))

                # Format the coordinates to the format expected by OpenCV
                contours = [
                    np.array(
                        [
                            (int(1520 * points[i]), int(1008 * points[i + 1]))
                            for i in range(0, len(points), 2)
                        ],
                        dtype=np.int32,
                    )
                ]

                mask_gt = cv2.drawContours(
                    mask_gt, contours, -1, (255), thickness=cv2.FILLED
                )
        if exists(big_label_path):
            with open(big_label_path, "r") as label_file:
                labels = label_file.readlines()

            for line in labels:

                # Get the coordinates of the mask
                points = list(map(float, line.split()[1:]))

                # Format the coordinates to the format expected by OpenCV
                contours = [
                    np.array(
                        [
                            (int(1520 * points[i]), int(1008 * points[i + 1]))
                            for i in range(0, len(points), 2)
                        ],
                        dtype=np.int32,
                    )
                ]

                mask_gt = cv2.drawContours(
                    mask_gt, contours, -1, (255), thickness=cv2.FILLED
                )

    return mask_gt


def evaluate_final_output(mask_gt: np.array, mask_pred: np.array) -> dict:
    """Evaluates the performance of a model on the groud truth masks

    Arguments:
        mask_gt: a numpy array of shape (image_height, image_width) with the masks of the ground truth
        mask_pred: a numpy array of shape (image_height, image_width) with the predicted masks of the model

    Returns:
        perf_dict: a dictionary with the performances of the model
    """
    # Concatenate the ground truth and the prediction
    concat_array = np.array([mask_gt, mask_pred])
    # Get the connected components
    label_im, n_comp = label(concat_array)
    tp = 0
    fp = 0
    fn = 0

    best_dices = []
    sizes = []
    # Iterate through the connected components
    for i in range(1, n_comp + 1):
        # Get the array of the connected components and its indices
        array_comp = np.where(label_im == i, 1, 0)
        indices = np.argwhere(array_comp == 1)
        z_min = indices[:, 0].min()
        z_max = indices[:, 0].max()
        if z_min == z_max:
            # If the connected component is only on the ground truth array then it is a false negative and the dice is null
            if z_min == 0:
                fn += 1
                best_dices.append(0)
                sizes.append(array_comp[0].sum())
            # If the connected component is only on the predicted array then it is a false positive and no dice is computed
            else:
                fp += 1
        # If the connected component is on both arrays then it is a true positive and the dice is computed
        else:
            tp += 1
            summed_mask = array_comp[0] + array_comp[1]
            dice = 2 * np.where(summed_mask == 2, 1, 0).sum() / summed_mask.sum()
            best_dices.append(dice)
            sizes.append(array_comp[0].sum())
    # Get the non zero dices and the corresponding sizes
    best_dices_non_zero = [best_dice for best_dice in best_dices if best_dice != 0]
    sizes_non_zero = [sizes[i] for i in range(len(best_dices)) if best_dices[i] != 0]

    # Compute the metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * recall * precision / (recall + precision)
    mDice = np.mean(best_dices)
    mDice_non_zero = np.mean(best_dices_non_zero)
    weigthed_mDice = np.sum(
        [sizes[i] * best_dices[i] for i in range(len(sizes))]
    ) / np.sum(sizes)
    weigthed_mDice_non_zero = np.sum(
        [
            sizes_non_zero[i] * best_dices_non_zero[i]
            for i in range(len(best_dices_non_zero))
        ]
    ) / np.sum(sizes_non_zero)

    perf_dict = {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "recall": recall,
        "precision": precision,
        "f1-score": f1_score,
        "mDice": mDice,
        "nonzero_mDice": mDice_non_zero,
        "weighted_mDice": weigthed_mDice,
        "nonzero_weighted_mDice": weigthed_mDice_non_zero,
        "best_dices": best_dices,
        "best_dices_non_zero": best_dices_non_zero,
        "sizes": sizes,
        "sizes_nonzero": sizes_non_zero,
    }

    return perf_dict


def main(
    path_labeled_folder: str,
    path_model_full: str,
    path_dataset_full: str,
    path_model_patch: str,
    path_dataset_patch: str,
    use_patch: bool,
    n_rows_patch: int = 0,
    n_cols_patch: int = 0,
):
    """Main function to evaluate the performance of a model

    Arguments:
        path_labeled_folder: a string denoting the path to the ground truth folder
        path_model_full: a string denoting the path to the weights of the full model
        path_dataset_full: a string denoting the path to the the full images in test set
        path_model_patch: a string denoting the path to the weights of the patch model if used
        path_dataset_patch: a string denoting the path to the the patch images in test set if used
        use_patch: a boolean denoting if a patch model is used
        n_rows_patch: a scalar denoting the number of rows of patch if used
        n_cols_patch: a scalar denoting the number of columns of patch if used
    """
    if use_patch:
        path_list = [
            path_labeled_folder,
            path_model_full,
            path_dataset_full,
            path_model_patch,
            path_dataset_patch,
        ]
    else:
        path_list = [path_labeled_folder, path_model_full, path_dataset_full]
    for path in path_list:
        if not exists(path):
            raise ValueError(f"{path} does not exist")
    if use_patch and (n_cols_patch <= 0 or n_rows_patch <= 0):
        raise ValueError(
            "The number of patches rows and columns can't be negative if you use patches"
        )

    if use_patch:
        images_small = set(
            [
                basename(file_path).split("_")[0]
                for file_path in glob(join(path_dataset_patch, "images", "*.jpg"))
            ]
        )
        images_big = set(
            [
                basename(file_path).split("_")[0]
                for file_path in glob(join(path_dataset_full, "images", "*.jpg"))
            ]
        )
        images = list(images_small.union(images_big))
    else:
        images = list(
            set(
                [
                    basename(file_path).split("_")[0]
                    for file_path in glob(join(path_dataset_full, "images", "*.jpg"))
                ]
            )
        )

    test_tp_new = 0
    test_fp_new = 0
    test_fn_new = 0
    best_dices_new = []
    best_dices_non_zero_new = []
    sizes_new = []
    sizes_nonzero_new = []

    # Iterate through the images of the test set
    for img_name in images:
        # Get the mask from the ground truth
        mask_gt = get_mask_from_gt(path_labeled_folder, img_name)
        # Get the mask from the original model
        img_stack = tiff.imread(f"dep_files/{img_name}-dep.tiff")
        gray_image = cv2.cvtColor(img_stack, cv2.COLOR_BGR2GRAY)
        cropped_gray_image = gray_image[95:-167, 39:-57]
        _, mask_previous_model = cv2.threshold(
            cropped_gray_image.astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        # Get the mask from the model if it uses patches or not
        if use_patch:
            _, _, _, combined_slices, _ = visualise(
                img_name,
                True,
                path_dataset_patch,
                path_dataset_full,
                path_model_full,
                path_model_patch,
                True,
                n_rows_patch,
                n_cols_patch,
            )
        else:
            # Get the mask from the combined model
            _, _, _, combined_slices, _ = visualise(
                img_name, True, None, path_dataset_full, path_model_full, None, False
            )

        # Evaluate the combined model's mask
        perf_dict_new = evaluate_final_output(
            mask_gt, np.where(combined_slices > 0, 1, 0)
        )
        test_tp_new += perf_dict_new["true_positive"]
        test_fp_new += perf_dict_new["false_positive"]
        test_fn_new += perf_dict_new["false_negative"]
        best_dices_new += perf_dict_new["best_dices"]
        best_dices_non_zero_new += perf_dict_new["best_dices_non_zero"]
        sizes_new += perf_dict_new["sizes"]
        sizes_nonzero_new += perf_dict_new["sizes_nonzero"]

        # Output the performances of each model for the current image
        print(f"For image {img_name}:")
        for key in perf_dict_new.keys():
            if key not in [
                "best_dices",
                "best_dices_non_zero",
                "sizes",
                "sizes_nonzero",
            ]:
                print(f"{key} : {round(perf_dict_new[key],2)}")
        print("")
    # Compute the final performances on the test set
    test_recall_new = test_tp_new / (test_tp_new + test_fn_new)
    test_precision_new = test_tp_new / (test_tp_new + test_fp_new)
    test_f1_score_new = (
        2
        * test_recall_new
        * test_precision_new
        / (test_recall_new + test_precision_new)
    )
    test_mDice_new = np.mean(best_dices_new)
    test_mDice_non_zero_new = np.mean(best_dices_non_zero_new)
    test_weighted_mDice_new = np.sum(
        [sizes_new[i] * best_dices_new[i] for i in range(len(sizes_new))]
    ) / np.sum(sizes_new)
    test_weighted_mDice_non_zero_new = np.sum(
        [
            sizes_nonzero_new[i] * best_dices_non_zero_new[i]
            for i in range(len(best_dices_non_zero_new))
        ]
    ) / np.sum(sizes_nonzero_new)

    # Output the final performances on the test set
    print("On test set:")
    print(f"True Positives : {round(test_tp_new,2)}")
    print(f"False Positives : {round(test_fp_new,2)}")
    print(f"False Negatives : {round(test_fn_new,2)}")
    print(f"Recall : {round(test_recall_new,2)}")
    print(f"Precision : {round(test_precision_new,2)}")
    print(f"F1-Score : {round(test_f1_score_new,2)}")
    print(f"mDice : {round(test_mDice_new,2)}")
    print(f"mDice_nonzero :  {round(test_mDice_non_zero_new,2)}")
    print(f"weighted_mDice : {round(test_weighted_mDice_new,2)}")
    print(f"weighted_mDice_nonzero : {round(test_weighted_mDice_non_zero_new,2)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_labeled_folder", type=str)
    parser.add_argument("--path_model_full", type=str)
    parser.add_argument("--path_dataset_full", type=str)
    parser.add_argument("--path_model_patch", type=str, default="")
    parser.add_argument("--path_dataset_patch", type=str, default="")
    parser.add_argument("--use_patch", action="store_true", default=False)
    parser.add_argument("--n_rows_patch", type=int, default=0)
    parser.add_argument("--n_cols_patch", type=int, default=0)
    args = parser.parse_args()

    ultralytics.checks()

    main(
        args.path_labeled_folder,
        args.path_model_full,
        args.path_dataset_full,
        args.path_model_patch,
        args.path_dataset_patch,
        args.use_patch,
        args.n_rows_patch,
        args.n_cols_patch,
    )
