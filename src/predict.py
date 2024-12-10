import ultralytics
from ultralytics import YOLO
from os.path import basename, join, exists
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import argparse
import warnings
import tifffile as tiff
from load_from_raw import adjust_contrast, preprocess_image
import shutil


def get_patches_images_only(
    n_rows_patch: int,
    n_cols_patch: int,
    img_array: np.array,
) -> tuple[list, list]:
    """Creates the patches for a given image

    Arguments:
        n_rows_patch: a scalar denoting the number of rows of patches
        n_cols_patch: a scalar denoting the number of columns of patches
        img_array: a numpy array of shape (image_height, image_width)

    Returns:
        patches: a list containing the patches
    """
    reg_patches = []
    inter_patches = []
    hor_border_patches = []
    ver_border_patches = []

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
            reg_patches.append(
                img_array[y_min_patch:y_max_patch, x_min_patch:x_max_patch]
            )

            # Create inter patches shifted of a half step along x and y to cover the edges if it is not the last row or column
            if i < n_rows_patch - 1 and j < n_cols_patch - 1:

                inter_patches.append(
                    img_array[
                        y_min_patch + y_half_step : y_max_patch + y_half_step,
                        x_min_patch + x_half_step : x_max_patch + x_half_step,
                    ]
                )

            # Create a patch to cover the edges between the patches of the first and last rows that are not covered by the inter patches because of the half step shift
            if (i == 0 or i == n_rows_patch - 1) and j < n_cols_patch - 1:

                hor_border_patches.append(
                    img_array[
                        y_min_patch:y_max_patch,
                        x_min_patch + x_half_step : x_max_patch + x_half_step,
                    ]
                )

            # Create a patch to cover the edges between the patches of the first and last columns that are not covered by the inter patches because of the half step shift
            if (j == 0 or j == n_cols_patch - 1) and i < n_rows_patch - 1:

                ver_border_patches.append(
                    img_array[
                        y_min_patch + y_half_step : y_max_patch + y_half_step,
                        x_min_patch:x_max_patch,
                    ]
                )

    # Concatente the list of all patches and labels and return them
    patches = reg_patches + inter_patches + hor_border_patches + ver_border_patches

    return patches


def get_masks(
    model_path: str, images_path: list, image_height: int, image_width: int
) -> list:
    """Runs the model to predict the binary mask for a given list of images

    Arguments:
        model_path: a string denoting the path to the model weights to use for the segmentation
        images_path: a list of strings denoting the paths to images
        image_height: a scalar denoting the height of the images in pixels
        image_width: a scalar denoting the width of the images in pixels

    Returns:
        masks: a list of numpy array containing the binary mask for each image
    """
    # Load the model and run the segmentation
    model = YOLO(model_path)
    result_patches = model(images_path, verbose=False)

    masks = []
    # Iterate through the results of the segmentation
    for result_patch in result_patches:
        # Create an zeros array that will store the binary mask
        binary_mask_patch = np.zeros((image_height, image_width))
        # Get the predicted mask
        mask = result_patch.masks
        if mask != None:
            # Get the coordinates of the mask and check that it is not empty
            coordinates = mask.xy
            non_empty_masks = [coords for coords in coordinates if coords.size > 0]

            # Add the contours of the mask to the numpy array if it is not empty
            if non_empty_masks:
                contours = [
                    contour.astype(np.int32).reshape((-1, 1, 2))
                    for contour in non_empty_masks
                ]
                binary_mask_patch = cv2.drawContours(
                    binary_mask_patch, contours, -1, (255), thickness=cv2.FILLED
                ).astype(np.uint8)

        masks.append(binary_mask_patch)

    return masks


def combine_mask_patches(
    patches_masks: list,
    n_patches: dict,
    n_rows_patch: int,
    n_cols_patch: int,
    img_shape: tuple,
) -> np.array:
    """Combines the masks from the patches to reconstruct the binary mask of the full images from the patch segmentations

    Arguments:
        patches_masks: a list of numpy array containing the binary mask for each patch of the image
        n_patches: a dictionary denoting the number of patches for each type of patch
        n_rows_patch: a scalar denoting the number of rows of patches
        n_cols_patch: a scalar denoting the number of columns of patches
        img_shape: a tuple denoting the shape of the full image to reconstruct

    Returns:
        combined_array: a numpy array containing the reconstructed binary mask of the full image from the masks of each patches
    """
    # Get the list of patches for each type of patch
    reg_patches = patches_masks[: n_patches["reg"]]
    inter_patches = patches_masks[
        n_patches["reg"] : n_patches["reg"] + n_patches["inter"]
    ]
    hor_patches = patches_masks[
        n_patches["reg"]
        + n_patches["inter"] : n_patches["reg"]
        + n_patches["inter"]
        + n_patches["hor"]
    ]
    ver_patches = patches_masks[
        n_patches["reg"]
        + n_patches["inter"]
        + n_patches["hor"] : n_patches["reg"]
        + n_patches["inter"]
        + n_patches["hor"]
        + n_patches["ver"]
    ]

    # Initialize a count to help locating the patches on the full image from its index
    reg_patches_count = 0
    inter_patches_count = 0
    hor_patches_count = 0
    ver_patches_count = 0

    # Initialize zeros array to first combine the masks of same-type patches
    reg_array = np.zeros(img_shape)
    inter_array = np.zeros(img_shape)
    hor_array = np.zeros(img_shape)
    ver_array = np.zeros(img_shape)

    # Get the dimensions of the full image and of each patch
    img_height, img_width = img_shape
    img_height_patch, img_width_patch = (
        img_height // n_rows_patch,
        img_width // n_cols_patch,
    )

    # Iterate through the rows and columns of patches
    for i in range(n_rows_patch):
        for j in range(n_cols_patch):

            # Get the minimum and maximum coordinates of the patch along the x and y axis
            x_min_patch, x_max_patch = j * img_width_patch, (j + 1) * img_width_patch
            y_min_patch, y_max_patch = i * img_height_patch, (i + 1) * img_height_patch

            # Set the half step along x and y axis to create patches on the edges between regular patches
            y_half_step = int(0.5 * img_height_patch)
            x_half_step = int(0.5 * img_width_patch)

            # Update the mask of regular patches and the count of regular patches
            reg_array[y_min_patch:y_max_patch, x_min_patch:x_max_patch] = reg_patches[
                reg_patches_count
            ]
            reg_patches_count += 1

            # Update the mask of inter patches and the count of inter patches if it is not the last row or column
            if i < n_rows_patch - 1 and j < n_cols_patch - 1:

                inter_array[
                    y_min_patch + y_half_step : y_max_patch + y_half_step,
                    x_min_patch + x_half_step : x_max_patch + x_half_step,
                ] = inter_patches[inter_patches_count]
                inter_patches_count += 1

            # Update the mask of horizontal patches and the count of horizontal patches if it is the first or last row
            if (i == 0 or i == n_rows_patch - 1) and j < n_cols_patch - 1:

                hor_array[
                    y_min_patch:y_max_patch,
                    x_min_patch + x_half_step : x_max_patch + x_half_step,
                ] = hor_patches[hor_patches_count]
                hor_patches_count += 1

            # Update the mask of vertical patches and the count of vertical patches if it is the first or last column
            if (j == 0 or j == n_cols_patch - 1) and i < n_rows_patch - 1:

                ver_array[
                    y_min_patch + y_half_step : y_max_patch + y_half_step,
                    x_min_patch:x_max_patch,
                ] = ver_patches[ver_patches_count]
                ver_patches_count += 1

    # Combine the final masks of each type of patches by taking the maximum cell-wise value
    combined_array = np.maximum.reduce(
        [reg_array, inter_array, hor_array, ver_array]
    ).astype(np.uint8)

    return combined_array


def combine_mask_models(mask_model_a: np.array, mask_model_b: np.array) -> np.array:
    """Combines two binary masks

    Arguments:
        mask_model_a: a numpy array of shape (image_height, image_width)
        mask_model_b: a numpy array of shape (image_height, image_width)

    Returns:
        combined_mask: a numpy array of shape (image_height, image_width) combining the two arrays by taking the maximum cell-wise value
    """

    return np.maximum(mask_model_a, mask_model_b)


def predict_image(
    n_rows_patch: int,
    n_cols_patch: int,
    path_dataset_full_images: str,
    path_dataset_patch_images: str,
    path_model_full: str,
    path_model_patch: str,
    img_name: str,
) -> tuple[list, list]:
    """Predicts the segmentation masks for each slice of an image

    Arguments:
        n_rows_patch: a scalar denoting the number of rows of patches
        n_cols_patch: a scalar denoting the number of columns of patches
        path_dataset_full_images: a string denoting the path to the image
        path_dataset_patch_images: a string denoting the path to the patches
        path_model_full: a string denoting the path to the weights of the model to segment the full image
        path_model_patch: a string denoting the path to the weights of the model to segment the patches
        img_name: a string denoting the name of the image

    Returns:
        img_mask: a list of the numpy arrays of shape (image_height, image_width) for the mask of each slice of the image
        img: a list of the numpy arrays of shape (image_height, image_width) for each slice of the image
    """
    # Get the number of each kind of patch based on the number of rows and columns of patch
    n_reg_patches = n_rows_patch * n_cols_patch
    n_inter_patches = (n_rows_patch - 1) * (n_cols_patch - 1)
    n_hor_patches = 2 * (n_cols_patch - 1)
    n_ver_patches = 2 * (n_rows_patch - 1)

    img_mask = []
    img = []

    # Set the dictionary with the number of each kind of patch
    n_patches = {
        "reg": n_reg_patches,
        "inter": n_inter_patches,
        "hor": n_hor_patches,
        "ver": n_ver_patches,
    }

    # Retrieve the paths to the slices of the image
    path_slices_full_image = glob(join(path_dataset_full_images, img_name + "*.jpg"))

    # Sort the slices
    path_slices_full_image_sorted = [0] * len(path_slices_full_image)
    for img_path in path_slices_full_image:
        slice_idx = basename(img_path).split(".")[0].split("-")[-1]
        path_slices_full_image_sorted[int(slice_idx)] = img_path

    # Iterate through the slices of the image
    for slice_idx, slice_path in tqdm(enumerate(path_slices_full_image_sorted)):

        # Retrieve the paths to the patches of the slice
        path_patches_slice = glob(
            join(
                path_dataset_patch_images, basename(slice_path).split(".")[0] + "_*.jpg"
            )
        )
        # Sort the patches
        path_patches_sorted = [0] * len(path_patches_slice)
        for img_path in path_patches_slice:
            patch_idx = basename(img_path).split(".")[0].split("-")[-2]
            path_patches_sorted[int(patch_idx) - 1] = img_path

        # Open the full slice
        full_image = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

        # Get the dimensions of the full image and the patches
        image_height, image_width = full_image.shape
        patch_height, patch_width = (
            image_height // n_rows_patch,
            image_width // n_cols_patch,
        )

        # Get the segmentation masks of the full image and each patch
        image_mask = get_masks(
            path_model_full, [slice_path], image_height, image_width
        )[0]
        patches_masks = get_masks(
            path_model_patch, path_patches_sorted, patch_height, patch_width
        )

        # Combine the masks of the patches
        combined_patches_masks = combine_mask_patches(
            patches_masks,
            n_patches,
            n_rows_patch,
            n_cols_patch,
            (image_height, image_width),
        )
        del patches_masks

        # Combine the masks of the patches and the mask of the full image
        final_mask = combine_mask_models(image_mask, combined_patches_masks)
        del image_mask, combined_patches_masks

        # Add the final mask and the slice to their respective lists
        img_mask.append(final_mask)
        img.append(full_image)

    return img_mask, img


def save_outputs(img_mask: list, img: list, img_name: str, path_output_dir: str):
    """Saves the outputs of the segmentation masks

    Arguments:
        img_mask: a list of the numpy arrays of shape (image_height, image_width) for the mask of each slice of the image
        img: a list of the numpy arrays of shape (image_height, image_width) for each slice of the image
        img_name: a string denoting the name of the image
        path_output_dir: a string denoting the path to the output directory
    """
    # Combine all masks and all slices in a single array of shape (n_slices, image_height, image_width)
    all_slices = np.array(img_mask)
    all_images = np.array(img)
    # Create a zero array to store the aggregated masks of all slices
    combined_slices = np.zeros((all_slices.shape[1], all_slices.shape[2]))
    # Create an array filled fill 255 to store the aggregated segmentations of all slices
    combined_images = np.ones((all_slices.shape[1], all_slices.shape[2])) * 255

    # Create the necessary directories in the output folder
    os.makedirs(join(path_output_dir, img_name), exist_ok=True)
    os.makedirs(join(path_output_dir, img_name, "slices"), exist_ok=True)

    # Iterate through the slices
    for i in range(all_slices.shape[0]):
        # Aggregate the current slice's mask and segmentation
        combined_slices = np.where(all_slices[i] == 255, i + 1, combined_slices)
        combined_images = np.where(all_slices[i] == 255, all_images[i], combined_images)
        # Save the current slice's mask
        cv2.imwrite(
            join(path_output_dir, img_name, "slices", f"{img_name}_slice_{i}_mask.jpg"),
            all_slices[i],
        )
        segmented_image = np.where(all_slices[i] == 255, all_images[i], 255)
        # Save the current slice's segmentation
        cv2.imwrite(
            join(
                path_output_dir,
                img_name,
                "slices",
                f"{img_name}_slice_{i}_segmented_image.jpg",
            ),
            segmented_image,
        )

    # Save the aggregated masks and segmentations
    cv2.imwrite(join(path_output_dir, img_name, "combined_mask.jpg"), combined_slices)
    cv2.imwrite(
        join(path_output_dir, img_name, "combined_segmented_images.jpg"),
        combined_images,
    )


def predict_image_from_raw(
    image_path: str,
    preprocess: bool,
    min_range: int,
    max_range: int,
    n_rows_patch: int,
    n_cols_patch: int,
    path_model_full: str,
    path_model_patch: str,
    path_output_dir: str,
):
    """Predict the segmentation masks of an image from the raw .tif file

    Arguments:
        image_path: a string denoting the path to the image .tif file
        preprocess: a boolean denoting if the original image needs to be preprocessed
        min_range: a scalar denoting the minimum pixel value accepted from the original image
        max_range: a scalar denoting the maximum pixel value accepted from the original image
        n_rows_patch: a scalar denoting the number of rows of patches
        n_cols_patch: a scalar denoting the number of columns of patches
        path_model_full: a string denoting the path to the weights of the model to segment the full image
        path_model_patch: a string denoting the path to the weights of the model to segment the patches
        path_output_dir: a string denoting the path to the output directory
    """
    # Open the original tif file
    img_stack = tiff.imread(image_path)
    img_name = basename(image_path).split(".")[0]
    # Iterate through the slices of the image
    for slice_idx, slice_array in enumerate(img_stack):
        # Crop the array to remove white padding from original image
        cropped_array = slice_array[95:-137, 39:-57]
        # Adjust the contrast of the image to the chosen range
        adjust_slice_array = adjust_contrast(cropped_array, min_range, max_range)
        # Preprocess the image if necessary
        if preprocess:
            preprocessed_array = preprocess_image(
                adjust_slice_array, (4, 4), (5, 5), 1, 1, 0, 1
            )
        else:
            preprocessed_array = adjust_slice_array.astype(np.uint8)

        # Save the slice to the temporary directory
        image_path = join(
            "temp",
            "full_images",
            f"{img_name}_slice-{slice_idx}.jpg",
        )
        cv2.imwrite(image_path, preprocessed_array)

        # Get the patches of the slice
        patches = get_patches_images_only(
            n_rows_patch,
            n_cols_patch,
            preprocessed_array,
        )

        n_patches = len(patches)

        # Iterate through the patches
        for patch_idx in range(n_patches):
            # Save the patch to the temporary directory
            patch_path = join(
                "temp",
                "patches",
                f"{img_name}_slice-{slice_idx}_patch-{str(patch_idx+1)}-{str(n_patches)}.jpg",
            )
            cv2.imwrite(patch_path, patches[patch_idx])

    # Predict the segmentation masks of the image
    img_mask, img = predict_image(
        n_rows_patch,
        n_cols_patch,
        join("temp", "full_images"),
        join("temp", "patches"),
        path_model_full,
        path_model_patch,
        img_name,
    )
    # Create and save the outputs from the segmentation mask of the image
    save_outputs(img_mask, img, img_name, path_output_dir)


def main(
    path_images: str,
    path_output_dir: str,
    n_rows_patch: int,
    n_cols_patch: int,
    path_model_full: str,
    path_model_patch: str,
    preprocess: bool,
    min_range: int,
    max_range: int,
):
    """Main function to predict the segmentation masks of images located in a given directory

    Arguments:
        path_images: a string denoting the path to the images to predict
        preprocess: a boolean denoting if the original image needs to be preprocessed
        min_range: a scalar denoting the minimum pixel value accepted from the original image
        max_range: a scalar denoting the maximum pixel value accepted from the original image
        n_rows_patch: a scalar denoting the number of rows of patches
        n_cols_patch: a scalar denoting the number of columns of patches
        path_model_full: a string denoting the path to the weights of the model to segment the full image
        path_model_patch: a string denoting the path to the weights of the model to segment the patches
        path_output_dir: a string denoting the path to the output directory
    """
    if not exists(path_images):
        raise ValueError(f"{path_images} path for images does not exist")
    if not exists(path_model_full):
        raise ValueError(f"{path_model_full} path for images does not exist")
    if not exists(path_model_patch):
        raise ValueError(f"{path_model_patch} path for images does not exist")
    if min_range < 0 or max_range > 255:
        warnings.warn(
            f"({min_range},{max_range}) range provided for contrast adjustment while the pixel values range is (0,255). Nothing will happen.",
            UserWarning,
        )
    if n_rows_patch <= 0 or n_cols_patch <= 0:
        raise ValueError(
            f"{(n_rows_patch,n_cols_patch)} provided for number of rows and columns of patch. A strictly positive number of rows and columns is needed. You must use the ones used for training"
        )
    if n_rows_patch == 1 and n_cols_patch == 1:
        warnings.warn(
            "Only 1 row and 1 column of patch won't change the original image. You must use the ones used for training",
            UserWarning,
        )
    # Create the temporary directory where the intermediary images will be saved
    os.makedirs(path_output_dir, exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs(join("temp", "full_images"), exist_ok=True)
    os.makedirs(join("temp", "patches"), exist_ok=True)
    # Get the path to the images
    images_path = glob(join(path_images, "*"))
    # Iterate through the images' path
    for image_path in tqdm(images_path):
        # Predict and save the outputs for the image
        predict_image_from_raw(
            image_path,
            preprocess,
            min_range,
            max_range,
            n_rows_patch,
            n_cols_patch,
            path_model_full,
            path_model_patch,
            path_output_dir,
        )
    # Delete the temporary folder
    shutil.rmtree("temp")
    print(f"Outputs saved at {path_output_dir} !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_images", type=str)
    parser.add_argument("--path_output_dir", type=str)
    parser.add_argument("--n_rows_patch", type=int, default=8)
    parser.add_argument("--n_cols_patch", type=int, default=8)
    parser.add_argument("--path_model_full", type=str)
    parser.add_argument("--path_model_patch", type=str)
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--min_contrast", type=int, default=0)
    parser.add_argument("--max_contrast", type=int, default=255)
    args = parser.parse_args()

    ultralytics.checks()
    main(
        args.path_images,
        args.path_output_dir,
        args.n_rows_patch,
        args.n_cols_patch,
        args.path_model_full,
        args.path_model_patch,
        args.preprocess,
        args.min_contrast,
        args.max_contrast,
    )
