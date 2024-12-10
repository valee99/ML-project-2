import cv2
import albumentations as alb
from glob import glob
import argparse
from os.path import join, exists
import numpy as np
from tqdm import tqdm


def get_masks_from_label(label_path: str, image_path: str) -> tuple[list, dict]:
    """Gets the masks of the annotated objects from the label file

    Arguments:
        label_path: a string denoting the path to the label file
        image_path: a string denoting the path to the image

    Returns:
        masks: a list of numpy arrays as binary masks for each annotated object
        class_labels: a list of integers as the class labels for each annotated object
    """
    # Open the image and get the dimensions
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_width = img.shape[1]
    image_height = img.shape[0]

    # Open the labels
    with open(label_path, "r") as label_file:
        labels = label_file.readlines()

    masks = []
    class_labels = []

    # Iterate through the labels to get their masks
    for line in labels:

        class_labels.append(line.split()[0])

        # Get the coordinates of the mask
        points = list(map(float, line.split()[1:]))

        # Format the coordinates to the format expected by OpenCV
        contours = [
            np.array(
                [
                    (int(image_width * points[i]), int(image_height * points[i + 1]))
                    for i in range(0, len(points), 2)
                ],
                dtype=np.int32,
            )
        ]

        # Create a binary mask and draw the contours on it
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        mask = cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # Add the binary mask to the list of masks
        masks.append(mask)

    return masks, class_labels


def augment_image(
    image_path: str, label_path: str, transform: alb.Compose, task: str
) -> dict:
    """Augment the image and the labels given a transformation to apply and the task at hand

    Arguments:
        image_path: a string denoting the path to the image
        label_path: a string denoting the path to the label file
        transform: the transformation to apply from the Albumentaton library
        task: a string denoting the chosen task between object detection "box" or instance segmentation "seg"

    Returns:
        transformed: a dictionary storing the transformed images, labels and class labels. The type of the label depends on the task at hand
    """
    # Open the image
    image = cv2.imread(image_path)
    if task == "box":
        # Open the labels and format it to the expected format by Albumentation for bounding boxes
        with open(label_path, "r") as label_file:
            boxes = label_file.readlines()
        bboxes = [box.split(" ")[1:] for box in boxes]
        class_labels = [box.split(" ")[0] for box in boxes]
        # Apply the transformation
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    elif task == "seg":
        # Open the labels and format it to the expected format by Albumentation for segmentation masks
        masks, class_labels = get_masks_from_label(label_path, image_path)
        # Apply the transformation
        transformed = transform(image=image, masks=masks, class_labels=class_labels)
    else:
        print("Wrong task specified. Only box and seg available")
    return transformed


def save_augmented(
    transformed: dict, transform_name: str, image_path: str, label_path: str, task: str
):
    """Saves the augmented images and labels

    Arguments:
        transformed: a dictionary storing the transformed images, labels and class labels. The type of the label depends on the task at hand
        transform_name: a string denoting a short name for the transformation applied used ot identify the transformed images and labels files
        image_path: a string denoting the path to the image
        label_path: a string denoting the path to the label file
        task: a string denoting the chosen task between object detection "box" or instance segmentation "seg"
    """
    # Retrieve the transformed image
    transformed_image = transformed["image"]
    transformed_class_labels = transformed["class_labels"]

    # Save the transformed image
    augmented_image_path = image_path.split(".")[0] + f"-{transform_name}.jpg"
    cv2.imwrite(augmented_image_path, transformed_image)

    transformed_labels = []

    if task == "box":
        # Retrieve the transformed bounding boxes
        bboxes = transformed["bboxes"]
        # Iterate through the transformed boxes and reformat them to the format expected by YOLO models for bounding boxes labels
        for idx, annotation in enumerate(bboxes):
            str_annotation = [str(elem) for elem in annotation]
            transformed_labels.append(
                " ".join([str(transformed_class_labels[idx])] + str_annotation)
            )
    elif task == "seg":
        masks = transformed["masks"]
        # Iterate through the transformed masks and reformat them to the format expected by YOLO models for segmentation labels
        for idx, mask in enumerate(masks):
            # Find the contours of the transformed object on the binary mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Retrieve the coordinates of the contour
            contour = contours[0] / mask.shape[::-1]
            # Format it
            str_annotation = [" ".join(point[0].astype(str)) for point in contour]
            transformed_labels.append(
                " ".join([str(transformed_class_labels[idx])] + str_annotation)
            )
    else:
        print("Wrong task specified. Only box and seg available")
    # Combine the label of each annotated objects with one per line
    transformed_labels = "\n".join(transformed_labels)

    # Save the transformed labels
    augmented_label_path = label_path.split(".")[0] + f"-{transform_name}.txt"
    with open(augmented_label_path, "w") as label_file:
        label_file.write(transformed_labels)


def main(path_data_train: str, task: str):
    """Main function applying the augmentations to images and labels and saving them

    Arguments:
        path_data_train: a string denoting the path to the training split of the dataset
        task: a string denoting the chosen task between object detection "box" or instance segmentation "seg"
    """
    if not exists(path_data_train):
        raise ValueError(
            f"{path_data_train} provided for training directory path does not exist"
        )
    if task not in ["box", "seg"]:
        raise ValueError(f"Wrong task specified : {task}. Only box and seg available.")
    # Retrieve all the training images
    images_list = glob(join(path_data_train, "images", "*.jpg"))

    # Define the transformations to apply : horizontal flip, vertical flip, diagonal flip, counter diagonal flip
    transformations = {
        "flip_lr": alb.Compose(
            [
                alb.HorizontalFlip(p=1),
            ],
            bbox_params=alb.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
        "flip_tb": alb.Compose(
            [
                alb.VerticalFlip(p=1),
            ],
            bbox_params=alb.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
        "flip_rev_diag": alb.Compose(
            [alb.HorizontalFlip(p=1), alb.VerticalFlip(p=1)],
            bbox_params=alb.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
        "flip_diag": alb.Compose(
            [alb.Transpose(p=1)],
            bbox_params=alb.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
    }

    # Iterate through the transformations defined earlier and apply them to the images and labels
    for transform_name, transformation in transformations.items():
        for image_path in tqdm(images_list):
            label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
            transformed = augment_image(image_path, label_path, transformation, task)
            save_augmented(transformed, transform_name, image_path, label_path, task)
        print(f"All images transformed with {transform_name} !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data_train", type=str)
    parser.add_argument("--task", type=str, default="seg")
    args = parser.parse_args()

    main(args.path_data_train, args.task)
