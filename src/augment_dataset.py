import cv2
import albumentations as alb
from glob import glob
import argparse
from os.path import join
import numpy as np
from tqdm import tqdm


def get_masks_from_label(label_path: str, image_path: str) -> np.array:

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_width = img.shape[1]
    image_height = img.shape[0]

    with open(label_path, "r") as label_file:
        labels = label_file.readlines()

    masks = []
    class_labels = []

    for line in labels:

        class_labels.append(line.split()[0])

        points = list(map(float, line.split()[1:]))

        contours = [
            np.array(
                [
                    (int(image_width * points[i]), int(image_height * points[i + 1]))
                    for i in range(0, len(points), 2)
                ],
                dtype=np.int32,
            )
        ]

        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        mask = cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        masks.append(mask)

    return masks, class_labels


def augment_image(
    image_path: str, label_path: str, transform: alb.Compose, task: str
) -> dict:
    image = cv2.imread(image_path)
    if task == "box":
        with open(label_path, "r") as label_file:
            boxes = label_file.readlines()
        bboxes = [box.split(" ")[1:] for box in boxes]
        class_labels = [box.split(" ")[0] for box in boxes]
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    elif task == "seg":
        masks, class_labels = get_masks_from_label(label_path, image_path)
        transformed = transform(image=image, masks=masks, class_labels=class_labels)
    else:
        print("Wrong task specified. Only box and seg available")
    return transformed


def save_augmented(
    transformed: dict, transform_name: str, image_path: str, label_path: str, task: str
):
    transformed_image = transformed["image"]
    transformed_class_labels = transformed["class_labels"]

    augmented_image_path = image_path.split(".")[0] + f"-{transform_name}.jpg"
    cv2.imwrite(augmented_image_path, transformed_image)

    transformed_labels = []

    if task == "box":
        bboxes = transformed["bboxes"]
        for idx, annotation in enumerate(bboxes):
            str_annotation = [str(elem) for elem in annotation]
            transformed_labels.append(
                " ".join([str(transformed_class_labels[idx])] + str_annotation)
            )
    elif task == "seg":
        masks = transformed["masks"]
        for idx, mask in enumerate(masks):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            try:
                contour = contours[0] / mask.shape[::-1]
            except Exception as e:
                print(len(masks))
                print(contours)
                print(image_path)
                print(label_path)
                raise e
            str_annotation = [" ".join(point[0].astype(str)) for point in contour]
            transformed_labels.append(
                " ".join([str(transformed_class_labels[idx])] + str_annotation)
            )
    else:
        print("Wrong task specified. Only box and seg available")

    transformed_labels = "\n".join(transformed_labels)

    augmented_label_path = label_path.split(".")[0] + f"-{transform_name}.txt"

    with open(augmented_label_path, "w") as label_file:
        label_file.write(transformed_labels)


def main(path_data_train: str, task: str):
    images_list = glob(join(path_data_train, "images", "*.jpg"))

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

    for transform_name, transformation in transformations.items():
        for image_path in tqdm(images_list):
            label_path = image_path.replace("images","labels").replace(".jpg",".txt")
            transformed = augment_image(image_path, label_path, transformation, task)
            save_augmented(transformed, transform_name, image_path, label_path, task)
        print(f"All images transformed with {transform_name} !")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for processing files.")
    parser.add_argument("--path_data_train", type=str)
    parser.add_argument("--task", type=str, default="seg")
    args = parser.parse_args()

    main(args.path_data_train, args.task)
