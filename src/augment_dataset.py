import cv2
import albumentations as alb
from glob import glob
import argparse
from os.path import join


def augment_image(image_path: str, label_path: str, transform: alb.Compose) -> dict:
    image = cv2.imread(image_path)
    with open(label_path, "r") as label_file:
        boxes = label_file.readlines()
    bboxes = [box.split(" ")[1:] for box in boxes]
    class_labels = [box.split(" ")[0] for box in boxes]
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed


def save_augmented(
    transformed: dict, transform_name: str, image_path: str, label_path: str
):
    transformed_image = transformed["image"]
    bboxes = transformed["bboxes"]
    transformed_class_labels = transformed["class_labels"]
    cv2.imwrite(image_path.split(".")[0] + f"-{transform_name}.jpg", transformed_image)

    transformed_labels = []
    for idx, annotation in enumerate(bboxes):
        str_annotation = [str(elem) for elem in annotation]
        transformed_labels.append(
            " ".join([str(transformed_class_labels[idx])] + str_annotation)
        )
    transformed_labels = "\n".join(transformed_labels)

    with open(label_path.split(".")[0] + f"-{transform_name}.txt", "w") as label_file:
        label_file.write(transformed_labels)


def main(path_data_train: str):
    images_list = glob(join(path_data_train, "images", "*.jpg"))
    labels_list = glob(join(path_data_train, "labels", "*.txt"))

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
        "full_flip": alb.Compose(
            [alb.HorizontalFlip(p=1), alb.VerticalFlip(p=1)],
            bbox_params=alb.BboxParams(format="yolo", label_fields=["class_labels"]),
        ),
    }

    for transform_name, transformation in transformations.items():
        for idx, image_path in enumerate(images_list):
            transformed = augment_image(image_path, labels_list[idx], transformation)
            save_augmented(transformed, transform_name, image_path, labels_list[idx])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script for processing files.")
    parser.add_argument("--path_data_train", type=str)
    args = parser.parse_args()

    main(args.path_data_train)
