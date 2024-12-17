import ultralytics
from ultralytics import YOLO
from os.path import basename, join, exists
import argparse
import warnings


def main(
    path_model: str,
    name_dataset: str,
    epochs: int,
    imgsz_small: int,
    imgsz_big: int,
    batch_size: int,
    workers: int,
    device: str,
):
    """Main function to train the models

    Arguments:
        path_model: a string denoting the path to the pre-trained model to use. Will be downloaded if not existing
        name_dataset: a string denoting the name of the dataset to find the config files for the small and big objects models
        epochs: a scalar denoting the number of epochs to train
        imgsz_small: a scalar denoting the shape (imgsz_small, imgsz_small) to which images will be rescaled for small objects model
        imgsz_big: a scalar denoting the shape (imgsz_big, imgsz_big) to which images will be rescaled for big objects model
        batch_size: a scalar denoting the number of images per batch during training
        workers: a scalar denoting the number of workers to use for training
        device: a string denoting the device to use for training
    """
    if not exists(path_model):
        warnings.warn(
            f"{path_model} does not exist. It will be downloaded", UserWarning
        )
    if epochs <= 0 or imgsz_small <= 0 or imgsz_big <= 0:
        raise ValueError(
            f"The number of epochs or the image size is negative. It must be strictly greater than zero"
        )
    if device not in ["cpu", "0", "1", "mps"]:
        raise ValueError(
            f"{device} is not in the list of posible devices : ['cpu','0','1','mps']"
        )

    small_config_path = join("configs", name_dataset + "_small_labels" + ".yaml")
    train_name_small = f"{basename(path_model).split(".")[0]}_{name_dataset + "_small_labels"}_epochs-{epochs}_imgsz-{imgsz_small}_batch-{batch_size}"
    if device in ["0", "1"]:
        device = int(device)
    model = YOLO(path_model)
    model.train(
        data=small_config_path,
        epochs=epochs,
        imgsz=imgsz_small,
        device=device,
        name=train_name_small,
        seed=0,
        pretrained=True,
        plots=True,
        save=True,
        single_cls=True,
        workers=workers,
    )

    big_config_path = join("configs", name_dataset + "_big_labels" + ".yaml")
    train_name_big = f"{basename(path_model).split(".")[0]}_{name_dataset + "_big_labels"}_epochs-{epochs}_imgsz-{imgsz_big}_batch-{batch_size}"
    model = YOLO(path_model)
    model.train(
        data=big_config_path,
        epochs=epochs,
        imgsz=imgsz_big,
        device=device,
        name=train_name_big,
        seed=0,
        pretrained=True,
        plots=True,
        save=True,
        single_cls=True,
        workers=workers,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model", type=str, default="models/yolo11n-seg.pt")
    parser.add_argument(
        "--name_dataset", type=str, default="ctrst-0-255_srfc-200_prcs-0_seg"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz_small", type=int, default=190)
    parser.add_argument("--imgsz_big", type=int, default=1520)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    ultralytics.checks()

    main(
        args.path_model,
        args.name_dataset,
        args.epochs,
        args.imgsz_small,
        args.imgsz_big,
        args.batch_size,
        args.workers,
        args.device,
    )
