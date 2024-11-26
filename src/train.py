from ultralytics import YOLO
from os.path import basename

def main(path_model: str, path_dataset: str, epochs: int, device: str):
    train_name = f"{basename(path_model).split(".")[0]}_{basename(path_dataset).split(".")[0]}_epochs-{epochs}"
    model = YOLO(path_model)
    model.train(data=path_dataset, epochs=epochs, imgsz=1616, device=device, name=train_name, seed=0, pretrained=True, plots=True, save=True, single_cls=True,save_period=10)

if __name__ == "__main__":

    PATH_MODEL = "./models/yolo11/yolo11s.pt"
    PATH_DATASET = "./ctrst-210-255.yaml"
    EPOCHS = 100
    DEVICE = "cpu"
    TRAIN_NAME = f"yolo11s_210-255_{EPOCHS}"

    main(PATH_MODEL, PATH_DATASET, EPOCHS, DEVICE)
