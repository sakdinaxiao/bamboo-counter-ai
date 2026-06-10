from ultralytics import YOLO
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parent

data_yaml = project_root / "Couting_dataset_Clahe" / "data.yaml"

output_path = project_root/ f"training_result"


model = YOLO("yolo26n.pt")

def get_available_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def training_model(modelname):
    if not data_yaml.exists():
        print("Wrong data path")
        return
    print("start training")
    results = model.train(
        data=data_yaml,
        project=output_path,
        name=modelname,

        epochs=150,
        imgsz=640,
        batch=16,
        workers=4,

        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.2,
        perspective=0.0,
        erasing=0.1, 

        device=get_available_device(),
        patience=50,
        close_mosaic=10
    )

    if not results:
        print("fail to train model")
    else:
        print("Model training done")

def test_set_unbiased_benchmark(modelname):
    model_path = output_path/ modelname / 'weights' / 'best.pt'
    
    if not model_path.exists():
        print("Cannot find model")
        return
    
    model_trained = YOLO(model_path)

    metrics = model_trained.val(
        data=data_yaml,
        split="test",
        project=output_path,
        name=modelname+"_test_result",
        plots=True
    )

    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr    

    print("\n" + "-"*20)
    print(f"mAP@50:    {map50:.4f}")
    print(f"mAP@50-95: {map50_95:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    

    

if __name__ == "__main__":
    #train model using colab
    test_set_unbiased_benchmark("detection_small")
