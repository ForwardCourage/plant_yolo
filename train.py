from ultralytics import YOLO
import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
if __name__ == '__main__':
    print("Start importing model...")
    model = YOLO("yolov8m.pt")
    print("Model imported!")


    print("Start training...")
    model.train(data = "datasets/data.yaml", epochs = 1)
    print("Training complete!")

