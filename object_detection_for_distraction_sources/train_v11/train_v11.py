from ultralytics import YOLO

model = YOLO("testv11.yaml")

results = model.train(data='data.yaml', epoch=30, imgsz=640)