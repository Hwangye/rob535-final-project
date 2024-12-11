from ultralytics import YOLO
import cv2

# Path to your custom YOLOv8 model
#model_path = "yolov8c.pt"  # e.g. 'best.pt'
model_path = "yolov8x-oiv7.pt"
#model_path = "train_v11/runs/detect/train27/weights/best.pt"
#model_path = "yolov8x-oiv7_openimgs.pt"

# Path to the input image
#image_path = "pedestrians.jpg"  # e.g. 'test.jpg'
image_path = "result_132.jpg"

# Load the YOLOv8 model
model = YOLO(model_path)

# Perform inference
results = model.predict(image_path)

# The results list contains results for each image. Since we have one image, let's take the first element.
res = results[0]

# Print out the boxes, which is a 'Boxes' object. To get the tensor:
print(res.boxes.xyxy)     # bounding boxes in [x1, y1, x2, y2] format
print(res.boxes.conf)     # confidence scores
print(res.boxes.cls)      # class IDs

desired_classes = [42,46]

# Filter boxes, confidences, and class IDs
keep_indices = [i for i, c in enumerate(res.boxes.cls) if int(c) in desired_classes]

#filtered_boxes = res.boxes[keep_indices]

# Now render the image only with the filtered boxes
#res.boxes = filtered_boxes
annotated_image = res.plot()

# To visualize the results, YOLO provides a built-in show() or you can manually draw boxes using OpenCV:
#annotated_image = res.plot()  # returns a numpy array with drawings

# Display the image with OpenCV
#cv2.imshow("Detections", annotated_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# If you want to save the result image:
cv2.imwrite("output.jpg", annotated_image)
