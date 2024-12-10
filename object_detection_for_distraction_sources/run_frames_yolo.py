import cv2
import os
from ultralytics import YOLO

# Load the pretrained YOLO model (e.g., YOLOv8)
model = YOLO('yolov8x-oiv7.pt') # You can use 'yolov8s.pt', 'yolov8m.pt', etc.

# Path to the directory containing the input images
input_dir = 'demo_frames/input'
output_dir = 'demo_frames/output'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Path to the log file for images containing object index 42 and their bounding boxes
log_file_path = 'demo_frames/images_with_index_42.txt'

# List all image files in the input directory, sorted by frame order
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Open the log file for writing
with open(log_file_path, 'w') as log_file:
    # Process each image
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        # Perform object detection
        results = model(image)
        
        # Get the annotated image
        #annotated_image = results[0].plot()

        # Track if any object with class index 42 is found
        detected_boxes = []
        
        # Iterate through the results to find objects with class index 42
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == 42:
                    # Get the bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_boxes.append((x1, y1, x2, y2))

        res = results[0]

        desired_classes = [42,339]

        # Filter boxes, confidences, and class IDs
        keep_indices = [i for i, c in enumerate(res.boxes.cls) if int(c) in desired_classes]

        filtered_boxes = res.boxes[keep_indices]

        # Now render the image only with the filtered boxes
        res.boxes = filtered_boxes
        
        annotated_image = res.plot()
        
        # Save the resulting image in the output directory
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, annotated_image)

        # If objects with index 42 were found, log the image file and bounding boxes
        if detected_boxes:
            box_strings = ', '.join([f"({x1}, {y1}, {x2}, {y2})" for x1, y1, x2, y2 in detected_boxes])
            log_file.write(f"{image_file} - Bounding Boxes: {box_strings}\n")
            print(f"Object(s) with index 42 found in: {image_file}")
        
        print(f"Processed and saved: {output_path}")

print("All images processed successfully.")
