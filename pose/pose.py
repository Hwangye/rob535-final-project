from mmpose.apis import MMPoseInferencer
import numpy as np
import cv2
import os

class PedestrianAnalyzer:
    def __init__(self):
        self.inferencer = MMPoseInferencer('human')
    
    def analyze_head_direction(self, image_path):
        distracted_boxes = []
        image = cv2.imread(image_path)
        result_generator = self.inferencer(image_path, show=False)
        result = next(result_generator)
        predictions = result['predictions'][0]
        warnings = []
        for pred in predictions:
            keypoints = pred['keypoints']
            x_coords = [kp[0] for kp in keypoints]
            y_coords = [kp[1] for kp in keypoints]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            pred['area'] = (x_max - x_min) * (y_max - y_min)
    
        sorted_predictions = sorted(predictions, key=lambda x: x['area'], reverse=True)#[:4]
        for pred in sorted_predictions:
            keypoints = pred['keypoints']
            scores = pred['keypoint_scores']
            direction = self._analyze_single_person(keypoints, scores)
            self._draw_warning(image, keypoints, direction)
            #if direction != 0:
            #    warnings.append(direction)
            if direction == 3:
                x_coords = [kp[0] for kp in keypoints]
                y_coords = [kp[1] for kp in keypoints]
                bbox = {'x_min': int(min(x_coords)),'y_min': int(min(y_coords)),
                        'x_max': int(max(x_coords)),'y_max': int(max(y_coords))}
                distracted_boxes.append(bbox)
        #cv2.imwrite('resulto.jpg', image)
        with open('distracted_pedestrians.txt', 'a') as f:
            f.write(f"\nImage: {image_path}\n")
            for i, box in enumerate(distracted_boxes):
                f.write(f"Pedestrian {i+1}: {box}\n")
    
        return image, warnings
    
    def _analyze_single_person(self, keypoints, scores):
        nose = keypoints[0]
        left_ear = keypoints[3]
        right_ear = keypoints[4]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]

        nose_visible = scores[0] > 0.5
        left_ear_visible = scores[3] > 0.5
        right_ear_visible = scores[4] > 0.5

        
        # 0: front, 1: distracted, 2: unknown, 3: hold phone
        flag = 0
        if left_elbow[1] > left_wrist[1] or right_elbow[1] > right_wrist[1]:
            flag = 3

        if left_ear_visible and not right_ear_visible:
            return 1
        elif right_ear_visible and not left_ear_visible:
            return 1
        elif left_ear_visible and right_ear_visible:
            if nose_visible:
                ear_center = (left_ear[0] + right_ear[0]) / 2
                nose_offset = abs(nose[0] - ear_center)
                if nose_offset < 10.0:
                    return flag
                else:
                    return flag if flag == 3 else 1
        return 2

    def _draw_warning(self, image, keypoints, direction):
        show_points = [0,3, 4,7, 8,9, 10]
        x_coords = [kp[0] for kp in keypoints]
        y_coords = [kp[1] for kp in keypoints]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        color =  (0, 255, 0)
        if direction == 1 or direction == 2:
            color = (0, 0, 255)
        elif direction == 3:
            color = (0, 255, 255)
            cv2.rectangle(image, (int(x_min), int(y_min)),(int(x_max), int(y_max)), color, 5)
        #for i in show_points:
        #    x, y = keypoints[i]
        #    cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)

def main():
    analyzer = PedestrianAnalyzer()
    image_folder = "pics"
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) ]
    for i, image_path in enumerate(image_paths):
        result_image, warnings = analyzer.analyze_head_direction(image_path)
        cv2.imwrite(f'result_{i+1}.jpg', result_image)
    

    #if warnings:
    #    print(f"Warning:{len(warnings)} pedestrains distracted!")

if __name__ == "__main__":
    main()