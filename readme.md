# Distraction-Aware Safety in Autonomous Driving

Authors: Ali Ghadami, Yingrui Huang, Jeeho Ahn

## Human Pose Estimation

## Object of Distraction Detection
Detection using a trained model can be tested by running `infer_v8.py` python file.



```
# required: opencv-python, ultralytics
# resulting image will be saved as 'output.jpg'

python infer_v8.py
```

For the full video frames, we run `run_frames_yolo.py` file, which does the same operation for multiple images.

## Pedestrian Trajectory Estimation