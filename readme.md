# Distraction-Aware Safety in Autonomous Driving

Authors: Ali Ghadami, Yingrui Huang, Jeeho Ahn

## Human Pose Estimation
First install open-mmlab toolbox
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmpose>=1.1.0"
```
Then run  `python pose.py` for result. Change path to your image in the file.

## Object of Distraction Detection
Detection using a trained model can be tested by running `infer_v8.py` python file.



```
# required: opencv-python, ultralytics
# resulting image will be saved as 'output.jpg'

python infer_v8.py
```

For the full video frames, we run `run_frames_yolo.py` file, which does the same operation for multiple images.

## Pedestrian Trajectory Estimation
