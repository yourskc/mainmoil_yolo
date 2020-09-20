# Mainmoil Yolo

This is an OpenCV based MOIL application with YOLO real-time object detection.

## Algorithm


## Demo Video

[![Demo video](https://github.com/yourskc/mainmoil_yolo/blob/master/Screenshot.png?raw=true)](https://www.youtube.com/watch?v=i3c43llwoFc)


## Requirements

OpenCV 4.3.0 

cd moildev_install
cd { your OpenCV major version and platform folder }
sudo ./install.sh

## Download yolo3.weights

The weights file is not included in this repository, please download it by yourself.

cd yolo-coco

wget https://pjreddie.com/media/files/yolov3.weights

## CUDA support

The CUDA support is by default enabled, if you don't have CUDA, please disable it by remark the line in src/yolo.cpp
 
// #define USE_GPU true

## How to compile?

Provided with this repo is a CMakeLists.txt file, which you can use to directly compile the code as follows:

```bash
mkdir build
cd build
cmake ..
make
```

## How to run? 
After compilation, in the build directly, type the following:
```bash
./mainmoil_yolo
```
## Before you run
In order to run this algorithm, you need to have either your image sequence data, or a video file.




