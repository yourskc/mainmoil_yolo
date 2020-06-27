# Mainmoil Yolo

This is an OpenCV based MOIL application with YOLO real-time object detection.

## Algorithm


## Demo Video

[![Demo video](https://github.com/yourskc/mainmoil-mono-vo/blob/master/screenshot.png?raw=true)](https://www.youtube.com/watch?v=EQnlH8Dkjh0)


## Requirements
OpenCV 4.3.0 

## yolo3.weights
cd yolo-coco
wget https://pjreddie.com/media/files/yolov3.weights

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
./vo
```
## Before you run
In order to run this algorithm, you need to have either your own data, 
or else the sequences from [KITTI's Visual Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
In order to run this algorithm on your own data, you must modify the intrinsic calibration parameters in the code.

## Performance
![Results on the KITTI VO Benchmark](http://avisingh599.github.io/images/visodo/2K.png)

## Contact
For any queries, contact: avisingh599@gmail.com

## License
MIT