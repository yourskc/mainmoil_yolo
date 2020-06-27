#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
#include "moildev.h"

using namespace cv;
using namespace dnn;
using namespace std;

Moildev *md;
Mat image_input, image_input_s;
Mat image_display[6];
Mat mapX[6], mapY[6];
Mat image_result[6];

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image
vector<string> classes;
Net net;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat &frame, const vector<Mat> &out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net &net);
void dnn_prepare(string modelWeights, string modelConfiguration, string classesFile);
void detect_image(Mat &image);
void detect_image_file(string image_path, string modelWeights, string modelConfiguration, string classesFile);
void detect_video(string video_path, string modelWeights, string modelConfiguration, string classesFile);

void moil_yolo_images();
void moil_yolo_video(string video_path);
void moil_init();