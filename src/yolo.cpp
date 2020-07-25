#include "yolo.h"
#include "moildev.h"
#define USE_GPU true
// #define TEST_ORIGINAL
const std::string modelConfiguration = "../yolo-coco/yolov3.cfg";
const std::string modelWeights = "../yolo-coco/yolov3.weights";
const std::string classesFile = "../yolo-coco/coco.names";
#define imageFilesFmt "../record/50/img%03d.jpg"

void moil_yolo_images()
{

	moil_init();

	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Front", WINDOW_NORMAL);
	namedWindow("Left", WINDOW_NORMAL);
	namedWindow("Right", WINDOW_NORMAL);
	namedWindow("Down", WINDOW_NORMAL);
	namedWindow("Lower left", WINDOW_NORMAL);
	namedWindow("Lower right", WINDOW_NORMAL);

	moveWindow("Input", 1200, 0);

	moveWindow("Left", 40, 40);
	moveWindow("Front", 660, 40);
	moveWindow("Right", 1260, 40);
	moveWindow("Lower left", 0, 520);
	moveWindow("Down", 660, 520);
	moveWindow("Lower right", 1260, 520);
	resizeWindow("Lower right", 600, 450);

	char filename[100];
	string Title[6] = {"Front", "Left", "Right", "Down", "Lower left", "Lower right"};
	dnn_prepare(modelWeights, modelConfiguration, classesFile);
	int numFrame = 1;
	bool isExit = false;

	while (!isExit)
	{
		sprintf(filename, imageFilesFmt, numFrame);
		cout << filename << endl;
		image_input = imread(filename);
		if (image_input.empty())
			isExit = true;
		else
		{
			cv::resize(image_input, image_input_s,
					   Size(640, 480));
			cout << "process" << endl;
			imshow("Input", image_input_s);
#ifdef TEST_ORIGINAL
				cv::resize(image_input, image_display[0],
						   Size(1920, 1080));
				detect_image(image_display[0]);
				imshow("image_input", image_display[0]);
#else
			for (int i = 0; i < 6; i++)
			{
				remap(image_input, image_result[i], mapX[i], mapY[i], INTER_CUBIC,
					  BORDER_CONSTANT, Scalar(0, 0, 0));
				cv::resize(image_result[i], image_display[i],
						   Size(600, 450));
				detect_image(image_display[i]);
				imshow(Title[i], image_display[i]);				
			}
#endif

		}
		char c = waitKey(200);
		if (c == 27) // esc to quit
			isExit = true;
		else if(c == 32){ // space to pause
			c = 0;
			while (c != 32 && c!= 27) {
				c = waitKey(200);			
			}
			}
		numFrame += 1;
	}
	waitKey(0);
}

void moil_yolo_video(string video_path)
{

	moil_init();

	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("Front", WINDOW_NORMAL);
	namedWindow("Left", WINDOW_NORMAL);
	namedWindow("Right", WINDOW_NORMAL);
	namedWindow("Down", WINDOW_NORMAL);
	namedWindow("Lower left", WINDOW_NORMAL);
	namedWindow("Lower right", WINDOW_NORMAL);

	moveWindow("Input", 1200, 0);

	moveWindow("Left", 40, 40);
	moveWindow("Front", 660, 40);
	moveWindow("Right", 1260, 40);
	moveWindow("Lower left", 0, 520);
	moveWindow("Down", 660, 520);
	moveWindow("Lower right", 1260, 520);
	resizeWindow("Lower right", 600, 450);

	char filename[100];
	string Title[6] = {"Front", "Left", "Right", "Down", "Lower left", "Lower right"};
	dnn_prepare(modelWeights, modelConfiguration, classesFile);
	int numFrame = 1;
	bool isExit = false;

	VideoCapture cap;
	try
	{
		ifstream ifile(video_path);
		if (!ifile)
			throw("error");
		cap.open(video_path);
	}
	catch (...)
	{
		cout << "Could not open the input image/video stream" << endl;
		return;
	}

	while (!isExit)
	{
		sprintf(filename, "frame %03d", numFrame);
		cout << filename << endl;

		cap >> image_input;
		if (!image_input.empty())
		{
			cv::resize(image_input, image_input_s,
					   Size(640, 480));
			cout << "process" << endl;
			imshow("Input", image_input_s);
			for (int i = 0; i < 6; i++)
			{
				remap(image_input, image_result[i], mapX[i], mapY[i], INTER_CUBIC,
					  BORDER_CONSTANT, Scalar(0, 0, 0));
				cv::resize(image_result[i], image_display[i],
						   Size(600, 450));
				detect_image(image_display[i]);
				imshow(Title[i], image_display[i]);
			}
		}
		else
			isExit = true;

		char c = waitKey(30);
		if (c == 27)
			isExit = true;
		numFrame += 1;
	}
	waitKey(0);
	cap.release();
}

void moil_init()
{
	md = new Moildev();
	int w = 2592, h = 1944;
	double m_ratio;

	md->Config("rpi_220", 1.4, 1.4,
			   1320.0, 1017.0, 1.048,
			   2592, 1944, 3.4, // 4.05
			   // 0, 0, 0, 0, -47.96, 222.86
			   0, 0, 0, 10.11, -85.241, 282.21);

	double calibrationWidth = md->getImageWidth();
	m_ratio = w / calibrationWidth;

	for (uint i = 0; i < 6; i++)
	{
		mapX[i] = Mat(h, w, CV_32F);
		mapY[i] = Mat(h, w, CV_32F);
		image_result[i] = Mat(h, w, CV_32F);
	}
	double zoom = 6;
	md->AnyPointM2((float *)mapX[0].data, (float *)mapY[0].data,
				   mapX[0].cols, mapX[0].rows, 0, 0, zoom, m_ratio); // front view
	md->AnyPointM2((float *)mapX[1].data, (float *)mapY[1].data,
				   mapX[1].cols, mapX[1].rows, 0, -60, zoom, m_ratio); // left view
	md->AnyPointM2((float *)mapX[2].data, (float *)mapY[2].data,
				   mapX[2].cols, mapX[2].rows, 0, 60, zoom, m_ratio); // right view
	md->AnyPointM2((float *)mapX[3].data, (float *)mapY[3].data,
				   mapX[3].cols, mapX[3].rows, -60, 0, zoom, m_ratio); // Down view
	md->AnyPointM2((float *)mapX[4].data, (float *)mapY[4].data,
				   mapX[4].cols, mapX[4].rows, -50, -50, zoom, m_ratio); // Lower left view
	md->AnyPointM2((float *)mapX[5].data, (float *)mapY[5].data,
				   mapX[5].cols, mapX[5].rows, -50, 50, zoom, m_ratio); // Lower right view
}

void dnn_prepare(string modelWeights, string modelConfiguration, string classesFile)
{

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
		classes.push_back(line);
	// Load the network
	net = readNetFromDarknet(modelConfiguration, modelWeights);
#if USE_GPU
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA);
#else
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL);
#endif
}
void detect_image(Mat &frame)
{
	// Create a 4D blob from a frame.
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// Remove the bounding boxes with low confidence
	postprocess(frame, outs);
	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	// putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
}

void detect_image_file(string image_path, string modelWeights, string modelConfiguration, string classesFile)
{
	// Load names of classes
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
		classes.push_back(line);

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	cv::Mat frame = cv::imread(image_path);
	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Stop the program if reached end of video
	// Create a 4D blob from a frame.
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// Remove the bounding boxes with low confidence
	postprocess(frame, outs);
	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.25, Scalar(0, 0, 255));
	// Write the frame with the detection boxes
	imshow(kWinName, frame);
	cv::waitKey(30);
}

void detect_video(string video_path, string modelWeights, string modelConfiguration, string classesFile)
{
	string outputFile = "output_videos/output.avi";
	;
	// Load names of classes
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
		classes.push_back(line);

	// Load the network
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	VideoCapture cap;
	//VideoWriter video;
	Mat frame, blob;

	try
	{
		// Open the video file
		ifstream ifile(video_path);
		if (!ifile)
			throw("error");
		cap.open(video_path);
	}
	catch (...)
	{
		cout << "Could not open the input image/video stream" << endl;
		return;
	}

	// Get the video writer initialized to save the output video
	//video.open(outputFile,
	//	VideoWriter::fourcc('M', 'J', 'P', 'G'),
	//	28,
	//	Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Process frames.
	while (waitKey(1) < 0)
	{
		// get frame from the video
		cap >> frame;

		// Stop the program if reached end of video
		if (frame.empty())
		{
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << outputFile << endl;
			waitKey(3000);
			break;
		}
		// Create a 4D blob from a frame.
		blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255));

		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		//video.write(detectedFrame);
		imshow(kWinName, frame);
	}

	cap.release();
	//video.release();
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat &frame, const vector<Mat> &outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float *data = (float *)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
				 box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net &net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}
int main(int argc, char **argv)
{
	moil_yolo_images();
	// moil_yolo_video("../input_videos/toy_traffic.mp4");
}
