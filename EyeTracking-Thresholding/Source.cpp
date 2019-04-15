#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <chrono>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame, int runTime);

/** Global variables */
string window_name = "Capture - Face detection";
RNG rng(12345);
vector<Vec4i> hierarchy;

int stabilizingWindow = 6;
bool TLready, BRready, TLcleared, BRcleared;

Point TL, BR, screenPoint, centerGlobal;
vector<Point> centers;

/** @function main */
int main(int argc, const char** argv)
{
	VideoCapture capture("C:\\Users\\rrung\\Downloads\\kasperEye.mp4");
	int w = capture.get(CAP_PROP_FRAME_WIDTH);
	int h = capture.get(CAP_PROP_FRAME_HEIGHT);
	Mat frame;

	TLready = false;
	BRready = false;
	TLcleared = false;
	BRcleared = false;

	auto start_time = chrono::high_resolution_clock::now();

	//-- 2. Read the video stream
	for (int i = 0; i < capture.get(CAP_PROP_FRAME_COUNT); i++) {
		capture >> frame;

			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
				auto current_time = chrono::high_resolution_clock::now();
				int runTime = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
				detectAndDisplay(frame, runTime);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			int c = waitKey(1);
			if ((char)c == 'c') { break; }
	}
	return 0;
}

int remap(int value, int start1, int stop1, int start2, int stop2)
{
	int outgoing = (start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1)));
	if (outgoing < 0)
		outgoing = -outgoing;
	return outgoing;
}

int biggestContour(vector<vector<Point>> contours)
{
	int bigIndex = 0, i=0;
	double bigArea = 0;

	for (vector<vector<Point>>::iterator it = contours.begin(); it != contours.end(); it++, i++) {
		if (contourArea(*it) > bigArea)
		{
			bigArea = contourArea(*it);
			bigIndex = i;
		}
	}
	return bigIndex;
}

Point stabilize(vector<Point> &points, int windowSize)
{
	float sumX = 0;
	float sumY = 0;
	int count = 0;
	for (int i = max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
	{
		sumX += points[i].x;
		sumY += points[i].y;
		++count;
	}
	if (count > 0)
	{
		sumX /= count;
		sumY /= count;
	}
	return Point(sumX, sumY);
}


/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, int runTime)
{
	vector<Vec3f> circles;
	Vec3f eyeball;
	Mat frame_gray;

	Point2f pupCenter;
	float pupRadius;

	centerGlobal.x = frame.cols / 2;
	centerGlobal.y = frame.rows / 2;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	GaussianBlur(frame_gray, frame_gray, Size(7, 7),0,0);
	//inRange(frame_gray, Scalar(14,18,20), Scalar(25,30,38), frame_gray);
	threshold(frame_gray, frame_gray, 2, 255, 1);

	Mat elem1 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(frame_gray, frame_gray, MORPH_CLOSE, elem1);
	
	vector<vector<Point>> contours;
	findContours(frame_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	minEnclosingCircle((Mat)contours[biggestContour(contours)], pupCenter, pupRadius);
	circle(frame, pupCenter, (int)pupRadius, Scalar(0,255,0), 2, 8, 0);
	line(frame, Point(0, pupCenter.y), Point(frame.cols, pupCenter.y), Scalar(0, 0, 255), 2);
	line(frame, Point(pupCenter.x, 0), Point(pupCenter.x, frame.rows), Scalar(0, 0, 255), 2);

	//drawContours(frame, contours, biggestContour(contours), Scalar(0, 0, 255), -1);



	imshow("Eye", frame);
	imshow("Processed", frame_gray);
	/*
	if (runTime >= 4 && runTime < 7)
	{
		circle(frame, Point(15, 15), 15, Scalar(255, 255, 255), -1);
	}
	else if (runTime >= 7 && runTime < 12)
	{
		if (TLcleared == false)
		{
			centers.clear();
			TLcleared = true;
		}
		circle(frame, Point(15, 15), 15, Scalar(0, 255, 0), -1);
	}
	else if (runTime >= 12 && runTime < 15)
	{
		if (TLready == false)
		{
			TL = stabilize(centers, stabilizingWindow);
			TLready = true;
		}
		circle(frame, Point(frame.cols - 15, frame.rows - 15), 15, Scalar(255, 255, 255), -1);
	}
	else if (runTime >= 15 && runTime < 20)
	{
		if (BRcleared == false)
		{
			centers.clear();
			BRcleared = true;
		}
		circle(frame, Point(frame.cols - 15, frame.rows - 15), 15, Scalar(0, 255, 0), -1);
	}
	else if (runTime >= 20)
	{
		if (BRready == false)
		{
			BR = stabilize(centers, stabilizingWindow);
			BRready = true;
		}
		cout << "BR X: " << BR.x << "BR Y: " << BR.y << endl;
		cout << "TL X: " << TL.x << "TL Y: " << TL.y << endl;

		cout << "Global pos x: " << centerGlobal.x << "Global pos y: " << centerGlobal.y << endl;

		screenPoint.x = remap(centerGlobal.x, TL.x, BR.x, 15, frame.rows - 15);
		cout << "Screen point x: " << screenPoint.x << endl;
		screenPoint.y = remap(centerGlobal.y, TL.y, BR.y, 15, frame.cols - 15);
		cout << "Screen point y: " << screenPoint.y << endl;
		circle(frame, screenPoint, 10, Scalar(0, 0, 255), -1);
	}
	*/
}