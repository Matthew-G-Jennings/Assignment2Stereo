#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "CalibrationIO.h"

int main(int argc, char* argv[]) {

	int maxDisparity = 32;      //must be a multiple of 16
	int blockSize = 11;         //must be odd
	bool setBlockMatch = false; //toggles between blockmatching and semi-global matching
								//true = block matching false = semi-global matching
	int scale = 4;			    //applies a scale to the images used for computation

	std::string image1File = "../bell_left.jpg"; //pathes to images to use.
	std::string image2File = "../bell_right.jpg";

	std::string calibrationFile = "../s_cal.txt"; // Path to camera calibration, in base project dir.
	cv::Size imageSize(1920, 1080); //defines the size in pixels of the images we are using.

	int SGP1 = 1000; //set values for P1/P2 values in semi global matching.
	int SGP2 = 1050; //default 0/0, higher values increase disparity change penalty. More smooth.

	//set the mode to use for semi global matching. Options: MODE_SGBM MODE_HH MODE_SGBM_3WAY MODE_HH4 	
	int mode = cv::StereoSGBM::MODE_HH4;

	cv::Mat K1, K2, R, T;
	std::vector<double> d1, d2;
	readStereoCalibration(calibrationFile, K1, d1, K2, d2, R, T);

	/*
	std::cout << "K1" << std::endl << K1 << std::endl;
	std::cout << "K2" << std::endl << K2 << std::endl;
	std::cout << "R" << std::endl << R << std::endl;
	std::cout << "T" << std::endl << T << std::endl;
	*/

	cv::Mat I1 = cv::imread(image1File);
	cv::Mat I2 = cv::imread(image2File);

	//std::cout << "Begin rectify" << std::endl;
	//Begin stereo rectification
	cv::Mat R1, R2, P1, P2, Q; // matrices for return values from stereoRectify.
	cv::stereoRectify(K1, d1, K2, d2, imageSize, R, T, R1, R2, P1, P2, Q);
	cv::Mat map1a, map1b , map2a, map2b;
	cv::initUndistortRectifyMap(K1, d1, R1, P1, imageSize, CV_32FC1, map1a, map1b);
	cv::initUndistortRectifyMap(K2, d2, R2, P2, imageSize, CV_32FC1, map2a, map2b);
	cv::remap(I1, I1, map1a, map1b, cv::INTER_LINEAR);
	cv::remap(I2, I2, map2a, map2b, cv::INTER_LINEAR);
	//From here images are rectified.
	//std::cout << "End rectify" << std::endl;

	//image rescaling
	cv::Mat smallImage1;
	cv::resize(I1, smallImage1, I1.size() / scale, 0, 0, cv::INTER_AREA);
	cv::Mat smallImage2;
	cv::resize(I2, smallImage2, I2.size() / scale, 0, 0, cv::INTER_AREA);

	cv::Mat disparity;

	//grey scale conversion
	cv::Mat I1GS;
	cv::cvtColor(smallImage1, I1GS, cv::COLOR_BGR2GRAY);
	cv::Mat I2GS;
	cv::cvtColor(smallImage2, I2GS, cv::COLOR_BGR2GRAY);

	if (setBlockMatch) { // do block matching

		cv::Ptr<cv::StereoBM> blockMatcher = cv::StereoBM::create(maxDisparity, blockSize);
		blockMatcher->compute(I1GS, I2GS, disparity);
	}
	else { // do semi global block matching

		cv::Ptr<cv::StereoSGBM> sgMatcher = cv::StereoSGBM::create(0, maxDisparity, blockSize,
			SGP1, SGP2, 0, 0, 0, 0, 0, mode);
		sgMatcher->compute(I1GS, I2GS, disparity);
	}

	//convert disparity map into a visible image
	cv::Mat display;
	disparity.convertTo(display, CV_8UC1, 255.0 / (16 * maxDisparity));

	//cv::namedWindow("left");
	//cv::namedWindow("right");
	cv::namedWindow("result");
	//cv::imshow("left", I1);
	//cv::imshow("right", I2);
	cv::imshow("result", display);
	cv::waitKey();

	return 0;

}
