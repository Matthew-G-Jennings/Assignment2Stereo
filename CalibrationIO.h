#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

void saveStereoCalibration(const std::string& filename,
	const cv::Mat& K1, const std::vector<double>& distCoeff1,
	const cv::Mat& K2, const std::vector<double>& distCoeff2,
	const cv::Mat& R, const cv::Mat& T);

void readStereoCalibration(const std::string& filename,
	cv::Mat& K1, std::vector<double>& distCoeff1,
	cv::Mat& K2, std::vector<double>& distCoeff2,
	cv::Mat& R, cv::Mat& T);