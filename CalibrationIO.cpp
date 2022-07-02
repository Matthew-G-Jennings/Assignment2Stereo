#include "CalibrationIO.h"
#include <fstream>

void writeMatrix(std::ostream& out, const cv::Mat& M) {
	for (int r = 0; r < M.rows; ++r) {
		for (int c = 0; c < M.cols; ++c) {
			out << M.at<double>(r, c) << " ";
		}
		out << std::endl;


	}
}

cv::Mat readMatrix(std::istream& in, int rows, int cols) {
	cv::Mat M(rows, cols, CV_64F);
	for (int r = 0; r < M.rows; ++r) {
		for (int c = 0; c < M.cols; ++c) {
			in >> M.at<double>(r, c);
		}
	}
	return M;
}

void saveStereoCalibration(const std::string& filename,
	const cv::Mat& K1, const std::vector<double>& distCoeff1,
	const cv::Mat& K2, const std::vector<double>& distCoeff2,
	const cv::Mat& R, const cv::Mat& T) {

	std::ofstream fout(filename);

	writeMatrix(fout, K1);
	fout << distCoeff1.size();
	for (double c : distCoeff1) {
		fout << " " << c;
	}
	fout << std::endl;

	writeMatrix(fout, K2);
	fout << distCoeff2.size();
	for (double c : distCoeff2) {
		fout << " " << c;
	}
	fout << std::endl;

	writeMatrix(fout, R);
	writeMatrix(fout, T);

	fout.close();
}

void readStereoCalibration(const std::string& filename,
	cv::Mat& K1, std::vector<double>& distCoeff1,
	cv::Mat& K2, std::vector<double>& distCoeff2,
	cv::Mat& R, cv::Mat& T) {

	std::ifstream fin(filename);

	K1 = readMatrix(fin, 3, 3);
	size_t numCoeffs;
	fin >> numCoeffs;

	distCoeff1.resize(numCoeffs);
	for (double& c : distCoeff1) {
		fin >> c;
	}

	K2 = readMatrix(fin, 3, 3);
	fin >> numCoeffs;

	distCoeff2.resize(numCoeffs);
	for (double& c : distCoeff1) {
		fin >> c;
	}


	R = readMatrix(fin, 3, 3);
	T = readMatrix(fin, 3, 1);

	fin.close();
}