#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <assert.h>


//using namespace std;
//using namespace cv;
/* elemT is simple type, it maybe float or int, so we will have very simple code in manipulation */

#ifndef CV_PI
#define CV_PI 3.1415926535897932384626433832795
#endif /*CV_PI */
#define SQRT_2PI 2.50662827463

static float gradiant_x[3][3] = {
	{-1.0, 0, 1.0},
	{-2.0, 0, 2.0},
	{-1.0, 0, 1.0}
};

static float gradiant_y[3][3] = {
	{-1.0, -2.0, -1.0},
	{0,     0,      0},
	{1.0, 2.0, 1.0}
};

static float gm3[3][3] = {
	{0.1070, 0.1131, 0.1070},
	{0.1131, 0.1196, 0.1131},
	{0.1070, 0.1131, 0.1070}
};

static float gm5[5][5] ={
	{0.0369, 0.0392, 0.0400, 0.0392, 0.0369},
	{0.0392, 0.0416, 0.0424, 0.0416, 0.0392},
	{0.0400, 0.0424, 0.0433, 0.0424, 0.0400},
	{0.0392, 0.0416, 0.0424, 0.0416, 0.0392},
	{0.0369, 0.0392, 0.0400, 0.0392, 0.0369}
};

int get_orient(float, float, int);
unsigned int get_orient1(float);
void mag_n_orients(const cv::Mat&, cv::Mat&, cv::Mat&);
cv::Mat any_to_gray(const cv::Mat& img);

static inline cv::Mat
get_rect(const cv::Mat& img, int i, int j, int size)
{
	int radius = size /2;
	int offset = size % 2;
	return img(cv::Range(i-radius,i+radius+offset), cv::Range(j-radius, j+radius+offset));
}

static inline float get_right_angle(float angle)
{
	while (angle < 0)
		angle = angle + 360.0;
	return angle - (int)angle + ((int)angle % 360);	
}

//return a square gaussian mask using col vector * row vector;
static inline
double gaussian_func(const double i, const double sigma)
{
	return (1.0 / (sigma * SQRT_2PI)) * exp(-0.5 * (i*i / (sigma*sigma)));
}

static inline
cv::Mat gaussian_mask(const int w)
{
	double tw = (w % 2 == 0)? 0.5: 0;
	double sigma = (double)(w - 1) / 2.0;
	cv::Mat kernelx = cv::Mat::zeros(1,w, CV_32F);
	double sum = 0.0;

	for (int i = 0; i < w; i++) {//it is sad you used wrong index
//		std::cout << i-w/2+0.5 << std::endl;
		double item = gaussian_func(i- w/2+tw, sigma);
		sum += item;
		kernelx.at<float>(0,i) = item;
	}
	kernelx = (1 / sum)* kernelx;
	return kernelx.t() * kernelx;
}

static inline
void psmat(const cv::Mat& smat)
{
	int type = smat.type();
	for (int i = 0; i < smat.rows; i++) {
		std::cout << "| ";
		for (int j = 0; j < smat.cols; j++) {
			switch (type) {
			case CV_8U:
				std::cout << (int)(smat.at<char>(i,j)) << " ";
				break;
			case CV_32F:
				std::cout << smat.at<float>(i,j) << " ";
				break;
			case CV_64F:
				std::cout << smat.at<double>(i,j) << " ";
				break;
			}
		}
		std::cout << "|" << std::endl;
	}
}

#endif /* UTILS_H */
