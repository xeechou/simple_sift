#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <cstring>
#include <assert.h>
#include "utils.hpp"

using namespace std;
using namespace cv;



static inline void
collect_hist(const Mat& mag, const Mat& angles,
	     int m, int n, float main_angle,
	     float *window)
{
//	cout << mag.rows <<  " and " << mag.cols << endl;
//	assert(mag.rows == 16 && mag.cols == 16);
	memset(window, 0, sizeof(float) * 8);
	for (int i = m; i < m+4; i++)
		for (int j = n; j < n+4; j++)
		{
			unsigned int orient = get_orient1(angles.at<float>(i,j) - main_angle);
			window[orient] += mag.at<float>(i,j);
		}
}

static void get_descriptor(const Mat& magnitude,
			   const Mat& angles,
			   const KeyPoint& keypoint,
			   Mat& hist)
{
	assert(hist.rows == 1 && hist.cols == 128);

	int i = (int)keypoint.pt.y;
	int j = (int)keypoint.pt.x;
	float main_angle = keypoint.angle;

	//TODO: multiply mags to 16x16 gaussian mask element-wisely
	Mat mags = get_rect(magnitude, i,j, 16);
	Mat angs = get_rect(angles, i,j, 16);
	//TODO: we need to write a special gaussian function that accept even size
	Mat gmask = gaussian_mask(16);
	mags = mags.mul(gmask);
	Mat tmp = Mat::zeros(1,128,CV_32F);	
	float *data = (float*)tmp.data;
	float subwindow[8] = {0};

	for (int k = 0; k < 15; k++) {
		int i = k / 4;
		int j = k % 4;
		collect_hist(mags, angs, 4 * i, 4 * j, main_angle, subwindow);
		memcpy(data + 8*k, subwindow, sizeof(float)*8);
 	}
	normalize(tmp, hist);
}

void describe(const Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors)
{
	//actually before this step, the rotation and scale invariance should
	//already be done.
	Mat gray = any_to_gray(img);
	Mat magnitude;
	Mat angles;
	mag_n_orients(gray, magnitude, angles);
	//boundary test
	descriptors = Mat::zeros(keypoints.size(), 128, CV_32F);
	for (int i = 0; i < keypoints.size(); i++) {
		Mat hist = Mat::zeros(1, 128, CV_32F);
		get_descriptor(magnitude, angles, keypoints[i], hist);
		hist.copyTo(descriptors.row(i));
	}

}
//void detect(Mat& img, vector<KeyPoint>& keypoints);
//
//
//int main(int argc, char *argv[])
//{
//	Mat img, tmp, gray;
//	vector<KeyPoint> keypoints;
//	vector<Mat> descriptors;
//	if (argc != 2) {
//		cout << "invalid number of argument" << endl;
//		return -1;
//	}
//	img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
//	cvtColor(img, tmp, CV_BGR2GRAY);         //only use grayscale image for now. 
//	tmp.convertTo(gray, CV_32F, 1.0/255.0);
//	detect(gray, keypoints);
//
//	drawKeypoints(img, keypoints, tmp);
//	imshow("keypoints", tmp);
//	waitKey(0);
//	describe(img, keypoints, descriptors);
//	//for (int i = 0; i < descriptors.size(); i++)
//	//	psmat(descriptors[i]);
//}
