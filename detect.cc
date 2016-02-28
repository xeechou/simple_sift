#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <assert.h>
#include "utils.hpp"

using namespace std;
using namespace cv;

static int
get_neighbors(Mat& img,
	      int i, int j,
	      float *neighbors)
{
	float maxval = img.at<float>(i,j);
	int maxind = 4, ind = 0;
	for(int u = -1; u <= 1; u++)
		for (int v = -1; v <= 1; v++) {
			float a = img.at<float>(i+u, j+v);
			neighbors[ind] = a;
			if (maxval < a) {
				maxval = a;
				maxind = ind;
			}
			ind++;
		}
	return maxind;
}

static Mat harris_operator(const Mat& Ix2, const Mat& Iy2, const Mat& Ixy)
{
	Mat harris = Mat::zeros(Ix2.rows, Ix2.cols, Ix2.type());
	for (int i = 0; i < Ix2.rows; i++)
		for (int j = 0; j < Ix2.cols; j++) {
			float det = Ix2.at<float>(i,j) * Iy2.at<float>(i,j) -
				Ixy.at<float>(i,j) * Ixy.at<float>(i,j);
			float trace = Ix2.at<float>(i,j) + Iy2.at<float>(i,j);
			float op = det - 0.04 * (trace * trace);
			//float op = det/trace;
			harris.at<float>(i,j) =  op;
			//cout << op << endl;
		}
	return harris;
}
static float calc_main_angle(const Mat& magnitude, const Mat& angles,
			    int i, int j, float sigma)
{
	int r = (int)(3 * 1.5 * sigma + 0.5);
	float hist[36] = {0}; 
	Mat mags = get_rect(magnitude, i, j, r);
	Mat angs = get_rect(angles, i, j, r);
	assert(mags.rows == mags.cols && mags.cols == r);
	float max = 0, main_angle = 0.0;
	int maxind = -1;
	{
	for (int i = 0; i < r; i++)
		for (int j = 0; j < r; j++) {
			float angle = get_right_angle(angs.at<float>(i,j));
			int bin = (int)(angle / 10);
			hist[bin] += mags.at<float>(i,j);
			if (hist[bin] > max) {
				max = hist[bin];
				maxind = bin;
				main_angle = angle;
			}
		}
	}
	//cout << i  << " and " << j << endl;
	//cout << "angle: " << main_angle << endl;
	return main_angle;
//	float theta = hist(i)
}



void
detect(Mat& img, vector<KeyPoint>& keypoints)
{
	Mat gray = any_to_gray(img);
	Mat harris = Mat::zeros(gray.rows, gray.cols, gray.type());
	
	Mat gx = Mat(3,3, CV_32F, &gradiant_x);
	Mat gy = Mat(3,3, CV_32F, &gradiant_y);
	Mat gm = gaussian_mask(5);
	//psmat(gm);
	/* the whole formula is:
	   E(x,y) = Sigma_uv(W(u,v) * (Ix^2 + Iy^2 + 2IxIy))
	   = [u,v] * (W(u,v) * |Ix^2,  IxIy|) *[u]
	                      (|IxIy,  Iy^2|)  [v]

	   one important thing is: the W(u,v) is inside, which makes the
	   formular isotropic. Maybe we don't need w(x,y) in continues version,
	   but in discreate version...
	   
	   Important: We don't know what u,v is, in other words. We don't know
	   which direction will have largest changes. But a corner means move to
	   any direction gonna make sufficient change. So we choose the smallest
	   eigenvalue. Now it makes senese right :)
	*/

	//first step, compute Ix, Iy, Ixy.
	{
		Mat tmp;
		Mat Ix2, Iy2, Ixy;
		Mat Ix2g, Iy2g, Ixyg;
		{
			Mat Ix, Iy;
			filter2D(gray, Ix, -1, gx, Point(-1,-1), 0, BORDER_DEFAULT);
			filter2D(gray, Iy, -1, gy, Point(-1,-1), 0, BORDER_DEFAULT);
			Ix2 = Ix.mul(Ix);
			Iy2 = Iy.mul(Iy);
			Ixy = Ix.mul(Iy);
		}
		//Convolute with gaussian mask, make them isotropic.
		filter2D(Ix2, Ix2g, -1, gm, Point(-1,-1), 0, BORDER_DEFAULT);
		filter2D(Iy2, Iy2g, -1, gm, Point(-1,-1), 0, BORDER_DEFAULT);
		filter2D(Ixy, Ixyg, -1, gm, Point(-1,-1), 0, BORDER_DEFAULT);

		tmp = harris_operator(Ix2g, Iy2g, Ixyg);
		normalize(tmp,harris);
	}
	//do thresholding and max-supperation
	int mini = 10, minj = 10;
	int maxi = harris.rows - 10, maxj = harris.cols - 10;
	float neighbors[9];
	Mat magnitude, angles;
	mag_n_orients(gray, magnitude, angles);
	for (int i = 1; i < harris.rows -1; i++)
		for (int j = 1; j < harris.cols -1; j++) {
			if (!(i > mini && i < maxi && j > minj && j < maxj))
				continue;
			float point = harris.at<float>(i,j);
			//threshold, bad threshold
//			cout << point << endl;
			if (point < 0.002)
				continue;
			//non-maximum suppression
			int max = get_neighbors(harris, i,j, neighbors);
			if (max != 4)
				continue;
			float main_angle = calc_main_angle(magnitude, angles,
							   i,j,1);
			//compute the main angle of the feature point
			keypoints.push_back(KeyPoint(float(j),float(i),1,main_angle));
		}
}

void scale(const Mat& input, Mat& output, int scale)
{
	Mat gray = any_to_gray(input);
	// we get original image if scale == 0
	assert(scale <= 5);
	//here is algorithm, the k scale image gets by DoG:
	// gaussian_blur(K^(scale-1)*sigma, img) - gaussian_blur(K^(scale)*sigma, img)
	double k = pow(2, 1.0/3.0);
	double sigma0 = pow(2, scale-1)*0.5;
	double sigma1 = pow(2, scale) * 0.5;
	//cout << sigma0 << " and " << sigma1 << endl;

	Mat g0, g1;
	int size0 = (int)2*sigma0+1;
	if (size0 % 2 == 0)
		size0 += 1;
	int size1 = (int)2*sigma1+1;
	if (size1 % 2 == 0)
		size1 += 1;

	GaussianBlur(gray,g0,Size(size0,size0),sigma0);
	GaussianBlur(gray,g1,Size(size1,size1),sigma1);
	output = g0-g1;
//	imshow("scale", g0-g1+0.5);
//	waitKey(0);
}

//int main(int argc, char *argv[])
//{
//	Mat img, tmp, gray;
//	vector<KeyPoint> keypoints;
//	if (argc != 2) {
//		cout << "invalid number of argument" << endl;
//		return -1;
//	}
//	img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
//	cvtColor(img, tmp, CV_BGR2GRAY);         //only use grayscale image for now. 
//	tmp.convertTo(gray, CV_32F, 1.0/255.0);
//	scale(gray, tmp, 3);
//	//imshow("scaled", tmp);
//	//waitKey(0);
//	detect(tmp, keypoints);
//	//describe(img, keypoints);
// 	drawKeypoints(img, keypoints, tmp);
//	imshow("keypoints", tmp);
//	waitKey(0);
//	
//}
