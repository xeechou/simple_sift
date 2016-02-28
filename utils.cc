#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <assert.h>
#include "utils.hpp"

using namespace std;
using namespace cv;

static float main_angles1[9] = 
{
/*0*/0.0,
/*pi/4*/45.0,
/*pi/2*/90.0,
/*3pi/4*/135.0,
/*pi*/180.0,
/*5pi/4*/225.0,
/*3pi/2*/270.0,
/*7pi/4*/315.0,
/*2pi*/360.0
};

static const char *main_angle_strings[8] =
{
	"right","right lower corner","down","left lower corner",
	"left","left upper corner", "up", "right upper corner"
};

/* return a interger smaller than 8 */
unsigned int
get_orient1(float angle)
{

	float min_diff = 1000000.0;
	int orientation = -1;
	//this is important, we limit our angle between 0 to 360
	angle = get_right_angle(angle);
	
	for (int i = 0; i < 9; i++) {
		float diff = abs(angle-main_angles1[i]);
		if (diff < min_diff) {
			orientation = i;
			min_diff = diff;
		}
	}
	return orientation % 8;
}
/** 
 * @breif, calculate the orientation given the range
 * 
 * @angle: is the angle you need to calculate,
 * @offset: is the offset of your first section
 * @slices: how many interval?, must can be divided by 360
 * 
 * 
 */

int
get_orient(float angle, float offset, int slices)
{
	if (360 % slices != 0)
		return -1;
	angle = get_right_angle(angle - offset);
	return (int(angle+0.5) % (360 / slices));
}

static const char *orients_to_string(int o)
{
	assert(o < 9);
	return main_angle_strings[o];
}

void mag_n_orients(const Mat& img, Mat& mags, Mat& orients)
{
	
	mags = Mat::zeros(img.rows, img.cols, CV_32F);
	orients = Mat::zeros(img.rows, img.cols, CV_32F);//hope float wont be
							 //problem
	Mat gray = any_to_gray(img);
	Mat gx, gy;
	{
		Mat mgx = Mat(3,3, CV_32F, &gradiant_x);
		Mat mgy = Mat(3,3, CV_32F, &gradiant_y);
		filter2D(gray, gx, -1, mgx, Point(-1,-1), 0, BORDER_DEFAULT);
		filter2D(gray, gy, -1, mgy, Point(-1,-1), 0, BORDER_DEFAULT);
	}
	for (int i = 0; i < gray.rows; i++)
		for (int j = 0; j < gray.cols; j++) {
			//p stands for points
			float pgx = gx.at<float>(i,j);
			float pgy = gy.at<float>(i,j);
			mags.at<float>(i,j) = abs(pgx) + abs(pgy);
			orients.at<float>(i,j) = fastAtan2(pgy, pgx);
		}
}


Mat
any_to_gray(const Mat& img)
{
	Mat gray;
	{
		Mat cimg;
		int type = img.type();
		if (type == CV_32F || type == CV_64F)
			img.copyTo(gray);
		else if (type == CV_8UC3 || type == CV_32FC3 || type == CV_64FC3) {
			cvtColor(img, cimg, CV_BGR2GRAY);
			type = cimg.type();
			if (type == CV_8U)
				cimg.convertTo(gray, CV_32F, 1.0/255.0);
		}
	}
	return gray;
}

