#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <cstring>
#include <assert.h>
#include <stdbool.h>
#include "utils.hpp"

using namespace std;
using namespace cv;

/* just a draft, what we need to do for the  matching. First, coming up with a interface:
   no images are need anymore, just two descriptors? */
typedef vector<KeyPoint> Keypoints;
typedef Mat Descriptors;
typedef vector<DMatch> Matches;

/** 
 * @breif matching two set of keypoints.
 * 
 * Well, how it works? In a matching procedure, we can only matching one to
 * another. Firstly, when I do matching from A to B, at least I can find all
 * points can be matched from A. But we dont know anything about B. If all
 * points are unique(you cannot find matches within one set), I should find all
 * matches no mater I do matching from A to B or B to A.
 *
 */
static inline double euclidean_dist(const Mat& left, const Mat& right)
{
	return norm(left-right, NORM_L2);
}

static inline bool detect_as_good_match(const double best_match, const double second_match)
{
	if (best_match > 0.8)
		return false;
	else if (best_match / second_match > 0.8)
		return false;
	return true;
}
void match(const Descriptors& lkps, const Descriptors& rkps, Matches& matches)
{
	/* the naive algorithm that matches all points from one set to anther,
	 * since all the vectors are normalized, the similarity of two vectors
	 * can be expressed by cosine value?
	 */
	for (int i = 0; i < lkps.rows; i++) {
		int best_match = -1;//-1 means there is no match
		int second_match = -1;
		double min_dist = 100;
		double second_dist;
		double dist;
		for (int j = 0; j < rkps.rows; j++) {
			dist = euclidean_dist(lkps.row(i), rkps.row(j));
			if (dist < min_dist) {
				second_dist = min_dist;
				min_dist = dist;
				second_match = best_match;
				best_match = j;
			}
		}
//		cout << min_dist << " and " << second_dist << endl;
		if (!detect_as_good_match(min_dist, second_dist))
			continue;
		DMatch match(i,best_match, min_dist);
		matches.push_back(match);
	}

	//for debuging
	//cout << lkps.size() << " and " << rkps.size() << endl;
	//cout << matches.size() << endl;
}
void scale(const Mat& input, Mat& output, int scale);
void detect(Mat& img, vector<KeyPoint>& keypoints);
void describe(const Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors);
void
get_output_name(char *str, string& output_file)
{
	int len = strlen(str);
	int i;
	for (i=len-1; i >= 0; i--) {
		if (*(str+i) == '.')
			break;
	}
	char output[len+5];
	strncpy(output, str, i);
	strncpy(output+i, "-out", 4);
	strncpy(output+i+4, str+i, len-i);
	output[len+4] = '\0';
	output_file = output;

}

int main(int argc, char *argv[])
{
	Mat img, img1, tmp, gray;
	vector<KeyPoint> keypoints0, keypoints1;
	Mat descriptors0, descriptors1;
	if (argc != 3) {
		cout << "invalid number of argument" << endl;
		return -1;
	}
	img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cvtColor(img, tmp, CV_BGR2GRAY);         //only use grayscale image for now. 
	tmp.convertTo(gray, CV_32F, 1.0/255.0);
	scale(gray, tmp, 3);
	detect(tmp, keypoints0);
	describe(tmp, keypoints0, descriptors0);
	
	img1 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	cvtColor(img1, tmp, CV_BGR2GRAY);
	tmp.convertTo(gray, CV_32F, 1.0/255.0);
	scale(gray, tmp, 3);
	detect(tmp, keypoints1);
	describe(tmp, keypoints1, descriptors1);

	Matches matches;
	match(descriptors0,descriptors1, matches);
	//FlannBasedMatcher matcher;
	//matcher.match(descriptors0, descriptors1, matches);
	drawMatches(img,keypoints0, img1, keypoints1,
			matches, tmp,
			Scalar::all(-1),Scalar::all(-1),
			vector<char>(),
			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("matches", tmp);
	waitKey(0);
	//for (int i = 0; i < descriptors.size(); i++)
	//	psmat(descriptors[i]);
	string output_file = "match.jpg";

	imwrite(output_file, tmp);

}
