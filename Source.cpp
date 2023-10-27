
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/saturate.hpp>
#include "filter.h"
#include "Filter2.h"
#include "csv_util.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "training.h"

using namespace cv;
using namespace std;

/*
This is the begning of the code
simply edit the img location given below to run
can work on videos too uncomment the comments.
*/

int main(int argc, char* argv[]) {
	// cv::VideoCapture *capdev;

	// // // open the video device
	// capdev = new cv::VideoCapture(0);
	// if( !capdev->isOpened() ) {
	//         printf("Unable to open video device\n");
	//         return(-1);
	// }

	// // get some properties of the image
	// cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
	//                 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
	// printf("Expected size: %d %d\n", refS.width, refS.height);
	Mat img;
	while (true) {
		// *capdev >> img; // get a new frame from the camera, treat as a stream

		vector<double> object_features(0);

		// if( img.empty() ) {
		//     printf("Image is empty\n");
		//     // break;
		// }   

//Reading the images from directory
		img = cv::imread("training\\band1.jpg");
		string keys = "key";
		cv::resize(img, img, cv::Size(img.cols * 0.5, img.rows * 0.4), 0, 0,cv::INTER_LINEAR);
		imshow("Orignal", img);
		//processes the input images each from the video stream / image directory and return the clean output

		cv::Mat finalImg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
		cv::Mat finalImg2(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
		cv::Mat visualRegions(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));	

		//Threshold the input image to obtain the thresholded output

		cv::Mat thresholdImg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
		cv:Mat l1(img.rows, img.cols, CV_8UC1);
		img.copyTo(l1);
		// this is task1
		draw(img, thresholdImg);
		//cv::namedWindow("Before cleaning", cv::WINDOW_FULLSCREEN);
		imshow("Before cleaning ", thresholdImg);

		/*Shrink the input image to get the eroded output( it erodes away the boundaries
		of foreground object (Always try to keep foreground in white))*/
		//This is task 2 clean the image
		cv::Mat eroded(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
		// for eroding image
		for (int i = 1; i < thresholdImg.rows - 1; i++) {
			for (int j = 1; j < thresholdImg.cols - 1; j++) {
				if ((short)thresholdImg.at<uchar>(i - 1, j) > 250 || (short)thresholdImg.at<uchar>(i, j - 1) > 250 || (short)thresholdImg.at<uchar>(i, j + 1)
		> 250 || (short)thresholdImg.at<uchar>(i + 1, j) > 250) {
					eroded.at<uchar>(i, j) = 255;
				}
				else {
					eroded.at<uchar>(i, j) = 0;
				}
			}
		}
		// for dilation
		for (int i = 1; i < eroded.rows - 1; i++) {
			for (int j = 1; j < eroded.cols - 1; j++) {
				if ((short)eroded.at<uchar>(i - 1, j) < 5 || (short)eroded.at<uchar>(i, j - 1) < 5 || (short)eroded.at<uchar>(i, j + 1)
					< 5 || (short)eroded.at<uchar>(i + 1, j) < 5) {
					finalImg.at<uchar>(i, j) = 0;
				}
				else {
					finalImg.at<uchar>(i, j) = 255;
				}
			}
		}


		morphologyEx(finalImg, finalImg, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(60, 60)));
		//cv::namedWindow("after cleaning", cv::WINDOW_FULLSCREEN);
		imshow("after cleaning ", finalImg);

		//This is task 3 and 4
		/*computes the connected components labeled image */
		Mat labels, stats, centroids;
		int number_labels = connectedComponentsWithStats(finalImg, labels, stats, centroids, 8);
		/*Finds contours in a binary image.*/
		RNG rng(12345);
		vector<vector<Point>> contours;
		findContours(finalImg, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<RotatedRect> minm_rectangle(contours.size());

		int instance = 0;
		int flag = 0, flag2 = 0;
		int a = 0, b = 0;

		for (size_t i = 0; i < contours.size(); i++)
		{
			/*The function calculates and returns
			the minimum-area bounding rectangle for a specified point set.*/
			minm_rectangle[i] = minAreaRect(contours[i]);
			cv::Point2f vertices[4];
			minm_rectangle[i].points(vertices);

			if (contourArea(contours[i]) > 3000) {
				int count = 0;
				for (int i = 0; i < 4; i++) {
					cv::Point2f points = vertices[i];
					if (points.x > 10 && points.y > 10 && points.x < 1270 && points.y < 710) {
						count += 1;
					}
					else {
						continue;
					}
				}

				if (count == 4) {
					instance = 1;
					a = i;

					if (contourArea(contours[i]) > 50000 && contourArea(contours[i]) < 700000) {
						flag2 = 1;
						b = i;
					}
				}
			}
		}
		double percentage_area = 0;
		double aspect_ratio = 0;

		Mat drawing = Mat::zeros(finalImg.size(), CV_8UC3);
		for (size_t i = 0; i < contours.size(); i++) {
			Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

			drawContours(drawing, contours, (int)i, color);
		}

		if (instance == 1) {

			if (flag2 == 1) {
				a = b;
			}
			Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

			Point2f rectangle_points[4];
			//get 4 cornersa
			minm_rectangle[a].points(rectangle_points);
			for (int j = 0; j < 4; j++)
			{
				//draw the rectangle
				line(img, rectangle_points[j], rectangle_points[(j + 1) % 4], color);
			}
		}
		//cv::namedWindow("Image_Contours", cv::WINDOW_FULLSCREEN);
		imshow("Image_Contours", drawing);

		//calculate for % area filled                           
		percentage_area = (contourArea(contours[a]) / (minm_rectangle[a].size.width * minm_rectangle[a].size.height));

		// calculate the aspect_ratio (heightt/width)

		if (minm_rectangle[a].size.width > minm_rectangle[a].size.height)
		{
			swap(minm_rectangle[a].size.width, minm_rectangle[a].size.height);
			aspect_ratio = (minm_rectangle[a].size.width / minm_rectangle[a].size.height);

		}
		else {
			aspect_ratio = (minm_rectangle[a].size.height / minm_rectangle[a].size.width);
		}


		//To make the images translation,rotation,scale and mirror invariants calculate the Hu moments

		vector<Moments> mom_ents(contours.size());

		mom_ents[a] = moments(contours[a]);//The function computes moments, up to the 3rd order 

		float c = mom_ents[a].m02 / (mom_ents[a].m00 * mom_ents[a].m00);
		float d = mom_ents[a].m20 / (mom_ents[a].m00 * mom_ents[a].m00);

		if (c > d) {
			d = mom_ents[a].m20 / (mom_ents[a].m00 * mom_ents[a].m00);
			c = mom_ents[a].m02 / (mom_ents[a].m00 * mom_ents[a].m00);
		}

		double alpha;

		vector<Point2f> centr_mass(contours.size());

		centr_mass[a] = Point2f(mom_ents[a].m10 / mom_ents[a].m00, mom_ents[a].m01 / mom_ents[a].m00);

		alpha = 0.5 * atan2(2 * mom_ents[a].mu11, mom_ents[a].mu20 - mom_ents[a].mu02);// to calculate the central axis using central moments.

		Point2f p1 = Point2f(float(200 * cos(alpha) + centr_mass[a].x), float(200 * sin(alpha) + centr_mass[a].y));
		Point2f p2 = Point2f(float(centr_mass[a].x - 200 * cos(alpha)), float(centr_mass[a].y - 200 * sin(alpha)));

		line(img, p1, p2, Scalar(0, 0, 255));
		//cv::namedWindow("image_with_cenrl_xis", cv::WINDOW_FULLSCREEN);
		imshow("image_with_cenrl_xis", img);

		/*log transform for Hu moments */
		double hu_moments[7];
		HuMoments(mom_ents[a], hu_moments);

		for (int i = 0; i < 7; i++) {

			hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]));
			object_features.push_back(hu_moments[i]);
		}

		object_features.push_back(aspect_ratio);
		object_features.push_back(percentage_area);
		object_features.push_back(c);
		object_features.push_back(d);


		//classify new images
		// This is task 6
		char file[] = "C:\\Users\\athar\\source\\repos\\Project1\\Project1\\feature_training.csv";
		vector<vector<float>> featureV;
		vector<vector<string>> labelV;

		vector<char*> filenames;
		read_image_data_csv(file, filenames, featureV, 0);

		//Calculations of mean, std_deviation and scaled euclidean distance.

		vector<float> mean;
		vector<float> std_deviation;
		map<float, string> difference;
		//initilizing sum,sum sqq diff and diff
		float* sum = new float[featureV.at(0).size()]{ 0 };
		float* sum_sq_diff = new float[featureV.at(0).size()]{ 0 };
		float* diff = new float[featureV.size()]{ 0 };

		for (int j = 0; j < featureV.at(0).size(); j++) {
			for (int i = 0; i < featureV.size(); i++) {
				sum[j] += featureV.at(i).at(j);
			}
		}
		//Calculating Mean
		for (int i = 0; i < featureV.at(0).size(); i++) {
			mean.push_back(sum[i] / featureV.size());
		}
		//Calculaing Sum sq diff
		for (int j = 0; j < featureV.at(0).size(); j++) {
			for (int i = 0; i < featureV.size(); i++) {
				sum_sq_diff[j] += (featureV.at(i).at(j) - mean[j]) * (featureV.at(i).at(j) - mean[j]);
			}
		}
		//Calculating STD_Deviation
		for (int i = 0; i < featureV.at(0).size(); i++) {
			std_deviation.push_back(sqrt(sum_sq_diff[i] / featureV.size()));
		}
		// calculating diff        
		for (int j = 0; j < featureV.size(); j++) {
			for (int i = 0; i < featureV.at(0).size(); i++) {
				diff[j] += abs(object_features.at(i) - featureV.at(j).at(i)) / std_deviation.at(i);
			}
			difference.insert(pair<float, string>(diff[j], filenames[j]));

		}
		//Calculating scaled euclidean dist
		float distance[1];
		string  f[1];


		auto it = difference.begin();
		for (int i = 0; i < 1 && it != difference.end(); ++i) {
			++it;
			if (it != difference.end()) {
				distance[i] = it->first;
				f[i] = it->second;
			}
			else {
				std::cout << " Distance not found";
			}
		}
		delete[] sum;

		// classify new images using knn

		// This is task 7
		vector<vector<vector<float>>> kdiff;

		int maxidx = 1;
		vector<float> idx;
		idx.push_back(maxidx);

		for (int i = 1; i < filenames.size(); i++) {
			if (strcmp(filenames.at(i), filenames.at(i - 1)) == 0) {
				idx.push_back(maxidx);
			}
			else {
				maxidx++;
				idx.push_back(maxidx);
			}
		}
		// Making the number of list for the labels
		for (int i = 1; i <= maxidx; i++) {
			kdiff.push_back({ {{0}} });
			//Pushing the instances for each label into corresponding label index.
			for (int j = 0; j < featureV.size() - 1; j++) {
				if (idx.at(j) == i) {
					kdiff.at(i - 1).push_back({ {idx.at(j)} });
				}
			}
		}
		int* rows = new int[maxidx];
		rows[0] = 0;
		int counter = 0;
		// This loop counts the number of rows and store it in an array in order to use it as a reference for the feature vector.
		for (int ix = 0; ix < maxidx; ix++) {
			for (int jx = 1; jx < kdiff.at(ix).size(); jx++) {
				counter++;
			}
			rows[ix + 1] = counter;
		}
		// looping over all the indexes and the store the distance metrics of a particular label in a map.
		map<float, string> knn;
		char key = cv::waitKey(10);
		for (int ix = 0; ix < maxidx; ix++) {
			map<float, string> kdmap;

			// looping over the  instances of each label.
			for (int jx = 1; jx < kdiff.at(ix).size(); jx++) {
				float* d = new float[kdiff.at(ix).size()];
				d[jx - 1] = 0;

				// Calculation of the knn distance metric
				for (int k = 0; k < featureV.at(0).size(); k++) {
					d[jx - 1] += abs(object_features.at(k) - featureV.at(rows[ix] + jx).at(k)) / std_deviation.at(k);
					cout << d[jx - 1] << endl;
				}

				// storing all the distances and the file name in map.
				kdmap.insert(pair<float, string>(d[jx - 1], filenames[rows[ix] + jx - 1]));
			}
			float kdistance[1];
			string  kf[1];
			auto kit = kdmap.begin();
			for (int i = 0; i < 1 && kit != kdmap.end(); ++i) {
				if (kit != kdmap.end()) {
					kdistance[i] = kit->first;
					kf[i] = kit->second;
					++kit;
				}
				else {
					std::cout << "not found";
				}
			}
			float* kdist = new float[maxidx];
			string* kstring = new string[maxidx];

			// summing the two lowest distances in a particular label and the store them in an array
			kdist[ix] = kdistance[0] + kdistance[1];
			kstring[ix] = kf[0];

			knn.insert(pair<float, string>(kdist[ix], kstring[ix]));
		}
		for (auto itr = knn.begin(); itr != knn.end(); itr++) {
			cout << itr->first << ": " << itr->second << endl;
		}
		float knndist[1];
		string  knnfile[1];

		auto knnit = knn.begin();
		for (int i = 0; i < 1 && knnit != knn.end(); ++i) {
			if (knnit != knn.end()) {
				knndist[i] = knnit->first;
				knnfile[i] = knnit->second;
				++knnit;
			}
			else {
				std::cout << "knn distance not found";
			}
		}
		auto itr = knn.begin();
		//cv::namedWindow("knn", cv::WINDOW_FULLSCREEN);
		putText(img, itr->second, centr_mass[a], FONT_HERSHEY_COMPLEX, img.cols / 500, Scalar({ 250,200,0 }), img.cols / 300);
		imshow("knn", img);

		if (key == 'q') {
			break;
		}
	}
	return 1;
}
