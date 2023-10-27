#include <iostream>
#include <opencv2/opencv.hpp>
#include "Filter2.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "csv_util.h"
#include <math.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include<vector>
#include <dirent.h>
using namespace std;
using namespace cv;



int draw(cv::Mat &src, cv::Mat &dest) {
	cv::Mat grayImg(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
	//src.copyTo(grayImg);
	cv::cvtColor(src, grayImg, cv::COLOR_BGR2GRAY);
	for (int i = 0; i < grayImg.rows; i++) {
		for (int j = 0; j < grayImg.cols; j++) {
			if ((short)grayImg.at<uchar>(i, j) < 100) {
				dest.at<uchar>(i, j) = 255;
			}
		}
	}
	return 0;
}

void tval(cv::Mat src) {
	int Histogram[255] = { 0 };
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* ptr = src.ptr<cv::Vec3b>(i);
		for (int j = 0; j < src.cols; j++) {
			Histogram[(short)src.at<uchar>(i, j)] += 1;
		}
	}
	for (int i = 0; i < 255; i++) {
		printf("bin no %d \t Value %d \n", i,Histogram[i] );
	}
	//delete Histogram;
}
int erode(cv::Mat& src, cv::Mat& dst) {

    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            if ((short)src.at<uchar>(i - 1, j) > 250 || (short)src.at<uchar>(i, j - 1) > 250 || (short)src.at<uchar>(i, j + 1)
    > 250 || (short)src.at<uchar>(i + 1, j) > 250) {
                dst.at<uchar>(i, j) = 255;
            }
            else {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }

    return 0;
}


int dilate(cv::Mat& src, cv::Mat& dst) {
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            if ((short)src.at<uchar>(i - 1, j) < 5 || (short)src.at<uchar>(i, j - 1) < 5 || (short)src.at<uchar>(i, j + 1)
                < 5 || (short)src.at<uchar>(i + 1, j) < 5) {
                dst.at<uchar>(i, j) = 0;
            }
            else {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }

    return 0;
}
