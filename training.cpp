
/*
created by :- Atharva Pandkar
File name :- training.cpp
*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/saturate.hpp>
#include "demo.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "csv_util.cpp"


using namespace cv;
using namespace std;

Mat img;
int maines() {
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

    char filename[255] = "feature_training.csv";
    
    char image_filename[255] = "mouse";
    //this is task 5
    while (true) {
        // *capdev >> img; // get a new frame from the camera, treat as a stream

        vector<float> object_features(0);

        // if( img.empty() ) {
        //     printf("Image is empty\n");
        //     // break;
        // }   


        //Reading the images from direc  asdtory
        img = cv::imread("training\\mouse1.jpg");
        cv::resize(img, img, cv::Size(img.cols * 0.5, img.rows * 0.3), 0, 0);


        cv::Mat finalImg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        cv::Mat finalImg2(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        cv::Mat visualRegions(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        //Threshold the input image to obtain the thresholded output
        cv::Mat thresholdImg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        draw(img, thresholdImg);
        imshow("Before cleaning ", thresholdImg);

        /*Shrink the input image to get the eroded output*/
        cv::Mat eroded(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        erode(thresholdImg, eroded);

        dilate(eroded, finalImg);

        morphologyEx(finalImg, finalImg, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(60, 60)));
        imshow("after cleaning ", finalImg);


        cv::Mat labels, stats, centroids;
        int number_labels = connectedComponentsWithStats(finalImg, labels, stats, centroids, 8);

        /*Finds contours in a binary image.*/
        cv::RNG rng(12345);
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

            minm_rectangle[a].points(rectangle_points);  
            for (int j = 0; j < 4; j++)
            {
                line(img, rectangle_points[j], rectangle_points[(j + 1) % 4], color);
            }
        }

        imshow("Image_Contours", drawing);

        //calculations for % area filled

        percentage_area = (contourArea(contours[a]) / (minm_rectangle[a].size.width * minm_rectangle[a].size.height));

        // calculating the aspect_ratio (ht/wdth)

        if (minm_rectangle[a].size.width > minm_rectangle[a].size.height)
        {
            swap(minm_rectangle[a].size.width, minm_rectangle[a].size.height);
            aspect_ratio = (minm_rectangle[a].size.width / minm_rectangle[a].size.height);
        }
        else {
            aspect_ratio = (minm_rectangle[a].size.height / minm_rectangle[a].size.width);
        }

        imshow("Video", img);

        //To make the images translation,rotation,scale and mirror invariants calculate the Hu moments
        vector<Moments> mom_ents(contours.size());

        mom_ents[a] = moments(contours[a]); 

        float c = mom_ents[a].m02 / (mom_ents[a].m00 * mom_ents[a].m00);
        float d = mom_ents[a].m20 / (mom_ents[a].m00 * mom_ents[a].m00);

        if (c > d) {
            d = mom_ents[a].m20 / (mom_ents[a].m00 * mom_ents[a].m00);
            c = mom_ents[a].m02 / (mom_ents[a].m00 * mom_ents[a].m00);
        }

        double alpha;

        vector<Point2f> centr_mass(contours.size());

        centr_mass[a] = Point2f(mom_ents[a].m10 / mom_ents[a].m00, mom_ents[a].m01 / mom_ents[a].m00);

        alpha = 0.5 * atan2(2 * mom_ents[a].mu11, mom_ents[a].mu20 - mom_ents[a].mu02);//to calculate the central axis using central moments

        Point2f p1 = Point2f(float(200 * cos(alpha) + centr_mass[a].x), float(200 * sin(alpha) + centr_mass[a].y));
        Point2f p2 = Point2f(float(centr_mass[a].x - 200 * cos(alpha)), float(centr_mass[a].y - 200 * sin(alpha)));

        line(img, p1, p2, Scalar(0, 0, 255));
        imshow("image_with_cenrl_xis", img);
        /*log transform */
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
        

        for (int z = 0; z < object_features.size(); z++) {
            cout << object_features[z] << endl;
        }

        append_image_data_csv(filename, image_filename, object_features, 0);
        break;
    }


    return 0;
}