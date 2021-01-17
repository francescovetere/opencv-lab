//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// eigen
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Dense>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> 	// std::find, std::min_element
#include <numeric> 		// std::accumulate
#include <cmath> 		// std::abs
#include <cstdlib>   	// srand, rand

using namespace std;
using namespace cv;

/* ## Display an image in a new window with a given name ## */
void display(const Mat& image, const string name)
{
    namedWindow(name);
    imshow(name, image);
}


/* ## Generate an intensity histogram ## */
void histogram(const Mat& image, int h[], const int levels)
{

    for (int i = 0; i < levels; i++)
        h[i] = 0;

    for (size_t i = 0; i < image.rows * image.cols * image.elemSize(); i += image.elemSize())
    {
        h[image.data[i]]++;
    }
}


void CDF(const int h[], int eq_h[], int N)
{

    float cdf[256];

    cdf[0] = (float)h[0] / N;

    for (int i = 1; i < 256; i++)
    {
        cdf[i] = cdf[i - 1] + (float)h[i] / N;
    }

    for (int i = 0; i < 256; i++)
    {
        eq_h[i] = 255 * cdf[i];
        cout << h[i] << " --> " << cdf[i] << " --> " << eq_h[i] << endl;
    }
}

int main(int argc, char **argv)
{

    Mat image = imread("../images/Lenna.png");
    Mat equalized = Mat(image.size(), image.type(), Scalar(0));

    /*
        int h[256], eq_h[256];
        histogram(image, h, 256);
        CDF(h, eq_h, image.rows * image.cols);
        
        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                equalized.at<uchar>(i, j) = eq_h[image.at<uchar>(i, j)];
            }
        }
    */

    int r[256], g[256], b[256];
    int eq_r[256], eq_g[256], eq_b[256];
    Mat red = Mat(image.size(), CV_8UC1, Scalar(0));
    Mat green = Mat(image.size(), CV_8UC1, Scalar(0));
    Mat blue = Mat(image.size(), CV_8UC1, Scalar(0));
    int N = image.cols * image.rows;

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            red.at<uchar>(i, j) = image.at<Vec3b>(i, j)[0];
            green.at<uchar>(i, j) = image.at<Vec3b>(i, j)[1];
            blue.at<uchar>(i, j) = image.at<Vec3b>(i, j)[2];
        }
    }

    histogram(red, r, 256);
    CDF(r, eq_r, N);

    histogram(green, g, 256);
    CDF(g, eq_g, N);

    histogram(blue, b, 256);
    CDF(b, eq_b, N);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            equalized.at<Vec3b>(i, j)[0] = eq_r[image.at<Vec3b>(i, j)[0]];
            equalized.at<Vec3b>(i, j)[1] = eq_g[image.at<Vec3b>(i, j)[1]];
            equalized.at<Vec3b>(i, j)[2] = eq_b[image.at<Vec3b>(i, j)[2]];
        }
    }

    cout << equalized << endl;

    display(image, "dark");
    display(equalized, "equalized");

    waitKey(0);
    destroyAllWindows();
    return 0;
}
