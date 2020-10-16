#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

#define LEVELS 256
#define WHITE 255
#define BLACK 0

void generate_Histo(Mat image, int h[]){

    for(int i = 0; i < LEVELS; i++) h[i] = 0;

    for(size_t i = 0; i < image.rows*image.cols*image.elemSize(); i+= image.elemSize()){
        if(image.data[i] >= 50) h[image.data[i]]++;
    }

}

// Definizione di Wikipedia
int otsu(Mat image){

    int histogram[LEVELS];
    generate_Histo(image, histogram);

    double w1, w2, u1, u2;

    double variance[LEVELS-1];
    for(int i = 0; i < LEVELS-1; i++) variance[i] = 0;

    for(int th = 1; th < LEVELS; th++){

        w1 = 0;
        w2 = 0;

        for(int i = 0; i < th; i++) w1 += histogram[i];
        for(int i = th; i < LEVELS-1; i++) w2 += histogram[i];

        u1 = 0;
        u2 = 0;

        for(int i = 0; i < th; i++) u1 += i*histogram[i]/w1;
        for(int i = th; i < LEVELS-1; i++) u2 += i*histogram[i]/w2;

        variance[th-1] = w1*w2*pow(u1 - u2, 2);

    }

    double max_var = 0;
    int threshold = 1;

    for(int i = 0; i < LEVELS-1; i++){

        if(max_var < variance[i]){
            max_var = variance[i];
            threshold = i;
        }

    }

    cout << "Threshold generated: " << threshold << endl;
    return threshold;

}

Mat binarization(Mat image, int th){

    Mat binary = Mat(image.rows, image.cols, image.type(), Scalar(BLACK));

    for(size_t i = 0; i < image.rows*image.cols*image.elemSize(); i+= image.elemSize()){
        
        if(image.data[i] > th) binary.data[i] = WHITE;

    }

    return binary;

}


int main(int argc, char **argv)
{

    // Read the image file
    if (argv[1] != NULL)
        cout << "Trying to open: " << argv[1] << endl;
    else
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    Mat image = imread(argv[1], CV_8UC1);
    cout << "Loaded " << argv[1] << ": [" << image.type() << "]" << endl;
    cout << "Channels: " << image.channels() << " elemsize1: " << image.elemSize1() << endl;

    String windowName1 = "Original";
    namedWindow(windowName1);
    imshow(windowName1, image);

    String windowName2 = "Binary";
    namedWindow(windowName2);
    imshow(windowName2, binarization(image, otsu(image)));

    waitKey(0);          // Wait for any keystroke in the window
    destroyAllWindows(); //destroy the created window

    return 0;
}