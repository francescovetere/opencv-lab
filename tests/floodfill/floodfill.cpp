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

struct ArgumentList {
	std::string input_img;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 3;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <input_img>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i") {
			++i;
			args.input_img = std::string(argv[i]);
		}

		++i;
	}

	return true;
}

int main(int argc, char **argv) {
	int frame_number = 0;
	bool exit_loop = false;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop) {
		cv::Mat input_img = cv::imread(args.input_img, CV_8UC1);
		if(input_img.empty()) {
			std::cout << "Error loading input_img: " << argv[2] << std::endl;
    		return 1;
  		
		}
		
		//////////////////////
		//processing code here
		cv::Mat canny;
		cv::Canny(input_img, canny, 80, 720);

		cv::Mat label_image;
    	cv::cvtColor(canny, label_image, cv::COLOR_GRAY2RGB);

		//params for cv::floodFill
    	cv::Rect rect;
		int connectivity = 1;
    	int flags = connectivity == 0 ? 4 : 8;
		cv::Scalar l_diff(0, 0, 0);
		cv::Scalar u_diff(0, 0, 0);


    	for (int r = 0; r < label_image.rows; ++r) {
			for (int c = 0; c < label_image.cols; ++c) {
				uint8_t* bb = (uint8_t*)(label_image.data + (r * label_image.cols + c) * label_image.elemSize() + 0);
				uint8_t* gg = (uint8_t*)(label_image.data + (r * label_image.cols + c) * label_image.elemSize() + 1);
				uint8_t* rr = (uint8_t*)(label_image.data + (r * label_image.cols + c) * label_image.elemSize() + 2);
				if (*bb == 255 && *gg == 255 && *rr == 255) {
					uint8_t bc = (uint8_t) cv::theRNG() & 255;
					uint8_t gc = (uint8_t) cv::theRNG() & 255;
					uint8_t rc = (uint8_t) cv::theRNG() & 255;

					cv::floodFill(label_image, cv::Point(c, r), cv::Scalar(bc, gc, rc), &rect, l_diff, u_diff, flags);
				}
			}
		}
		/////////////////////

		//display images
		cv::namedWindow("input_img", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img", input_img);

		cv::namedWindow("canny", cv::WINDOW_AUTOSIZE);
		cv::imshow("canny", canny);

		cv::namedWindow("label_image", cv::WINDOW_AUTOSIZE);
		cv::imshow("label_image", label_image);

		//wait for key or timeout
		unsigned char key = cv::waitKey(0);
		std::cout<<"key "<<int(key)<<std::endl;

		//here you can implement some looping logic using key value:
		// - pause
		// - stop
		// - step back
		// - step forward
		// - loop on the same frame

		if(key == 'q')
			exit_loop = true;

		++frame_number;
	}

	return 0;
}
