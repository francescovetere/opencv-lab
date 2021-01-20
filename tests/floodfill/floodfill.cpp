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
	std::string image_L;
	std::string image_R;
	unsigned short w_size;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 6;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <image_L> <image_R> -w <w_size>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i") {
			++i;
			args.image_L = std::string(argv[i]);
			++i;
			args.image_R = std::string(argv[i]);
		}

		else if(std::string(argv[i]) == "-w") {
			++i;
			args.w_size = (unsigned char) atoi(argv[i]);
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
		cv::Mat image_L = cv::imread(args.image_L, CV_8UC1);
		if(image_L.empty()) {
			std::cout << "Error loading image_L: " << argv[2] << std::endl;
    		return 1;
  		
		}

		cv::Mat image_R = cv::imread(args.image_R, CV_8UC1);
		if(image_R.empty()) {
			std::cout << "Error loading image_R: " << argv[3] << std::endl;
    		return 1;
  		
		}
		
		unsigned short w_size = args.w_size;
		std::cout << "w_size: " << w_size << std::endl;

		//////////////////////
		//processing code here

		/////////////////////

		//display images
		cv::namedWindow("image_L", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_L", image_L);

		cv::namedWindow("image_R", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_R", image_R);

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
