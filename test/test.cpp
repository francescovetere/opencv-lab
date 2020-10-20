//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

#include "utils.h"

struct ArgumentList {
	std::string image_name;    
	int wait_t;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
		std::cout << "exit:  type q" <<std::endl << std::endl;
		std::cout << "Allowed options:" << std::endl <<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i")
			args.image_name = std::string(argv[++i]);

		if(std::string(argv[i]) == "-t") 
			args.wait_t = atoi(argv[++i]);
		else
			args.wait_t = 0;

		++i;
	}

	return true;
}

int main(int argc, char **argv) {
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	//////////////////////
	//parse argument list
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop) {
		//generating file name

		if(args.image_name.find('%') != std::string::npos)	//multi frame case
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.image_name.c_str());

		//opening file
		std::cout << "Opening " << frame_name << std::endl;

		cv::Mat input_img = cv::imread(frame_name, CV_8UC3);
		if(input_img.empty()) {
			std::cout << "Unable to open " << frame_name << std::endl;
			return 1;
		}

		//////////////////////
		//processing code here
		
		//zero padding
		int dim_padding = 15;
		cv::Mat padded_img(input_img.rows + dim_padding, input_img.cols + dim_padding, CV_8UC3);
		zero_padding(input_img, dim_padding, padded_img);
		cv::namedWindow("padded_img", cv::WINDOW_NORMAL);
		cv::imshow("padded_img", padded_img);

		/////////////////////

		//display
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

		//wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
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
