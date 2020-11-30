//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList {
	std::string image_L;
	std::string image_R;
	unsigned short w_size;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	// if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	// {
	// 	std::cout<<"usage: ./ex1 -i <image_L> <image_R> -w <w_size>"<<std::endl;
	// 	return false;
	// }

	int i = 1;
	while(i<argc)
	{
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

// Faccio alpiù 128 confronti (scritto sulle slide)
const int max_range = 128;

void SAD_Disparity(const cv::Mat& L, const cv::Mat& R, unsigned short w_size, cv::Mat& out) {
	out = cv::Mat::zeros(L.rows, L.cols, CV_8UC1);

	for(int row_L = 0; row_L < L.rows; ++row_L) {
		for(int col_L = 0; col_L < L.cols; ++col_L) {
			for(int offset = 0; offset < max_range; ++offset) {
				for(int row_window = 0; row_window < w_size; ++row_window) {
					for(int col_window = 0; col_window < w_size; ++col_window) {
						// TODO
					}
				}
			}
		}
	}
}

int main(int argc, char **argv) {
	int frame_number = 0;
	bool exit_loop = false;

	std::cout<<"Starting ex1"<<std::endl;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop) {
		cv::Mat image_L = cv::imread(args.image_L);
		cv::Mat image_R = cv::imread(args.image_R);

		cv::Mat disparity;
		
		//////////////////////
		//processing code here

		std::cout << "w_size: " << args.w_size << std::endl;
		SAD_Disparity(image_L, image_R, 7, disparity);

		/////////////////////

		//display images
		cv::namedWindow("image_L", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_L", image_L);

		cv::namedWindow("image_R", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_R", image_R);

		cv::namedWindow("disparity", cv::WINDOW_AUTOSIZE);
		cv::imshow("disparity", disparity);

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
