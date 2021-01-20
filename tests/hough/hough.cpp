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

/**
 * Riceve in ingresso un'immagine binaria e 
 * retituisce un vector contenente le coordinate dei pixel neri
 */
void extractPoints(const cv::Mat& input, std::vector<cv::Point2f>& points) {
	for(int r = 0; r < input.rows; ++r) {
		for(int c = 0; c < input.cols; ++c) {
			if(input.at<uint8_t>(r, c) == 0) points.push_back(cv::Point2f(c, r));
		}
	}
}

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
		std::vector<cv::Point2f> points;
		extractPoints(input_img, points);

		// for(int i = 0; i < points.size(); ++i)
		// 	std::cout << points[i] << std::endl;

		// Creo il piano dei parametri tho-theta
		int rho_accumulators = 200;
		int theta_accumulators = 200;
		cv::Mat accumulators(rho_accumulators, theta_accumulators, CV_8UC1, cv::Scalar(0));

		// xcos0 + ysin0 = p
		float step = 2*M_PI / rho_accumulators;
		for(int i = 0; i < points.size(); ++i) {
			int x = points[i].x;
			int y = points[i].y;
			for(int count = 0; count < theta_accumulators; ++count) {
				float theta = count*step;
				float rho = x*std::cos(theta) + y*std::sin(theta);
				if(rho > 0 && rho < accumulators.rows && theta > 0 && theta < accumulators.cols) {
					++accumulators.at<uint8_t>(rho, count);
				}
			}
		}

		// Controllo quali accumulatori sono aumentati di valore
		int threshold = 10;

		for(int r = 0; r < accumulators.rows; ++r) {
			for(int c = 0; c < accumulators.cols; ++c) {
				if(accumulators.at<uint8_t>(r, c) > threshold) {
					std::cout << "rho: " << r << " theta: " << c*step << std::endl;
				}
			}
		}

		/////////////////////

		//display images
		cv::namedWindow("input_img", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img", input_img);

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
