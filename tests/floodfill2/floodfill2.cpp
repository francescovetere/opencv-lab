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

void neighbours(const cv::Mat& image, int r, int c, int w_size, cv::Mat& neighs) {
	neighs.create(w_size, w_size, image.type());

	for(int rr = -(w_size/2); rr <= (w_size/2); ++rr) {
		for(int cc = -(w_size/2); cc <= (w_size/2); ++cc) {
			if(rr > 0 && rr < image.rows && cc > 0 && cc < image.cols) {
				int r = rr + w_size/2;
				int c = cc + w_size/2;
				neighs.at<cv::Vec3b>(r, c)[0] = image.at<cv::Vec3b>(rr, cc)[0];
				neighs.at<cv::Vec3b>(r, c)[1] = image.at<cv::Vec3b>(rr, cc)[1];
				neighs.at<cv::Vec3b>(r, c)[2] = image.at<cv::Vec3b>(rr, cc)[2];

				std::cout << "ok" << std::endl;
			}
		}
	}
}

void myFloodfill(cv::Mat& label_image, int r, int c, int bc, int rc, int gc) {
	int w_size = 3;
	if(label_image.at<cv::Vec3b>(r, c)[0] == 255 &&
		label_image.at<cv::Vec3b>(r, c)[1] == 255 &&
		label_image.at<cv::Vec3b>(r, c)[2] == 255) {	

		label_image.at<cv::Vec3b>(r, c)[0] = bc;
		label_image.at<cv::Vec3b>(r, c)[1] = gc;
		label_image.at<cv::Vec3b>(r, c)[2] = rc;
		// cv::Mat neighs;
		// neighbours(label_image, r, c, 3, neighs);

		// for(int r = 0; r < neighs.rows; ++r) {
		// 	for(int c = 0; c < neighs.cols; ++c) {
		// 		myFloodfill(label_image, r, c, bc, rc, gc);
		// 	}
		// }
		for(int rr = -(w_size/2); rr <= (w_size/2); ++rr) {
			for(int cc = -(w_size/2); cc <= (w_size/2); ++cc) {
				if(r+rr > 0 && r+rr < label_image.rows && c+cc > 0 && c+cc < label_image.cols) {
					myFloodfill(label_image, r+rr, c+cc, bc, rc, gc);
				}
			}
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
	srand(time(NULL));
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
		srand(time(NULL));
		cv::Mat input_img = cv::imread(args.input_img, CV_8UC1);
		if(input_img.empty()) {
			std::cout << "Error loading input_img: " << argv[2] << std::endl;
    		return 1;
  		
		}
		
		//////////////////////
		//processing code here
		cv::Mat canny;
		cv::Canny(input_img, canny, 80, 720);

		cv::Mat label_image = canny.clone();
		cv::cvtColor(label_image.clone(), label_image, cv::COLOR_GRAY2RGB);

		for(int r = 0; r < label_image.rows; ++r) {
			for(int c = 0; c < label_image.cols; ++c) {
				if(label_image.at<cv::Vec3b>(r, c)[0] == 255 &&
					label_image.at<cv::Vec3b>(r, c)[1] == 255 &&
					label_image.at<cv::Vec3b>(r, c)[2] == 255) {
					int bc = rand() % 255;
					int gc = rand() % 255;
					int rc = rand() % 255;
					myFloodfill(label_image, r, c, bc, gc, rc);
					// c = label_image.cols; r = label_image.rows; // per uscire
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
