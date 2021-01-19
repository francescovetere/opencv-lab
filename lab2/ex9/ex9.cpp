
/**
 * Caricate l’immagine di Lena, suddividerla in 4 sottoparti e scambiarle casualmente
 */

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList {
	std::string img_name;		    //!< input_img file name
	int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <input_img_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   input_img name. Use %0xd format for multiple input_imgs."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.img_name = std::string(argv[++i]);
		}

		if(std::string(argv[i]) == "-t") {
			args.wait_t = atoi(argv[++i]);
		}
		else
			args.wait_t = 0;

		++i;
	}

	return true;
}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	std::cout<<"Simple program."<<std::endl;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop)
	{
		//multi frame case
		if(args.img_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.img_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.img_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat input_img = cv::imread(frame_name, CV_8UC1); /// open in RGB mode
		if(input_img.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		cv::Mat output_img(input_img.rows, input_img.cols, input_img.type());
		
		cv::Mat top_left(input_img.rows/2, input_img.cols/2, input_img.type());
		cv::Mat top_right(input_img.rows/2, input_img.cols/2, input_img.type());
		cv::Mat bottom_left(input_img.rows/2, input_img.cols/2, input_img.type());
		cv::Mat bottom_right(input_img.rows/2, input_img.cols/2, input_img.type());

		int val = input_img.at<uint8_t>(262, 448);
		std::cout << val << std::endl;
		
		// Riempimento delle 4 sottomatrici
		for(int r = 0; r < input_img.rows; ++r) {
			for(int c = 0; c < input_img.cols; ++c) {
				// std::cout << "r, c: " << r << ", " << c << std::endl;
				// std::cout << "rows, cols " << input_img.rows << ", " << input_img.cols << std::endl;

				float val = input_img.at<uint8_t>(r, c);
				// std::cout << "val " << val << std::endl;

				if(r < input_img.rows/2 && c < input_img.cols/2) {
					// std::cout << "top-left" << std::endl;
					top_left.at<uint8_t>(r, c) = val;
				}
				else if(r < input_img.rows/2 && c >= input_img.cols/2) {
					// std::cout << "top-right" << std::endl;
					top_right.at<uint8_t>(r, c - input_img.cols/2) = val;
				}
				else if(r >= input_img.rows/2 && c < input_img.cols/2) {
					// std::cout << "bottom-left" << std::endl;
					bottom_left.at<uint8_t>(r - input_img.rows/2, c) = val;
				}
				else if(r >= input_img.rows/2 && c >= input_img.cols/2) {
					// std::cout << "bottom-right" << std::endl;
					bottom_right.at<uint8_t>(r - input_img.rows/2, c - input_img.cols/2) = val;
				}
			}
		}
		
		// Riempimento matrice di output
		cv::Mat tmp1, tmp2;
		cv::hconcat(top_left, top_right, tmp1);
		cv::hconcat(bottom_left, bottom_right, tmp2);
		cv::vconcat(tmp1, tmp2, output_img);
		
		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

		cv::namedWindow("top_left", cv::WINDOW_NORMAL);
		cv::imshow("top_left", top_left);
		cv::namedWindow("top_right", cv::WINDOW_NORMAL);
		cv::imshow("top_right", top_right);
		cv::namedWindow("bottom_left", cv::WINDOW_NORMAL);
		cv::imshow("bottom_left", bottom_left);
		cv::namedWindow("bottom_right", cv::WINDOW_NORMAL);
		cv::imshow("bottom_right", bottom_right);
		
		//display output_img
		cv::namedWindow("output_img", cv::WINDOW_NORMAL);
		cv::imshow("output_img", output_img);

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

		frame_number++;
	}

	return 0;
}