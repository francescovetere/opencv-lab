
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
		//generating file name
		//
		//multi frame case
		if(args.img_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.img_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.img_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat input_img = cv::imread(frame_name); /// open in RGB mode
		if(input_img.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		cv::Mat output_img(input_img.rows, input_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));

		//////////////////////
		//processing code here

		// Creo e inizializzo un vettore di 4 sottomatrici
		const int PARTS = 4;
		cv::Mat output_subimg[PARTS];
		for(int i = 0; i < PARTS; ++i) output_subimg[i].create(input_img.rows / 2, input_img.cols / 2, CV_8UC3);

		// L'angolo top_left(individuato da 2 valori) variera' ad ogni iterazione del ciclo for,
		// quindi creo un array di struct di questi valori
		struct top_left
		{
			int row;
			int col;
		};

		top_left top_lefts[PARTS] = {{0, input_img.cols/2}, {0, 0}, {input_img.rows/2, 0}, {input_img.rows/2, input_img.cols/2}};

		// for(int i = 0; i < PARTS; ++i) 
		// {
		// 	std::cout << top_lefts[i].row << " " << top_lefts[i].col << std::endl;
		// 	for(int v = 0; v < output_subimg[i].rows; ++v)
		// 	{
		// 		for(int u = 0;u < output_subimg[i].cols; ++u)
		// 		{
		// 			for(int k = 0;k < output_subimg[i].channels(); ++k)
		// 			{	
		// 				output_subimg[i].data[(v*output_subimg[i].cols + u)*output_subimg[i].channels() + k] 
		// 				= input_img.data[((top_lefts[i].row + v*input_img.cols) + top_lefts[i].col + u)*input_img.channels() + k];
		// 			}
		// 		}
		// 	}
		// }

		for(int i = 0; i < PARTS; ++i) 
		{
			std::cout << top_lefts[i].row << " " << top_lefts[i].col << std::endl;
			for(int v = top_lefts[i].row; v < output_img.rows/2 + top_lefts[i].row; ++v)
			{
				for(int u = top_lefts[i].col; u < output_img.cols/2 + top_lefts[i].col; ++u)
				{	
					for(int k = 0; k < output_img.channels(); ++k)
					{	
						output_img.data[(v*output_img.cols + u)*output_img.channels() + k] 
						= input_img.data[(v*input_img.cols + u)*input_img.channels() + k]; 
					}
				}
			}
		}

		/////////////////////

		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

		//display output_img
		// for(int i = 0; i < PARTS; ++i) 
		// {
		// 	cv::namedWindow("output_img" + std::to_string(i), cv::WINDOW_NORMAL);
		// 	cv::imshow("output_img" + std::to_string(i), output_subimg[i]);
		// }

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