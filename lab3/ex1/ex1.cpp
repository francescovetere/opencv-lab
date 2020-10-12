/**
 * Caricare l’immagine organs.pgm e applicare una binarizzazione con soglia automatica:
 * 1. Provare tutte le soglie da 1 a 255
 * 2. Scegliere quella che minimizzi la somma pesata delle varianza all’interno di ogni gruppo
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

/**
 * Funzione che riceve in input un immagine a toni di grigio e ritorna il puntatore all'array che ne rappresenta l'istogramma
 */
int* compute_histogram(const cv::Mat& img) {
	static int histogram[255];
	
	for(int v = 0; v < img.rows; ++v)
	{	
		for(int u = 0; u < img.cols; ++u)
		{
			++histogram[img.data[(v*img.cols + u)]];
		}
	}

	return histogram;
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

		//////////////////////
		//processing code here
		                                                                                     
		cv::Mat output_img(input_img.rows, input_img.cols, CV_8U, cv::Scalar(0, 0, 0));

		/* Accesso riga/colonna per immagine a multi-canale di 1 byte ciascuno
		*/
		int threshold = 137;

		for(int v = 0; v < output_img.rows; ++v)
		{	
			for(int u = 0; u < output_img.cols; ++u)
			{
				// std::cout << (int)input_img.data[(v*input_img.cols + u)] << std::endl;
				if((int)input_img.data[(v*input_img.cols + u)] >= threshold)
					output_img.data[(v*output_img.cols + u)] = 255;

				else output_img.data[(v*output_img.cols + u)] = 0;

			}
		}

		int* histogram = compute_histogram(input_img);
		for(int i = 0; i < 255; ++i) std::cout << "histogram[" << i << "] = " << histogram[i] << std::endl; 
		
		// int sum = 0;
		// for(int i = 0; i < 255; ++i) sum += histogram[i];
		// assert(input_img.rows*input_img.cols == sum);

		
		/////////////////////

		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

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
