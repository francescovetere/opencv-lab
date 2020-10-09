/**
 * Scambiate l'ordine dei canali di colore:
 * Ad esempio R->G G->B B->R
 * Dovreste ottenere un'immagine con colori sbagliati
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
	std::string image_name;		    //!< image file name
	int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
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
		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.image_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat image = cv::imread(frame_name); /// open in RGB mode
		if(image.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		cv::Mat output_img(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		//////////////////////
		//processing code here

		/* Accesso riga/colonna per immagine a multi-canale di 1 byte ciascuno 
		   (metodo generale)
		*/
		for(int v = 0; v < image.rows; ++v)
		{
			for(int u = 0;u < image.cols; ++u)
			{
				for(int k = 0;k < image.channels(); ++k) {
					output_img.data[(v*image.cols + u)*image.channels() + k] = image.data[((v*image.cols + u)*image.channels() + ((k + 1)%image.channels()))];
				}
			}

			std::cout << "\n\n";
		}

		/////////////////////

		//display image
		cv::namedWindow("image", cv::WINDOW_NORMAL);
		cv::imshow("image", image);

		//display image
		cv::namedWindow("output_image", cv::WINDOW_NORMAL);
		cv::imshow("output_image", output_img);

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
