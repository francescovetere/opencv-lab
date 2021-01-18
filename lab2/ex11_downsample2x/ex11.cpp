/**
 * Demosaicatura DOWNSAMPLE 2X :
 * - l'immagine di uscita sara' di dimensioni w/2 x h/2, a toni di grigio (CV_8U)
 * - ogni 4 pixel GBRG dell'immagine originale estraete un tono di grigio pari alla media dei canali G 
 *   da inserire nell'immagine di output
 * - I pattern non si sovrappongono, vi spostate quindi di 4 in 4 nell'immagine originale
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
	std::string input_img_name;		    //!< input_img file name
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
			args.input_img_name = std::string(argv[++i]);
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
		if(args.input_img_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.input_img_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.input_img_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat input_img = cv::imread(frame_name, CV_8U); /// open in CV_8U mode
		if(input_img.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		//////////////////////
		//processing code here
		
		// Output image
		cv::Mat output_img(input_img.rows/2, input_img.cols/2, CV_8U, cv::Scalar(0, 0, 0));
	
		/** Accesso riga/colonna per immagine a 1 canale di 1 byte
		 * Pattern GBRG
		 * G B G B G B
		 * R G R G R G
		 **/ 

		int G_offset = 1; // quanto sono distanziati fra loro due valori G (sia per riga che per colonna)
		int stride = 2;   // di quanto avanzo nell'immagine di input

		int G1, G2;		  // i due valori di G nel pattern analizzato attualmente
		int G_avg;		  // media aritmetica di G1 e G2
		
		for(int v = 0; v < output_img.rows; ++v)
		{
			for(int u = 0;u < output_img.cols; ++u)
			{	
				/**** questo ciclo for sui channels e' inutile, siccome input_img.channels() = 1 (immagine CV_8U) ****/
				for(int k = 0;k < input_img.channels(); ++k) { 

					G1 = input_img.data[stride * ((v*input_img.cols + u)*input_img.channels() + k)];
					G2 = input_img.data[stride * (((v + G_offset)*input_img.cols + u + G_offset)*input_img.channels() + k)];
					G_avg = (G1 + G2) / 2;

					output_img.data[(v*output_img.cols + u)*output_img.channels() + k] = G_avg;
				}
			}
		}

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
