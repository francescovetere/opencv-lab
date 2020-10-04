/** ex-7bis
 * Caricate l’imagine di Lena e fate un Crop generico:
 * - Generalizziamo il caso precedente, con ampiezza e posizione del cropping configurabili
 * - Definiamo una regione di cropping con 4 valori:
		riga e colonna dell'estremo in alto a sinistra (top left)
		larghezza
		Altezza
	Si tratta sempre di estrarre una sottoparte dell'immagine e metterla in un altra della dimensione giusta corrispondente
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

		//////////////////////
		//processing code here
		
		// Rettangolo di crop, delimitato da questa quadrupla
		unsigned int top_left_row = 50;
		unsigned int top_left_col = 50;
		unsigned int width = 200;
		unsigned int height = 200;

		cv::Mat output_img(top_left_row + height, top_left_col + width, CV_8UC3, cv::Scalar(0, 0, 0));
		std::cout << output_img.rows << " " << output_img.cols << std::endl;

		/* Accesso riga/colonna per immagine a multi-canale di 1 byte ciascuno 
		   (metodo generale)
		*/
		for(int v = 0; v < output_img.rows; ++v)
		{
			for(int u = 0;u < output_img.cols; ++u)
			{
				for(int k = 0;k < output_img.channels(); ++k)
				{	
					output_img.data[(v*output_img.cols + u)*output_img.channels() + k] 
					= input_img.data[((top_left_row + v*input_img.cols) + top_left_col + u)*input_img.channels() + k];
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
