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


/*** FUNZIONI DI UTILITA'***/
void zero_padding(const cv::Mat& input, int padding_size, cv::Mat& output) {
	for(int v = 0; v < output.rows; ++v) {
		for(int u = padding_size; u < output.cols; ++u) {
			for(int k = 0; k < output.channels(); ++k) {	
				if(v < padding_size || v >= output.rows - padding_size || u < padding_size || u >= output.cols - padding_size)
					output.data[(v*output.cols + u)*output.channels() + k] = 0;
				else
					output.data[(v*output.cols + u)*output.channels() + k]
					= input.data[(v*input.cols + u)*input.channels() + k];
			}
		}
	}
}

/**
 * ES 1 - Max Pooling
 * Per risolvere il problema dei bordi utilizzo lo zero padding
 */
void maxPooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
	// Calcolo le dimensioni del padding 
	// mi basta dividere la imensione del kernel per 2, e prenderne la parte intera inferiore
	int padding_size = floor(size / 2);
	// std::cout << "padding_size: " << padding_size << std::endl;

	// Aggiungo il padding all'immagine, e salvo il risultato in una nuova immagine temporanea
	cv::Mat padded_image(image.rows + padding_size, image.cols + padding_size, image.type());
	zero_padding(image, padding_size, padded_image);

	out = padded_image.clone();

	// Calcolo le dimensioni di output
	
}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	std::cout << "Executing " << argv[0] << std::endl;

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

		// leggo l'immagine di input in un cv::Mat
		cv::Mat input_img = cv::imread(frame_name);
		if(input_img.empty()) {
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}


		//////////////////////
		//processing code here

		/*** ES 1 ***/
		// scelgo le dimensioni di size e stride per effettuare il max pooling
		int size_max_pooling = 6;
		int stride_max_pooling = 1;

		// dichiaro la matrice contenente il risultato del max pooling
		// (il suo dimensionamento è gestito direttamente nella funzione maxPooling)
		cv::Mat output_max_pooling;
		maxPooling(input_img, size_max_pooling, stride_max_pooling, output_max_pooling);


		/////////////////////

		// display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);
		
		// display output_max_pooling
		cv::namedWindow("output_max_pooling", cv::WINDOW_NORMAL);
		cv::imshow("output_max_pooling", output_max_pooling);

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
