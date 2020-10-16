/**
 * Sull’immagine binarizzata ottenuta nell'ex1, applicare:
 * 1. Dilation
 * 2. Erosion
 * 3. Closing
 * 4. Opening
 */

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

const int max_intensity = 255;

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

						/************************************** IMPORTANTISSIMO ***************************************
						 * MOLTIPLICARE PER strutural_element.rows e cols E' NECESSARIO,
						 * ALTRIMENTI IMMAGINE DI OUTPUT CONFUSA, PIXEL SOVRAPPOSTI!
						 * SERVE PER SPOSTARMI NELL'IMMAGINE DI INPUT DEL GIUSTO OFFSET
						 **********************************************************************************************/

// OR fra 1-pixel dell'immagine di input e corrispondenti pixel dell'elemento strutturale
// N.B. cornice piu' esterna immagine di input esclusa
void dilation(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	for(int v = 0; v < output.rows; ++v)
	{
		for(int u = 0; u < output.cols; ++u)
		{
			// se siamo su un 1-pixel dell'imm di input, procedo con l'effettuare l'OR sul vicinato
			if(input.data[(structural_element.rows*(v+1)*input.cols + structural_element.rows*(u+1))] == max_intensity) {
				// esamino tutti i 9 pixel dell'elemento strutturale, e ne faccio l'OR coi corrispondenti pixel sull'immagine
				for(int i = 0; i < structural_element.rows; ++i) {
					for(int j = 0; j < structural_element.cols; ++j) {
						if(input.data[(structural_element.rows*(v+i+1)*input.cols + structural_element.cols*(u+j+1))] != 0 || structural_element.data[i*structural_element.cols + j] != 0)
							output.data[(v+i)*output.cols + (u+i)] = max_intensity;
						else output.data[(v+i)*output.cols + (u+i)] = 0;
					}
				}
			}
		}
	}
}

// AND fra tutti i pixel dell'immagine di input e corrispondenti pixel dell'elemento strutturale
// N.B. cornice piu' esterna immagine di input esclusa
void erosion(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	bool ok;

	for(int v = 0; v < output.rows; ++v)
	{	
		for(int u = 0; u < output.cols; ++u)
		{
			ok = true;
			
			// esamino tutti i 9 pixel dell'elemento strutturale, e ne faccio l'AND coi corrispondenti pixel sull'immagine
			for(int i = 0; i < structural_element.rows; ++i) {
				for(int j = 0; j < structural_element.cols; ++j) {
					if(input.data[(structural_element.rows*(v+i+1)*input.cols + structural_element.cols*(u+j+1))] == 0 || structural_element.data[i*structural_element.cols + j] == 0) {
						ok = false;
					}
				}
			}
				
			if(ok) output.data[(v*output.cols + u)] = max_intensity;
			else output.data[(v*output.cols + u)] = 0;
		}
	}
}

void opening(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	cv::Mat tmp(input.rows-2, input.cols-2, CV_8UC1);
	erosion(input, structural_element, tmp);
	dilation(tmp, structural_element, output);
}

void closing(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	cv::Mat tmp(input.rows-2, input.cols-2, CV_8UC1);
	dilation(input, structural_element, tmp);
	erosion(tmp, structural_element, output);
	std::cout << "input: " << input.rows << "x" << input.cols << std::endl;
	std::cout << "dilation: " << tmp.rows << "x" << tmp.cols << std::endl;
	std::cout << "output: " << output.rows << "x" << output.cols << std::endl;
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

		cv::Mat input_img = cv::imread(frame_name);
		if(input_img.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		//////////////////////
		//processing code here
		                                                                                     
		cv::Mat dilation_img(input_img.rows-2, input_img.cols-2, CV_8UC1);
		cv::Mat erosion_img(input_img.rows-2, input_img.cols-2, CV_8UC1);
		cv::Mat opening_img(input_img.rows-4, input_img.cols-4, CV_8UC1);
		cv::Mat closing_img(input_img.rows-4, input_img.cols-4, CV_8UC1);

		cv::Mat structural_element = cv::Mat::ones(3, 3, CV_8UC1);
		
		dilation(input_img, structural_element, dilation_img);
		erosion(input_img, structural_element, erosion_img);
		// opening(input_img, structural_element, opening_img);
		// closing(input_img, structural_element, closing_img);
		
		/////////////////////

		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

		//display output images
		cv::namedWindow("dilation_img", cv::WINDOW_NORMAL);
		cv::imshow("dilation_img", dilation_img);

		cv::namedWindow("erosion_img", cv::WINDOW_NORMAL);
		cv::imshow("erosion_img", erosion_img);

		cv::namedWindow("opening_img", cv::WINDOW_NORMAL);
		cv::imshow("opening_img", opening_img);

		cv::namedWindow("closing_img", cv::WINDOW_NORMAL);
		cv::imshow("closing_img", closing_img);

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
