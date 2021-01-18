/**
 * Demosaicatura LUMINANCE :
 * - l'immagine di uscita sara' di dimensioni w x h, a toni di grigio (CV_8U)
 * - ogni 4 pixel GBRG dell'immagine originale estraete un tono di grigio nel seguente modo:
 *   R*0.30 + G*0.59 + B*0.11 (G = media dei canali G del pattern preso in considerazione)
 * - I pattern si sovrappongono, quindi vi spostate pixel per pixel nell'immagine originale
 * - Attenzione: in questo modo i pattern cambiano
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

		// Output image
		cv::Mat output_img(input_img.rows,input_img.cols,CV_8U,cv::Scalar(1,2,3));

		//////////////////////
		//processing code here

		int result_b, result_g, result_r;

		const float COEFF_B = 0.3f, COEFF_G = 0.59f, COEFF_R = 0.11f;

		/** Accesso riga/colonna per immagine a 3 canali di 1 byte ciascuno
		 * Pattern GBRG
		 * G B G B G B --> riga pari
		 * R G R G R G --> riga dispari
		 * G B G B G B --> riga pari
		 * | |
		 * | --> colonna dispari
		 * ---> colonna pari
		 **/
		for(int v = 0; v < output_img.rows; ++v) {

			for(int u = 0; u < output_img.cols; ++u) {

				// riga pari 
				if(v%2 == 0){
					// colonna pari
					if(u%2 == 0){
						result_b = input_img.data[((u+1) + v*input_img.cols)];
					
						// G (top-left --> down-right)
						result_g = (input_img.data[(u + v*input_img.cols)] + input_img.data[((u+1) + (v+1)*input_img.cols) ]) / 2;
					
						result_r = input_img.data[(u + (v+1)*input_img.cols)];
						}
					
					// colonna dispari			
					else {
						result_b = input_img.data[(u + v*input_img.cols)];

						// G (top-right --> down-left)
						result_g = (input_img.data[((u+1) + v*input_img.cols)] + input_img.data[(u + (v+1)*input_img.cols)]) / 2;

						result_r = input_img.data[((u+1) + (v+1)*input_img.cols)];
					}
				}

				// riga dispari
				else {
					// colonna pari
					if(u%2 == 0){
						result_b = input_img.data[((u+1) + (v+1)*input_img.cols)];

						// G (top-right --> down-left)
						result_g = (input_img.data[((u+1) + v*input_img.cols)] + input_img.data[(u + (v+1)*input_img.cols)]) / 2;

						result_r = input_img.data[(u + v*input_img.cols)];
					}

					// colonna dispari
					else {
						result_b = input_img.data[(u + (v+1)*input_img.cols)];

						// G (top-left --> down-right)
						result_g = (input_img.data[(u + v*input_img.cols)] + input_img.data[((u+1) + (v+1)*input_img.cols)]) / 2;

						result_r = input_img.data[((u+1) + v*input_img.cols)];
					}
					
				}

				output_img.data[(u + v*output_img.cols)] = result_r*COEFF_R + result_g*COEFF_G + result_b*COEFF_B;
				
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
