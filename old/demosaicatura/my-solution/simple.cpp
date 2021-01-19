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
				"   -i arg                   input_img name. Use %0xd format for multiple images."<<std::endl<<
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

/**
 * Contrast stretching
 * 
 * Riporta i valori della matrice input (CV_32FC1) nell'intervallo [0; MAX_RANGE], 
 * e mette il risultato in una matrice di output di tipo type, il quale puo' essere:
 * - CV_32FC1 => la matrice in output resta di tipo invariato, e dunque ne vengono semplicemente "schiacciati" i valori nell'intervallo richiesto (utile prima di una sogliatura)
 * - CV_8UC1 => la matrice in output, oltre a subire uno stretching dei valori, subisce anche una conversione di tipo (utile prima di una imshow)
 * 
 */
void contrast_stretching(const cv::Mat& input, cv::Mat& output, int type, float MAX_RANGE = 255.0f) {
	double min_pixel, max_pixel;

	// funzione di OpenCV per la ricerca di minimo e massimo
	cv::minMaxLoc(input, &min_pixel, &max_pixel);
	
	// DEBUG
	// std::cout << "min: " << min_pixel << ", max: " << max_pixel << std::endl;

	// In generale, contrast_and_gain(r, c) = a*f(r, c) + b
	// contrast_stretching ne è un caso particolare in cui:
	// a = 255 / max(f) - min(f)
	// b = (255 * min(f)) / max(f) - min(f)
	float a = (float) (MAX_RANGE / (max_pixel - min_pixel));
	float b = (float) (-1 * ((MAX_RANGE * min_pixel) / (max_pixel - min_pixel)));
	
	output.create(input.rows, input.cols, type);

	for(int r = 0; r < input.rows; ++r) {
		for(int c = 0; c < input.cols; ++c) {
			for(int k = 0; k < input.channels(); ++k) {
				float pixel_input;
				
				// distinguo il modo in cui accedo alla matrice di input in base al tipo
				if(type == CV_8UC1)
					pixel_input = input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()];
				else if(type == CV_32FC1)
					// nel caso di matrice float, devo castare correttamente il puntatore
					// per farlo, prendo l'indirizzo di memoria e lo casto in modo opportuno, dopodichè lo dereferenzio
					pixel_input = *((float*) &(input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()]));

				float stretched_pixel_input = a*pixel_input + b;
				
				// distinguo il modo in cui accedo alla matrice di output in base al tipo
				if(type == CV_8UC1)
					output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()] = (uchar) stretched_pixel_input;
				
				else if(type == CV_32FC1)
					// nel caso di matrice float, devo castare correttamente il puntatore
					// per farlo, prendo l'indirizzo di memoria e lo casto in modo opportuno, dopodichè lo dereferenzio
					*((float*)(&output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()])) = stretched_pixel_input;
			}
		}
	}
}



/**
 * Funzione che interpola i valori del vicinato di un pixel dell'immagine di input
 */
float interpolation_neighbors(const cv::Mat& input, int pixel_r, int pixel_c, int window_size) {
	float sum = 0;
	int num_pixels = 0;
	for(int r = -(window_size/2); r <= window_size/2; ++r) {
		for(int c = -(window_size/2); c <= window_size/2; ++c) {
			// controllo di rimanere dentro l'immagine
			if(pixel_r+r >= 0 && pixel_r+r <= input.rows-1 && pixel_c+c >= 0 && pixel_c+c <= input.cols-1 
			 && !(r == 0 && c == 0) 								// controllo di non considerare il pixel corrente
			//  && input.at<uint8_t>(pixel_r + r, pixel_c + c) != 0	// controllo di considerare un pixel valido
			) {
				++num_pixels;
				sum += input.at<uint8_t>(pixel_r + r, pixel_c + c);
				// std::cout << pixel_r + r << "," << pixel_c + c << "+\n";
			}
		}
	}

	// std::cout << sum << "/" << num_pixels << "=" << sum/num_pixels << std::endl;
	return sum/num_pixels;
}

/**
 * Funzione che esegue una demosaicatura GBRG, con il metodo linear interpolation
 */
void bayer_GBRG_interpolation(const cv::Mat& input_img, cv::Mat& output_B, cv::Mat& output_G, cv::Mat& output_R) {
	output_B = cv::Mat(input_img.rows, input_img.cols, CV_8UC1, cv::Scalar(0));
	output_G = cv::Mat(input_img.rows, input_img.cols, CV_8UC1, cv::Scalar(0));	
	output_R = cv::Mat(input_img.rows, input_img.cols, CV_8UC1, cv::Scalar(0));
	
	/**
	 * Pattern RGGB
	 * R G R G R G --> riga pari
	 * G B G B G B --> riga dispari
	 * R G R G R G --> riga pari
	 * | |
	 * | --> colonna dispari
	 * ---> colonna pari
	 **/
	for(int r = 0; r < input_img.rows; ++r) {
		for(int c = 0; c < input_img.cols; ++c) {
			int val = input_img.data[(c + r*input_img.cols)];

			if(val == 0) val = 1; // in questo modo, il controllo sul vicinato 3x3 è semplice 

			// riga pari 
			if(r%2 == 0) {
				// colonna pari
				if(c%2 == 0){
					output_R.data[(c + r*input_img.cols)] = val;
				}
					
				// colonna dispari			
				else {
					output_G.data[(c + r*input_img.cols)] = val;
				}
			}

			// riga dispari
			else {
				// colonna pari
				if(c%2 == 0) {
					output_G.data[(c + r*input_img.cols)] = val;
				}

				// colonna dispari
				else {
					output_B.data[(c + r*input_img.cols)] = val;
				}
					
			}
		}
	}

	// Se valore = 0, controllo sul vicinato 3x3: gli assegno interpolazione dei vicini validi
	for(int r = 0; r < input_img.rows; ++r) {
		for(int c = 0; c < input_img.cols; ++c) {
			if(output_B.at<uint8_t>(r, c) == 0) {
				float interpolated_val = interpolation_neighbors(input_img, r, c, 3);
				output_B.at<uint8_t>(r, c) = interpolated_val;
			}

			if(output_G.at<uint8_t>(r, c) == 0) {
				float interpolated_val = interpolation_neighbors(input_img, r, c, 3);
				output_G.at<uint8_t>(r, c) = interpolated_val;
			}
			
			if(output_R.at<uint8_t>(r, c) == 0) {
				float interpolated_val = interpolation_neighbors(input_img, r, c, 3);
				output_R.at<uint8_t>(r, c) = interpolated_val;
			}
		}
	}
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

		cv::Mat input_img = cv::imread(frame_name, CV_8UC1);
		if(input_img.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);


		//////////////////////
		//processing code here

		/***********/
		/*** ES1 ***/
		/***********/

		// Demosaicatura con interpolazione lineare
		// 3 immagini di output, una per ogni canale
		cv::Mat output_B, output_G, output_R;

		bayer_GBRG_interpolation(input_img, output_B, output_G, output_R);

		//display output_B
		cv::namedWindow("output_B", cv::WINDOW_NORMAL);
		cv::imshow("output_B", output_B);

		//display output_G
		cv::namedWindow("output_G", cv::WINDOW_NORMAL);
		cv::imshow("output_G", output_G);

		//display output_R
		cv::namedWindow("output_R", cv::WINDOW_NORMAL);
		cv::imshow("output_R", output_R);

		/***********/
		/*** ES2 ***/
		/***********/

		// Creazione immagine output a colori, mischiando le 3 precedenti
		cv::Mat output_BGR(input_img.rows, input_img.cols, CV_8UC3);

		// for(int r = 0; r < output_BGR.rows; ++r) {
		// 	for(int c = 0; c < output_BGR.cols; ++c) {
		// 		for(int k = 0; k < output_BGR.channels(); ++k) {
		// 			int val;
		// 			switch(k) {
		// 				case 0: // B
		// 					val = output_B.data[r*output_B.cols + c];
		// 					break;
		// 				case 1: // G
		// 					val = output_G.data[r*output_G.cols + c];
		// 					break;
		// 				case 2: // R
		// 					val = output_R.data[r*output_R.cols + c];
		// 					break;
		// 			}

		// 			output_BGR.data[((r*output_BGR.cols + c)*output_BGR.channels() + k)*output_BGR.elemSize1()] = val;
		// 		}
		// 	}
		// }
		for(int r = 0; r < output_BGR.rows; ++r) {
			for(int c = 0; c < output_BGR.cols; ++c) {
				// b
				output_BGR.at<cv::Vec3b>(r, c)[0] = output_B.at<uint8_t>(r, c);

				// g		
				output_BGR.at<cv::Vec3b>(r, c)[1] = output_G.at<uint8_t>(r, c);

				// r
				output_BGR.at<cv::Vec3b>(r, c)[2] = output_R.at<uint8_t>(r, c);
			}
		}
		
		//display output_BGR
		cv::namedWindow("output_BGR", cv::WINDOW_NORMAL);
		cv::imshow("output_BGR", output_BGR);

		/***********/
		/*** ES3 ***/
		/***********/

		// Creazione immagine output a singolo canale
		cv::Mat gray(input_img.rows, input_img.cols, CV_8UC1);

		for(int r = 0; r < gray.rows; ++r) {
			for(int c = 0; c < gray.cols; ++c) {
				float val = (output_R.data[r*output_R.cols + c] +
							output_G.data[r*output_G.cols + c] +
							output_B.data[r*output_B.cols + c]) / 3;
				gray.data[r*gray.cols + c] = val;
			}
		}

		//display output_gray
		cv::namedWindow("output_gray", cv::WINDOW_NORMAL);
		cv::imshow("output_gray", gray);


		/***********/
		/*** ES4 ***/
		/***********/
		// Creazione istogramma dell'immagine gray

		int MAX_RANGE = 255;
		
		int histogram[MAX_RANGE];
		for(int i = 0; i < MAX_RANGE; ++i) histogram[i] = 0;

		for(int r = 0; r < gray.rows; ++r) {
			for(int c = 0; c < gray.cols; ++c) {
				++histogram[gray.data[r*gray.cols + c]];
			}
		}

		for(int i = 0; i < MAX_RANGE; ++i) std::cout << "histogram[" << i << "] = " << histogram[i] << std::endl;

		/***********/
		/*** ES5 ***/
		/***********/

		// Creazione immagine singolo canale ottenuta da grayscale tramite contrast stretching

		cv::Mat stretched_image;
		contrast_stretching(gray, stretched_image, CV_8UC1);

		//display stretched_image
		cv::namedWindow("stretched_image", cv::WINDOW_NORMAL);
		cv::imshow("stretched_image", stretched_image);
		
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
