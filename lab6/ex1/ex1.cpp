//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <cmath> // abs()

struct ArgumentList {
	std::string image_L;
	std::string image_R;
	unsigned short w_size;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<6 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: ./ex1 -i <image_L> <image_R> -w <w_size>"<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			++i;
			args.image_L = std::string(argv[i]);
			++i;
			args.image_R = std::string(argv[i]);
		}

		else if(std::string(argv[i]) == "-w") {
			++i;
			args.w_size = (unsigned char) atoi(argv[i]);
		}

		++i;
	}

	return true;
}

// Faccio alpiù 128 confronti (scritto sulle slide)
const int max_range = 128;

void SAD_Disparity(const cv::Mat& L, const cv::Mat& R, unsigned short w_size, cv::Mat& out) {
	out = cv::Mat::zeros(L.rows, L.cols, CV_8UC1);

	int current_disparity; // ad ogni valore di offset, conterrà la disparità corrente calcolata
	int best_disparity; // ad ogni valore di offset, conterrà la migliore disparità calcolata fin' ora per (row_L, col_L)

	for(int row_L = 0; row_L < L.rows - w_size; ++row_L) { // tengo conto della window
		for(int col_L = 0; col_L < L.cols - w_size - max_range; ++col_L) { // tengo conto della window e dell'offset massimo
			best_disparity = std::numeric_limits<int>::max();
			
			for(int offset = 0; offset < max_range; ++offset) {
				current_disparity = 0;

				for(int row_window = 0; row_window < w_size; ++row_window) {
					for(int col_window = 0; col_window < w_size; ++col_window) {

						current_disparity += std::abs(
							L.at<unsigned char>(row_L + row_window, col_L + col_window) -
							// faccio la ricerca sulla stessa riga!
							R.at<unsigned char>(row_L + row_window, col_L + col_window + offset)
						);

						if(current_disparity < best_disparity) 
							best_disparity = current_disparity; // aggiorno l'attuale valore migliore di disparità
					}
				}
			}

			// Terminati i confronti a tutti gli offset possibili, ho in best_disparity il valore da assegnare a out(row_L, col_L)
			out.at<unsigned char>(row_L, col_L) = best_disparity;
		}
	}

	double min, max;
	cv::minMaxLoc(out, &min, &max);

	255*(out-min) / (max-min);
}

int main(int argc, char **argv) {
	int frame_number = 0;
	bool exit_loop = false;

	std::cout<<"Starting ex1"<<std::endl;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop) {
		cv::Mat image_L = cv::imread(args.image_L);
		cv::Mat image_R = cv::imread(args.image_R);
		unsigned short w_size = args.w_size;

		cv::Mat disparity;
		
		//////////////////////
		//processing code here

		std::cout << "w_size: " << w_size << std::endl;
		SAD_Disparity(image_L, image_R, w_size, disparity);

		/////////////////////

		//display images
		cv::namedWindow("image_L", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_L", image_L);

		cv::namedWindow("image_R", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_R", image_R);

		cv::namedWindow("disparity (w_size: " + std::to_string(w_size) + ")", cv::WINDOW_AUTOSIZE);
		cv::imshow("disparity (w_size: " + std::to_string(w_size) + ")", disparity);

		cv::imwrite("disparity (w_size: " + std::to_string(w_size) + ").jpg", disparity);
		//wait for key or timeout
		unsigned char key = cv::waitKey(0);
		std::cout<<"key "<<int(key)<<std::endl;

		//here you can implement some looping logic using key value:
		// - pause
		// - stop
		// - step back
		// - step forward
		// - loop on the same frame

		if(key == 'q')
			exit_loop = true;

		++frame_number;
	}

	return 0;
}
