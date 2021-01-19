//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// eigen
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Dense>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> 	// std::find, std::min_element
#include <numeric> 		// std::accumulate
#include <cmath> 		// std::abs
#include <cstdlib>   	// srand, rand

/////////////////////////////////////////// Soluzione prof
void mySAD_Disparity(const cv::Mat & left_image, const cv::Mat & right_image, int radius, cv::Mat & out)
{
	int ww = radius;
	int wh = radius;

	out = cv::Mat(left_image.rows, left_image.cols, CV_8UC1, cv::Scalar(0));

	for(int i = wh; i <  left_image.rows - wh; ++i)
	{
		for(int j = ww; j <  left_image.cols - ww; ++j)
		{
			float disp=0;
			int min_sum = std::numeric_limits<int>::max();     //valore iniziale grande per la ricerca del minimo
			int min_sum_2nd = std::numeric_limits<int>::max();
			for(int d = 1;d < 128; ++d)
			{
				if(j-d >= ww) //verifico di non uscire dall'immagine destra
				{
					int sum = 0;
					for(int k = -wh; k<wh+1; ++k) //righe della finestra
						for(int v = -ww; v<ww+1; ++v) //colonne della finestra
							sum += std::abs( int(*(left_image.data + (i+k)*left_image.cols + j + v)) - int(*(right_image.data + (i+k)*left_image.cols + j + v - d)));

					if(sum < min_sum) //ho trovato un nuovo minimo?
					{
						min_sum_2nd = min_sum;   //imposto il secondo minimo uguale al minimo attuale

						min_sum = sum;           //aggiorno il minimo attuale
						disp = d;                //aggiorno la disparita' attuale
					}
					else
						if(sum < min_sum_2nd)        //questo mi serve se il minimo e' subito il primo elemento
							min_sum_2nd = sum;
				}
			}

			//ES1
			//
			out.at<unsigned char>(i,j) = disp;

			/*
			//ES2
			//
			//verifichiamo se ho trovato una disparita' poco significativa
			if(min_sum == std::numeric_limits<int>::max()          || //non ho trovato nessun minimo
					min_sum_2nd == std::numeric_limits<int>::max() || //non ho trovato nessun secondo minimo
					float(min_sum)/float(min_sum_2nd) > 0.8)         //il minimo e' troppo vicino al secondo minimo
				out.at<float>(i,j) = 0;                               //allora metto la disparita' a 0 (o un altro valore convenzionale)
			*/
		}
	}
}
///////////////////////////////////////////


void SAD_Disparity(const cv::Mat& L, const cv::Mat& R, unsigned short w_size, cv::Mat& out) {
	int max_range = 128;
	
	out = cv::Mat::zeros(L.rows, L.cols, CV_8UC1);

	int current_SAD; // SAD corrente calcolata
	int best_SAD;    // la migliore SAD calcolata fin' ora per (row_L, col_L)
	int best_offset; // la migliore disparità calcolata fin' ora per (row_L, col_L)

	for(int row_L = 0; row_L < L.rows - w_size; ++row_L) { 		// tengo conto della window
		for(int col_L = 0; col_L < L.cols - w_size; ++col_L) {  // tengo conto della window e dell'offset massimo
			best_SAD = std::numeric_limits<int>::max();
			best_offset = 0;

			for(int offset = 0; offset < max_range; ++offset) {
				if(col_L - offset - w_size >= 0) { // verifico di non uscire dall'immagine destra
					current_SAD = 0;

					for(int row_window = 0; row_window < w_size; ++row_window) {
						for(int col_window = 0; col_window < w_size; ++col_window) {
								current_SAD += std::abs(
									L.at<uint8_t>(row_L + row_window, col_L + col_window) -
									// faccio la ricerca sulla stessa riga!
									R.at<uint8_t>(row_L + row_window, col_L - offset + col_window)
								);
						}
					}

					if(current_SAD < best_SAD) {
						best_SAD = current_SAD; // aggiorno l'attuale valore migliore di disparità
						best_offset = offset;
					}
				}
			}

			// Terminati i confronti a tutti gli offset possibili, ho in best_offset il valore da assegnare a out(row_L, col_L)
			out.at<uint8_t>(row_L, col_L) = best_offset;
		}
	}
}

struct ArgumentList {
	std::string image_L;
	std::string image_R;
	unsigned short w_size;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 6;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <image_L> <image_R> -w <w_size>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
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

int main(int argc, char **argv) {
	int frame_number = 0;
	bool exit_loop = false;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop) {
		cv::Mat image_L = cv::imread(args.image_L, CV_8UC1);
		if(image_L.empty()) {
			std::cout << "Error loading image_L: " << argv[2] << std::endl;
    		return 1;
  		
		}

		cv::Mat image_R = cv::imread(args.image_R, CV_8UC1);
		if(image_R.empty()) {
			std::cout << "Error loading image_R: " << argv[3] << std::endl;
    		return 1;
  		
		}
		
		cv::Mat disparity1, disparity2;

		unsigned short w_size = args.w_size;

		mySAD_Disparity(image_L, image_R, w_size/2, disparity1);
		SAD_Disparity(image_L, image_R, w_size, disparity2);

		//////////////////////
		//processing code here

		/////////////////////

		//display images
		cv::namedWindow("image_L", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_L", image_L);

		cv::namedWindow("image_R", cv::WINDOW_AUTOSIZE);
		cv::imshow("image_R", image_R);

		cv::namedWindow("SAD1", cv::WINDOW_AUTOSIZE);
		cv::imshow("SAD1", disparity1);

		cv::namedWindow("SAD2", cv::WINDOW_AUTOSIZE);
		cv::imshow("SAD2", disparity2);

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
