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

/** 
 * Funzione per il calcolo dell'istogramma 
 */
int* compute_histogram(const cv::Mat& img, int max_levels) {
	int* histogram = new int[max_levels];

	// inzializzo vettore a 0
	for(int i = 0; i < max_levels; ++i)
		histogram[i] = 0;

	// ogni pixel contribuisce col suo valore ad aumentare di 1 la corrispondente colonna dell'istogramma
	for(int v = 0; v < img.rows; ++v)
		for(int u = 0; u < img.cols; ++u)
			++histogram[img.data[(v*img.cols + u)]];

	// ritorno il puntatore all'area di memoria allocata con la new
	return histogram;
}

/** 
 * Funzione per il calcolo degli istogrammi r, g e b di un'immagine a colori
 * Restituisce un array contenente i 3 istogrammi
 */
int** compute_histogram_rgb(const cv::Mat& img, int max_levels) {
	int* histogram_b = new int[max_levels];
	int* histogram_g = new int[max_levels];
	int* histogram_r = new int[max_levels];

	// inzializzo vettori a 0
	for(int i = 0; i < max_levels; ++i) {
		histogram_b[i] = 0; histogram_g[i] = 0; histogram_r[i] = 0;
	}

	// ogni pixel contribuisce col suo valore ad aumentare di 1 la corrispondente colonna dell'istogramma
	for(int v = 0; v < img.rows; ++v) {
		for(int u = 0; u < img.cols; ++u) { 
			// for(int k = 0; k < img.channels(); ++k) {
				++histogram_b[img.data[(v*img.cols + u)*img.channels() + 0]]; // b
				++histogram_g[img.data[(v*img.cols + u)*img.channels() + 1]]; // g
				++histogram_r[img.data[(v*img.cols + u)*img.channels() + 2]]; // r
			// }
		}
	}

	int** histograms = new int*[3];
	for(int i = 0; i < 3; ++i) {
		histograms[i] = new int[max_levels];
	}

	for(int i = 0; i < max_levels; ++i) histograms[0][i] = histogram_b[i];
	for(int i = 0; i < max_levels; ++i) histograms[1][i] = histogram_g[i];
	for(int i = 0; i < max_levels; ++i) histograms[2][i] = histogram_r[i];

	// ritorno il puntatore all'area di memoria allocata con la new
	return histograms;
}

/**
 * Funzione per il calcolo dell'istogramma equalizzato
 */
int* equalize_histogram(int* histogram, int max_levels, int N_pixels) {
	// equalized_h[j] = CDF[j]*max_levels
	// CDF[j] = CI[j] / N_pixels = CDF[j-1] + histogram[j]/N_pixels
	// CI[j] = sum{i=0->j}(histogram[i]) = CI[j-1] + histogram[j]
	int CI[max_levels];
	CI[0] = histogram[0];
	for(int j = 1; j < max_levels; ++j) {
		CI[j] = CI[j-1] + histogram[j];
		// std::cout << "CI[" << j << "]: " << CI[j] << std::endl;
	}	
	
	double CDF[max_levels];
	CDF[0] = histogram[0]/N_pixels;
	for(int j = 1; j < max_levels; ++j) {
		CDF[j] = CDF[j-1] + (double)histogram[j]/N_pixels;
		// std::cout << "CDF[" << j << "]: " << CDF[j] << std::endl;
	}

	int* equalized_h = new int[max_levels];
	for(int j = 0; j < max_levels; ++j) {
		equalized_h[j] = CDF[j] * max_levels;
	}

	return equalized_h; 
}

/**
 * Funzione che calcola l'istogramma di un'immagine, lo equalizza e lo riapplica all'immagine
 */
void equalize_image_gray(const cv::Mat& input_img, cv::Mat& output_img) {
	int max_levels = 256;
	int* h = compute_histogram(input_img, max_levels);
	// for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << h[i] << std::endl;

	int* equalized_h = equalize_histogram(h, max_levels, input_img.rows*input_img.cols);
	// for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << equalized_h[i] << std::endl;

	output_img.create(input_img.rows, input_img.cols, input_img.type());
	for(int r = 0; r < output_img.rows; ++r)
		for(int c = 0; c < output_img.cols; ++c)
			output_img.at<uint8_t>(r, c) = equalized_h[input_img.at<uint8_t>(r, c)];
}

/**
 * Funzione che calcola l'istogramma di un'immagine, lo equalizza e lo riapplica all'immagine
 */
void equalize_image_rgb(const cv::Mat& input_img, cv::Mat& output_img) {
	int max_levels = 256;
	int** histograms = compute_histogram_rgb(input_img, max_levels);

	// Equalizzo i 3 istogrammi b, g, r
	int* equalized_hb = equalize_histogram(histograms[0], max_levels, input_img.rows*input_img.cols);
	int* equalized_hg = equalize_histogram(histograms[1], max_levels, input_img.rows*input_img.cols);
	int* equalized_hr = equalize_histogram(histograms[2], max_levels, input_img.rows*input_img.cols);

	// for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << equalized_hb[i] << std::endl;
	// for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << equalized_hg[i] << std::endl;
	// for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << equalized_hr[i] << std::endl;

	output_img.create(input_img.rows, input_img.cols, CV_8UC3);
	for(int r = 0; r < output_img.rows; ++r)
		for(int c = 0; c < output_img.cols; ++c) {
			// b
			output_img.at<cv::Vec3b>(r, c)[0] = equalized_hb[input_img.at<cv::Vec3b>(r, c)[0]];

			// g			
			output_img.at<cv::Vec3b>(r, c)[1] = equalized_hg[input_img.at<cv::Vec3b>(r, c)[1]];

			// r
			output_img.at<cv::Vec3b>(r, c)[2] = equalized_hr[input_img.at<cv::Vec3b>(r, c)[2]];
		}
}

struct ArgumentList {
	std::string input_img;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 3;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <input_img>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i") {
			++i;
			args.input_img = std::string(argv[i]);
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
		cv::Mat input_img_gray = cv::imread(args.input_img, CV_8UC1);
		cv::Mat input_img_rgb = cv::imread(args.input_img); // apro immagine in BGR mode
		
		if(input_img_gray.empty()) {
			std::cout << "Error loading input_img: " << argv[2] << std::endl;
    		return 1;
		}
		
		//////////////////////
		//processing code here
		cv::Mat output_img_gray, output_img_rgb;

		equalize_image_gray(input_img_gray, output_img_gray);
		equalize_image_rgb(input_img_rgb, output_img_rgb);
		/////////////////////

		//display images
		cv::namedWindow("input_img_gray", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img_gray", input_img_gray);
	
		cv::namedWindow("input_img_rgb", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img_rgb", input_img_rgb);

		cv::namedWindow("output_img_gray", cv::WINDOW_AUTOSIZE);
		cv::imshow("output_img_gray", output_img_gray);

		cv::namedWindow("output_img_rgb", cv::WINDOW_AUTOSIZE);
		cv::imshow("output_img_rgb", output_img_rgb);
		
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
