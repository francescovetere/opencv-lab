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

/**
 * Funzione che esegue una demosaicatura GBRG, con il metodo downsample 2x
 */
void bayer_GBRG_downsample(const cv::Mat& input_img, cv::Mat& output_img) {
	// Fattore moltiplicativo con cui mi muovo lungo l'input
	int stride = 2;
	
	// L'immagine di output subisce un downsample pari al valore di stride
	output_img.create(input_img.rows/stride, input_img.cols/stride, CV_8UC3);

	int result_b, result_g, result_r;

	// Ciclo sull'output, che avrà dimensioni pari alla metà dell'input in questo caso
	for(int r = 0; r < output_img.rows; ++r) {
		for(int c = 0; c < output_img.cols; ++c) {

			// Le righe e colonne di riferimento sull'input sono quindi ottenute moltiplicando
			// riga e colonna corrente dell'output per un fattore di stride = 2
			int r_ = r * stride; int c_ = c * stride;

			// Costruisco i valori risultato b, g e r

			// B = (top-right)
			result_b = input_img.data[((c_+1) + r_*input_img.cols)];
				
			// G = (top-left + bottom-right) / 2
			result_g = (input_img.data[(c_ + r_*input_img.cols)] + input_img.data[((c_+1) + (r_+1)*input_img.cols) ]) / 2;
			
			// R = (bottom-left)
			result_r = input_img.data[(c_ + (r_+1)*input_img.cols)];

			// Inserisco i valori b, g, r nell'output
			output_img.data[(c + r*output_img.cols)*output_img.channels() + 0] = result_b; 	//B
			output_img.data[(c + r*output_img.cols)*output_img.channels() + 1] = result_g;	//G
			output_img.data[(c + r*output_img.cols)*output_img.channels() + 2] = result_r; 	//R
		}
	}

}

// Struttura per rappresentate un cluster
struct cluster {
	int num_pixels = 0;
	float ur = 0;
	float ug = 0;
	float ub = 0;
};

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
		cv::Mat input_img = cv::imread(args.input_img);
		if(input_img.empty()) {
			std::cout << "Error loading input_img: " << argv[2] << std::endl;
    		return 1;
  		
		}

		//////////////////////
		//processing code here

		//ES1 - PARTE 1
		// Creo 3 immagini CV_8UC1 per ciascuno dei 3 canali
		cv::Mat blue(input_img.rows, input_img.cols, CV_8UC1);
		cv::Mat green(input_img.rows, input_img.cols, CV_8UC1);
		cv::Mat red(input_img.rows, input_img.cols, CV_8UC1);
		
		for(int r = 0; r < input_img.rows; ++r) {
			for(int c = 0; c < input_img.cols; ++c) {
				blue.at<uint8_t>(r, c) = input_img.at<cv::Vec3b>(r, c)[0];
				green.at<uint8_t>(r, c) = input_img.at<cv::Vec3b>(r, c)[1];
				red.at<uint8_t>(r, c) = input_img.at<cv::Vec3b>(r, c)[2];
			}
		}

		//ES1 - PARTE2
		//Le tre immagini ottenute al passo precedente contengono in realtà il pattern di Bayer GBRG 
		//di tre diverse immagini: left, center e right. Effettuare demosaicatura con DOWNSAMPLE_2X

		/** Accesso riga/colonna per immagine a 3 canali di 1 byte ciascuno
		 * Pattern GBRG
		 * G B G B G B --> riga pari
		 * R G R G R G --> riga dispari
		 * G B G B G B --> riga pari
		 * | |
		 * | --> colonna dispari
		 * ---> colonna pari
		 **/
		cv::Mat left;
		cv::Mat center;
		cv::Mat right;

		bayer_GBRG_downsample(blue, left);
		bayer_GBRG_downsample(green, center);
		bayer_GBRG_downsample(red, right);
		/////////////////////

		//display images
		cv::namedWindow("input_img", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img", input_img);

		// cv::namedWindow("blue", cv::WINDOW_AUTOSIZE);
		// cv::imshow("blue", blue);
		// cv::namedWindow("green", cv::WINDOW_AUTOSIZE);
		// cv::imshow("green", green);
		// cv::namedWindow("red", cv::WINDOW_AUTOSIZE);
		// cv::imshow("red", red);

		cv::namedWindow("left", cv::WINDOW_AUTOSIZE);
		cv::imshow("left", left);
		// cv::namedWindow("center", cv::WINDOW_AUTOSIZE);
		// cv::imshow("center", center);
		// cv::namedWindow("right", cv::WINDOW_AUTOSIZE);
		// cv::imshow("right", right);
		
		//ES2
		//////////////////////
		int cluster_count = 10;

		//lista cluster dell'immagine originale
		std::vector<cluster> cluster_list_original(cluster_count);
		
		//immagine dei cluster
		cv::Mat new_image(left.size(), left.type());

		// Definisco la matrice dei samples di input, da passare a cv::kmeans
		// Avrò tante righe quanti sono i pixel di input, ma solo 3 colonne (r g b)
		// quindi ogni riga -> 1 pixel
		cv::Mat samples(left.rows * left.cols, 3, CV_32FC1);
		for(int r = 0; r < left.rows; ++r)
			for(int c = 0; c < left.cols; ++c)
				for(int z = 0; z < 3; ++z)
					samples.at<float>(r + c*left.rows, z) = left.at<cv::Vec3b>(r, c)[z];

		// Dichiaro la matrice che cv::kmeans riempirà: conterrà per ogni pixel un valore [0...cluster_count]
		// Ovvero, per ogni pixel mi dice a quale cluster esso appartiene
		cv::Mat labels;
		int attempts = 5;
		cv::Mat centers;

		cv::kmeans(samples, cluster_count, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 10000, 0.0001), attempts, cv::KmeansFlags::KMEANS_PP_CENTERS, centers);
		
		std::cout << "samples: " << samples.rows << "x" << samples.cols << std::endl; // 76800 x 3
		std::cout << "labels: " << labels.rows << "x" << labels.cols << std::endl; 	  // 76800 x 1
		std::cout << "centers: " << centers.rows << "x" << centers.cols << std::endl; // 5 x 3

		std::cout << "centers\n" << centers << std::endl;

		// Riempimento di new_image coi valori dei centri dei cluster
		for(int r = 0; r < left.rows; ++r)
			for(int c = 0; c < left.cols; ++c)
			{
				// Per ogni pixel, recupero la label che gli è stata associata
				int cluster_idx = labels.at<int>(r + c*left.rows, 0); // matrice labels ha 1 sola colonna

				// Recupero il corrispondente centro, e lo assegno all'attuale pixel di new_image
				// matrice centers ha 3 colonne: ciascuna, mi dà il valore r, g o b del centro i-esimo
				new_image.at<cv::Vec3b>(r,c)[0] = centers.at<float>(cluster_idx, 0);
				new_image.at<cv::Vec3b>(r,c)[1] = centers.at<float>(cluster_idx, 1);
				new_image.at<cv::Vec3b>(r,c)[2] = centers.at<float>(cluster_idx, 2);

				// Se e' la prima volta che incontro questo cluster, aggiorno i centri di massa
				if(cluster_list_original[cluster_idx].num_pixels == 0) {
					cluster_list_original[cluster_idx].ur = centers.at<float>(cluster_idx, 0);
					cluster_list_original[cluster_idx].ug = centers.at<float>(cluster_idx, 1);
					cluster_list_original[cluster_idx].ub = centers.at<float>(cluster_idx, 2);
				}

				// In ogni caso, aggiorno il numero di pixel appartenenti al cluster cluster_idx
				++cluster_list_original[cluster_idx].num_pixels;
			}

		// std::sort(cluster_list_original.begin(), cluster_list_original.end());

		cv::namedWindow("clustered image", cv::WINDOW_NORMAL);
		cv::imshow( "clustered image", new_image );
		/////////////////////////////////////////////////////////////////


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