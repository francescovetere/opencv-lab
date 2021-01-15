//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
// #include <algorithm> 	// std::find, std::min_element
#include <numeric> 		// std::accumulate
// #include <cmath> 		// std::abs
// #include <cstdlib>   	// srand, rand

/**
 * Contrast stretching
 * 
 * Riporta i valori della matrice input (CV_32FC1) nell'intervallo [0; MAX_RANGE], 
 * e mette il risultato in una matrice di output di tipo type, il quale puo' essere:
 * - CV_32FC1 => la matrice in output resta di tipo invariato, e dunque ne vengono semplicemente "schiacciati" i valori nell'intervallo richiesto (utile prima di una sogliatura)
 * - CV_8UC1 => la matrice in output, oltre a subire uno stretching dei valori, subisce anche una conversione di tipo (utile prima di una imshow)
 * 
 */
void contrast_stretching(const cv::Mat& input, cv::Mat& output, int output_type, float MAX_RANGE = 255.0f) {
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
	
	output.create(input.rows, input.cols, output_type);

	for(int r = 0; r < input.rows; ++r) {
		for(int c = 0; c < input.cols; ++c) {
			for(int k = 0; k < input.channels(); ++k) {
				float pixel_input;
				
				// distinguo il modo in cui accedo alla matrice di input in base al suo tipo
                int input_type = input.type();
				if(input_type == CV_8UC1)
					pixel_input = input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()];
				else if(input_type == CV_32FC1)
					// nel caso di matrice float, devo castare correttamente il puntatore
					// per farlo, prendo l'indirizzo di memoria e lo casto in modo opportuno, dopodichè lo dereferenzio
					pixel_input = *((float*) &(input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()]));

				float stretched_pixel_input = a*pixel_input + b;
				
				// distinguo il modo in cui accedo alla matrice di output in base al tipo
				if(output_type == CV_8UC1)
					output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()] = (uchar) stretched_pixel_input;
				
				else if(output_type == CV_32FC1)
					// nel caso di matrice float, devo castare correttamente il puntatore
					// per farlo, prendo l'indirizzo di memoria e lo casto in modo opportuno, dopodichè lo dereferenzio
					*((float*)(&output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()])) = stretched_pixel_input;
			}
		}
	}
}

/**
 * Zero padding
 * 
 * Aggiunge alla matrice input una cornice di zeri, di padding_rows righe e padding_rows colonne,
 * ed inserisce il risultato in una matrice output
 */
void zero_padding(const cv::Mat& input, int padding_rows, int padding_cols, cv::Mat& output) {
	output = cv::Mat::zeros(input.rows + 2*padding_rows, input.cols + 2*padding_cols, input.type());

	for(int v = padding_rows; v < output.rows-padding_rows; ++v) {
		for(int u = padding_cols; u < output.cols-padding_cols; ++u) {
			for(int k = 0; k < output.channels(); ++k) {	
				output.data[((v*output.cols + u)*output.channels() + k)*output.elemSize1()]
				= input.data[(((v-padding_rows)*input.cols + (u-padding_cols))*input.channels() + k)*input.elemSize1()];
			}
		}
	}
}

/**
 * Convoluzione float
 */
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	// inizialmente calcolo le dimensioni senza 
	int padding_rows = 0, padding_cols = 0;

	// Calcolo le dimensioni di output dopo aver applicato il kernel (formula tratta dalle slide di teoria)
	int out_tmp_rows = floor((image.rows + 2*padding_rows - kernel.rows)/stride + 1);
	int out_tmp_cols = floor((image.cols + 2*padding_cols - kernel.cols)/stride + 1);

	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_tmp_rows, out_tmp_cols, CV_32FC1);

	// padding per riottenere le dimensioni di input
	padding_rows = floor((image.rows - out_tmp_rows)/2);
	padding_cols = floor((image.cols - out_tmp_cols)/2);
	zero_padding(out.clone(), padding_rows, padding_cols, out);
	
	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, pesata coi corrispondenti valori del kernel
	// grazie a questo, calcolerò poi il risultato della convoluzione come somma di questi valori
	std::vector<float> convolution_window;

	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {
			// Effettuo i calcoli solamente se, tenuto conto di size e stride, non fuoriesco dall'immagine
			if((r+kernel.rows)*stride <= image.rows && (c+kernel.cols)*stride <= image.cols) {
				for(int k = 0; k < out.channels(); ++k) {

					// 2 cicli per analizzare l'attuale kernel
					for(int r_kernel = 0; r_kernel < kernel.rows; ++r_kernel) {
						for(int c_kernel = 0; c_kernel < kernel.cols; ++c_kernel) {
							// estraggo il pixel corrente sull'immagine
							float image_pixel = image.data[((stride*(r+r_kernel)*image.cols + stride*(c+c_kernel))*image.channels() + k)*image.elemSize1()];
								
							// estraggo il pixel corrente sul kernel (ricondandomi di castare correttamente il puntatore restituito)
							float kernel_pixel = *((float*) &kernel.data[((r_kernel*kernel.cols + c_kernel)*kernel.channels() + k)*kernel.elemSize1()]);
								
							// calcolo il valore corrente della convoluzione, e lo inserisco nel vector
							float current_pixel = image_pixel*kernel_pixel;	

							convolution_window.push_back(current_pixel);
						}
					}

					// sommo i valori del vector, con la funzione accumulate
					float sum_val = std::accumulate(convolution_window.begin(), convolution_window.end(), 0.0f);

					// svuoto il vector per l'iterazione successiva
					convolution_window.clear();

					// accedo al pixel di output partendo dal pixel corrente nell'input, e sommando il padding necessario
					*((float*) &out.data[(((r+padding_rows)*out.cols + (c+padding_cols))*out.channels() + k)*out.elemSize1()]) = sum_val;
				}
			}
		}
	}
}

/**
 * Convoluzione intera
 */
void conv(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	// Richiamo la convoluzione float, e successivamente riporto i valori in un range [0; 255] con un contrast stretching
	// convertendo già a CV_8UC1 per ottenere un'immagine pronta da passare a imshow 
	cv::Mat convfloat_out;
	convFloat(image, kernel, convfloat_out, stride);

	contrast_stretching(convfloat_out, out, CV_8UC1);
}

/**
 * Derivata x con Sobel 3x3
 */
void sobel_x(const cv::Mat& image, cv::Mat& derivative_x) {
	// applico una convoluzione di image con Sobel orizzontale

	// creo il filtro di Sobel
	int sobel_size = 3;

	float sobel_x_data[] { 1,  2,  1,
						   0,  0,  0,
						  -1, -2, -1 };

	cv::Mat kernel_sobel_x(sobel_size, sobel_size, CV_32FC1, sobel_x_data);
  
	// applico le convoluzioni
	conv(image, kernel_sobel_x, derivative_x);
}

/**
 * Derivata y con Sobel 3x3
 */
void sobel_y(const cv::Mat& image, cv::Mat& derivative_y) {
	// applico una convoluzione di image con Sobel verticale

	// creo il filtro di Sobel
	int sobel_size = 3;

	float sobel_y_data[] { 1,  0,  -1,
						   2,  0,  -2,
						   1,  0,  -1};

	cv::Mat kernel_sobel_y(sobel_size, sobel_size, CV_32FC1, sobel_y_data);
  
	// applico le convoluzioni
	conv(image, kernel_sobel_y, derivative_y);
}

/**
 * Derivata x con gradiente 1x3
 */
void grad_x(const cv::Mat& image, cv::Mat& derivative_x) {
	// applico una convoluzione di image con gradiente orizzontale

	// creo il filtro gradiente
	float grad_x_data[] {-1, 0, 1};

	cv::Mat kernel_grad_x(1, 3, CV_32FC1, grad_x_data);
  
	// applico le convoluzioni
	convFloat(image, kernel_grad_x, derivative_x);
}

/**
 * Derivata y con gradiente 3x1
 */
void grad_y(const cv::Mat& image, cv::Mat& derivative_y) {
	// applico una convoluzione di image con gradiente verticale

	// creo il filtro gradiente
	float grad_x_data[] {-1, 0, 1};

	cv::Mat kernel_grad_x(1, 3, CV_32FC1, grad_x_data);
	cv::Mat kernel_grad_y;
	cv::transpose(kernel_grad_x, kernel_grad_y);

	// applico le convoluzioni
	convFloat(image, kernel_grad_y, derivative_y);
}

/**
 * Data l'immagine di input, l'edge da trovare e i due graidenti sull'immagine,
 * calcola l'immagine di output, completamente nera eccetto per il lato edge
 */
void find_edge(const cv::Mat& image, int edge, const cv::Mat& Gx, const cv::Mat& Gy, cv::Mat& output) {
	output.create(image.rows, image.cols, image.type());
	output.setTo(255);

	for(int r = 0; r < output.rows; ++r) {
		for(int c = 0; c < output.cols; ++c) {
			int val_Gx = Gx.at<float>(r, c);
			int val_Gy = Gy.at<float>(r, c);
			int val_output = 255;

			if(edge == 0 && val_Gx == 0 && val_Gy < 0) val_output = 0;
			else if(edge == 1 && val_Gx > 0 && val_Gy < 0) val_output = 0;
			else if(edge == 2 && val_Gx > 0 && val_Gy == 0) val_output = 0;
			else if(edge == 3 && val_Gx > 0 && val_Gy > 0) val_output = 0;
			else if(edge == 4 && val_Gx == 0 && val_Gy > 0) val_output = 0;
			else if(edge == 5 && val_Gx < 0 && val_Gy > 0) val_output = 0;
			else if(edge == 6 && val_Gx < 0 && val_Gy == 0) val_output = 0;
			else if(edge == 7 && val_Gx < 0 && val_Gy < 0) val_output = 0;

			output.at<uint8_t>(r, c) = val_output;
		}
	}
}

struct ArgumentList {
	std::string image;
	int matricola;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 5;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <image> -m <matricola>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i") {
			++i;
			args.image = std::string(argv[i]);
		}

		else if(std::string(argv[i]) == "-m") {
			++i;
			args.matricola = atoi(argv[i]);
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
		cv::Mat input_img = cv::imread(args.image, CV_8UC1);
		if(input_img.empty()) {
			std::cout << "Error loading image: " << argv[2] << std::endl;
    		return 1;
  		
		}
		
		//////////////////////
		//processing code here

		int matricola = args.matricola;
		std::cout << "matricola: " << matricola << std::endl;

		// Recupero l'ultima cifra della matricola, modulo 8
		int last_digit = (matricola % 10)%8;
		std::cout << "last_digit(mod 8): " << last_digit << std::endl;

		// Calcolo Gx e Gy
		cv::Mat Gx, Gy;
		grad_x(input_img, Gx);
		grad_y(input_img, Gy);

		// Calcolo l'immagine di output, completamente nera eccetto per il lato last_digit
		cv::Mat output_img;
		find_edge(input_img, last_digit, Gx, Gy, output_img);

		/////////////////////

		//display images
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

		cv::namedWindow("Gx", cv::WINDOW_NORMAL);
		cv::imshow("Gx", Gx);

		cv::namedWindow("Gy", cv::WINDOW_NORMAL);
		cv::imshow("Gy", Gy);

		cv::namedWindow("output_img", cv::WINDOW_NORMAL);
		cv::imshow("output_img", output_img);

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
