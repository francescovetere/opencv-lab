//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>   	// std::vector
#include <algorithm>	// std::for_each
#include <numeric>  	// std::accumulate
#include <cmath> 		// M_PI, sqrt(), pow()

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


/****************************/
/* FUNZIONI DI UTILITA' */

/**
 * Zero padding
 * aggiunge alla matrice di input una cornice di zeri, di padding_rows righe e padding_rows colonne,
 * e mette il risultato in una matrice di output
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
 * Contrast stretching
 * Riporta i valori della matrice di input tra [0, 255], e mette il risultato in una matrice di output
 * In generale, contrast_and_gain(r, c) = a*f(r, c) + b
 * contrast_stretching ne è un caso particolare in cui:
 * a = 255 / max(f) - min(f)
 * b = (255 * min(f)) / max(f) - min(f)
 */
void contrast_stretching(const cv::Mat& input, cv::Mat& output, int type, float MAX_RANGE = 255.0f) {
	double min_pixel, max_pixel;
	cv::minMaxLoc(input, &min_pixel, &max_pixel);
	
	// DEBUG
	// std::cout << "min: " << min_pixel << ", max: " << max_pixel << std::endl;

	float a = (float) (MAX_RANGE / (max_pixel - min_pixel));
	float b = (float) (-1 * ((MAX_RANGE * min_pixel) / (max_pixel - min_pixel)));
	
	output.create(input.rows, input.cols, type);

	for(int r = 0; r < input.rows; ++r) {
		for(int c = 0; c < input.cols; ++c) {
			for(int k = 0; k < input.channels(); ++k) {
				float pixel_input = *((float*) &(input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()]));
				
				float stretched_pixel_input = a*pixel_input + b;
				
				if(type == CV_8UC1)
					output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()] = (uchar) stretched_pixel_input;
				else if(type == CV_32FC1)
					*((float*)(&output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()])) = stretched_pixel_input;
			}
		}
	}
}


/***************************/


/**
 * ES 1 - Max Pooling
 */
void maxPooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
	int padding_size = 0; // inizialmente lavoro senza padding, lo aggiungo al termine

	// Calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - size)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - size)/stride + 1);
	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_rows, out_cols, image.type());
	
	// padding per riottenere le dimensioni di input
	padding_size = floor((image.rows - out.rows)/2);

	cv::Mat padded_out;
	zero_padding(out, padding_size, padding_size, padded_out);
	out = padded_out.clone();

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, di dimensione size x size,
	// su cui effettuare il max pooling
	std::vector<int> kernel;

	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {
			if((r+size)*stride <= image.rows && (c+size)*stride <= image.cols)
			for(int k = 0; k < out.channels(); ++k) {
				// 2 cicli per analizzare ogni pixel dell'attuale kernel size x size
				for(int r_kernel = r; r_kernel < r+size; ++r_kernel) {
					for(int c_kernel = c; c_kernel < c+size; ++c_kernel) {
						// casto il puntatore al pixel corrente ad un int* 
						int current_pixel = (int) image.data[((stride*r_kernel*image.cols + stride*c_kernel)*image.channels() + k)*image.elemSize1()];

						// inserisco il pixel corrente nel vettore che identifica il kernel attuale
						kernel.push_back(current_pixel);
					}
				}

				// in uscita dal doppio for del kernel, ho size x size pixel nel kernel
				// ora devo calcolarne il massimo e metterlo nel pixel corrente sull'immagine di output
				std::vector<int>::const_iterator max_val = std::max_element(kernel.begin(), kernel.end());
				
				// svuoto il vector per il kernel successivo 
				kernel.clear();
				
				// accedo all'output usando gli appositi indici, dichiarati prima dei for
				out.data[(((r+padding_size)*out.cols + (c+padding_size))*out.channels() + k)*out.elemSize1()] = *max_val;
			}
		}
	}
}



/**
 * ES 2 - Average Pooling
 */
void averagePooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
		int padding_size = 0; // inizialmente lavoro senza padding, lo aggiungo al termine

	// Calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - size)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - size)/stride + 1);
	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_rows, out_cols, image.type());
	
	// padding per riottenere le dimensioni di input
	padding_size = floor((image.rows - out.rows)/2);

	cv::Mat padded_out;
	zero_padding(out, padding_size, padding_size, padded_out);
	out = padded_out.clone();

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, di dimensione size x size,
	// su cui effettuare il max pooling
	std::vector<int> kernel;

	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {
			if((r+size)*stride <= image.rows && (c+size)*stride <= image.cols)
			for(int k = 0; k < out.channels(); ++k) {
				// 2 cicli per analizzare ogni pixel dell'attuale kernel size x size
				for(int r_kernel = r; r_kernel < r+size; ++r_kernel) {
					for(int c_kernel = c; c_kernel < c+size; ++c_kernel) {
						// casto il puntatore al pixel corrente ad un int* 
						int current_pixel = (int) image.data[((stride*r_kernel*image.cols + stride*c_kernel)*image.channels() + k)*image.elemSize1()];

						// inserisco il pixel corrente nel vettore che identifica il kernel attuale
						kernel.push_back(current_pixel);
					}
				}

				// in uscita dal doppio for del kernel, ho size x size pixel nel kernel
				// ora devo calcolarne la media e metterla nel pixel corrente sull'immagine di output

				// sommo innanzi tutto i valori del vector, con la funzione accumulate
				float sum_val = std::accumulate(kernel.begin(), kernel.end(), 0);

				// divido quindi per il totale degli elementi
				float avg_val = sum_val / kernel.size();

				// svuoto il vector per il kernel successivo 
				kernel.clear();
				
				// accedo all'output usando gli appositi indici, dichiarati prima dei for
				out.data[(((r+padding_size)*out.cols + (c+padding_size))*out.channels() + k)*out.elemSize1()] = avg_val;
			}
		}
	}
}



/**
 * ES 3 - Convoluzione float
 * Per risolvere il problema dei bordi, eseguo uno zero-padding al termine del calcolo
 * (In questo modo, riporto l'output ad avere la stessa dimensione dell'input)
 */
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	int padding_rows = 0, padding_cols = 0; // inizialmente calcolo le dimensioni senza padding, quindi ridotte

	// Calcolo le dimensioni dopo aver applicato il kernel (formula tratta dalle slide di teoria)
	int out_tmp_rows = floor((image.rows + 2*padding_rows - kernel.rows)/stride + 1);
	int out_tmp_cols = floor((image.cols + 2*padding_cols - kernel.cols)/stride + 1);
	out.create(out_tmp_rows, out_tmp_cols, CV_32FC1);

	// ora calcolo il padding necessario per riportarmi alle dimensioni di input
	padding_rows = floor((image.rows - out_tmp_rows)/2);
	padding_cols = floor((image.cols - out_tmp_cols)/2);

	cv::Mat padded_out;
	zero_padding(out, padding_rows, padding_cols, padded_out);
	out = padded_out.clone();
	

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, pesata coi corrispondenti valori del kernel
	// grazie a questo, calcolerò poi il risultato della convoluzione ad ogni iterazione
	// i valori saranno chiaramente float
	std::vector<float> convolution_window;

	// 3 cicli per posizionarmi su ogni pixel dell'immagine di input, muovendomi di un passo pari alla stride
	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {
			if((r+kernel.rows)*stride <= image.rows && (c+kernel.cols)*stride <= image.cols)
			for(int k = 0; k < out.channels(); ++k) {

				// 2 coppie di indici:
				// r_kernel e c_kernel ciclano semplicemente sul kernel
				// r_image e c_image ciclano invece sui pixel dell'immagine che si trovano intorno al pixel attuale  
				for(int r_kernel = 0; r_kernel < kernel.rows; ++r_kernel) {
					for(int c_kernel = 0; c_kernel < kernel.cols; ++c_kernel) {
							// eseguo la somma di prodotti tra pixel sull'immagine (spostandomi di passo pari a stride su righe e colonne)
							// e pixel sul kernel
							float image_pixel = image.data[((stride*(r+r_kernel)*image.cols + stride*(c+c_kernel))*image.channels() + k)*image.elemSize1()];
							// il kernel è CV_32FC1: devo ricordarmi di castare il puntatore verso un float*, e solo successivamente prenderne l'elemento puntato
							float kernel_pixel = *((float*) &kernel.data[((r_kernel*kernel.cols + c_kernel)*kernel.channels() + k)*kernel.elemSize1()]);
							
							float current_pixel = image_pixel*kernel_pixel;	

							convolution_window.push_back(current_pixel);
						
					}
				}

				float sum_val = std::accumulate(convolution_window.begin(), convolution_window.end(), 0.0f);

				// svuoto il vector per la window successiva
				convolution_window.clear();

				// accedo all'immagine di output usando gli appositi indici, dichiarati prima dei for
				// l'output è CV_32FC1: devo ricordarmi di castare il puntatore verso un float*, e solo successivamente prenderne l'elemento puntato
				*((float*) &out.data[(((r+padding_rows)*out.cols + (c+padding_cols))*out.channels() + k)*out.elemSize1()]) = sum_val;
			}
		}
	}
}



/**
 * ES 4 - Convoluzione intera
 * Richiamo la convoluzione float, e successivamente riporto i valori in un range 0-255 con un contrast stretching
 */
void conv(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	cv::Mat convfloat_out;
	convFloat(image, kernel, convfloat_out, stride);

	contrast_stretching(convfloat_out, out, CV_8UC1);
}



/**
 * ES 5 - Kernel di un blur gaussiano orizzontale
 */
void gaussianKernel(float sigma, int radius, cv::Mat& kernel) {
	// dimensionamento del kernel
	int kernel_rows = 1;
	int kernel_cols = (radius*2) + 1;
	
	kernel.create(kernel_rows, kernel_cols, CV_32FC1);
	
	// dichiaro un vettore che conterrà i valori calcolati nel ciclo for
	std::vector<float> kernel_values;

	// riempimento del kernel, tramite la formula della gaussiana orizzontale
	for(int c = 0; c < kernel_cols; ++c) {
		float gaussian_c = exp(-1*pow(c-radius, 2) / (2*pow(sigma, 2))) / sqrt(2*M_PI*pow(sigma, 2));
		kernel_values.push_back(gaussian_c);
	}

	// calcolo la somma dei valori trovati
	float sum = std::accumulate(kernel_values.begin(), kernel_values.end(), 0.0f);

	// normalizzo i valori trovati, dividendo ciascuno per sum
	for(int c = 0; c < kernel_cols; ++c) kernel_values[c] /= sum;

	// assegno ad ogni pixel del kernel il suo valore finale
	for(int c = 0; c < kernel_cols; ++c) {
		*((float *) &kernel.data[c*kernel.elemSize1()]) = kernel_values[c];
	}
}



/**
 * ES 7 - Magnitudo e orientazione di Sobel 3x3
 */
void sobel(const cv::Mat& image, cv::Mat& magnitude, cv::Mat& orientation) {
	// applico una convoluzione di image con Sobel orizzontale e Sobel verticale
	// ottenendo quindi l'immagine di input derivata nelle due direzioni

	// creo i due filtri di Sobel
	int sobel_size = 3;

	float sobel_x_data[] { 1,  2,  1,
						   0,  0,  0,
						  -1, -2, -1 };

	cv::Mat kernel_sobel_x(sobel_size, sobel_size, CV_32FC1, sobel_x_data);

	float sobel_y_data[] { 1, 0, -1,
						   2, 0, -2,
						   1, 0, -1 };

	cv::Mat kernel_sobel_y(sobel_size, sobel_size, CV_32FC1, sobel_y_data);

	// applico le convoluzioni
	cv::Mat derivative_x;
	cv::Mat derivative_y;

	convFloat(image, kernel_sobel_x, derivative_x);
	convFloat(image, kernel_sobel_y, derivative_y);

	// ora applico le formule per il calcolo di magnitudo e orientazione
	// magnitudo = sqrt( (df/dx)^2 + (df/dy)^2 )
	// orientazione = arctan( (df/dy) / (df/dx) )

	magnitude.create(image.rows, image.cols, CV_32FC1);
	orientation.create(image.rows, image.cols, CV_32FC1);

	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {
				float current_derivative_x = *((float*) &(derivative_x.data[(r*derivative_x.cols + c)*derivative_x.elemSize()]));
				float current_derivative_y = *((float*) &(derivative_y.data[(r*derivative_y.cols + c)*derivative_y.elemSize()]));
				
				float* current_magnitude_pixel = ((float*) &(magnitude.data[(r*magnitude.cols + c)*magnitude.elemSize()]));
				float* current_orientation_pixel = ((float*) &(orientation.data[(r*orientation.cols + c)*orientation.elemSize()]));

				*current_magnitude_pixel = (float) sqrt(pow(current_derivative_x, 2) + pow(current_derivative_y, 2));
				
				*current_orientation_pixel = atan2(current_derivative_y, current_derivative_x);

				// se ho valori negativi sommo 2*M_PI per rifasare
				if(*current_orientation_pixel < 0) 
					*current_orientation_pixel += 2*M_PI;                                                
		}
	}
}



/**
 * ES 8 - Magnitudo e orientazione di Sobel 3x3
 * Date due coordinate r e c non discrete, calcolo un valore corrispondente ad esse
 * interpolando il valore reale dei 4 vicini discreti, moltiplicati con opportuni pesi
 * 
 * Uso un template di funzione, in quanto ho necessità di due versioni:
 * - uint_8, per l'ES 8
 * - float, per l'ES 9
 */
template <class T>
float bilinear(const cv::Mat& image, float r, float c) {
	// calcolo s e t, ossia le parti decimali di r e c
	int r_int = floor(r);
	int c_int = floor(c);
	float s = r - floor(r);
	float t = c - floor(c);

	// DEBUG
	// std::cout << "r: " << r << " c: " << c << std::endl;
	// std::cout << "floor(r): " << floor(r) << " floor(c): " << floor(c) << std::endl;
	// std::cout << "s: " << s << " t: " << t << std::endl;

	// calcolo il valore dei 4 vicini discreti
	T f00 = image.at<T>(r_int, c_int);
	T f10 = image.at<T>(r_int + 1, c_int);
	T f01 = image.at<T>(r_int, c_int + 1);
	T f11 = image.at<T>(r_int + 1, c_int + 1);

	// calcolo i contributi dei 4 vicini, moltiplicandoli per i pesi adeguati
	float contribute_00 = f00*(1 - s)*(1 - t);
	float contribute_10 = f10*s*(1 - t);
	float contribute_01 = f01*(1 - s)*t;
	float contribute_11 = f11*s*t;

	float final_value = contribute_00 + contribute_10 + contribute_01 + contribute_11;

	//DEBUG
	// std::cout << "final_value: " << final_value << std::endl;

	return final_value;
}



/**
 * ES 9 - Find peaks
 * Nota: non ritorno alcun valore, in quanto non necessario in questa funzione
 */
void findPeaks(const cv::Mat& magnitude, const cv::Mat& orientation, cv::Mat& out, float th) {

	out.create(magnitude.rows, magnitude.cols, magnitude.type());

	for(int r = 0; r < magnitude.rows; ++r) {
		for(int c = 0; c < magnitude.cols; ++c) {
			float theta = orientation.at<float>(r, c);

			// calcolo i vicini e1 ed e2, usando la formula data sulle slide
			float e1_x = r + 1*cos(theta);
			float e1_y = c + 1*sin(theta);
			float e2_x = r - 1*cos(theta);
			float e2_y = c - 1*sin(theta);

			// controllo se i vicini fuoriescono dal bordo
			// in tal caso sopprimo, subito il massimo e proseguo con la prossima iterazione
			if(
				e1_x > magnitude.rows - 1 || e1_y > magnitude.cols - 1 ||
				e2_x > magnitude.rows - 1 || e2_y > magnitude.cols - 1
			)
				out.at<float>(r, c) = 0.0f;

			// altrimenti, proseguo confrontando il pixel corrente coi vicini trovati
			else {
				// e1 ed e2 hanno coord non intere, dunque uso la funzione bilinear
				float e1_val = bilinear<float>(magnitude, e1_x, e1_y);
				float e2_val = bilinear<float>(magnitude, e2_x, e2_y);

				// non-maximum suppression, usando la formula data sulle slide
				if(
					magnitude.at<float>(r, c) >= e1_val &&
					magnitude.at<float>(r, c) >= e2_val &&
					magnitude.at<float>(r, c) >= th
				) 
					out.at<float>(r, c) = magnitude.at<float>(r, c);

				else
					out.at<float>(r, c) = 0.0f;
			}
		}
	}
}



/**
 * ES 10 - Double th
 * Nota: non ritorno alcun valore, in quanto non necessario in questa funzione
 */
void doubleTh(const cv::Mat& magnitude, cv::Mat& out, float th1, float th2) {
	out.create(magnitude.rows, magnitude.cols, CV_8UC1);

	for(int r = 0; r < out.rows; ++r) {
		for(int c = 0; c < out.cols; ++c) {
			float current_pixel = magnitude.at<float>(r, c);
			uint8_t* out_pixel = &(out.at<uint8_t>(r, c));

			if(current_pixel > th1) *out_pixel = 255;
			else if(current_pixel <= th1 && current_pixel > th2) *out_pixel = 128;
			else *out_pixel = 0;
		}
	}
}



/**
 * ES 11 - Canny
 * Nota: non ritorno alcun valore, in quanto non necessario in questa funzione
 */
void canny(const cv::Mat& image, cv::Mat& out, float th, float th1, float th2) {
	// Calcolo gradiente con sobel (magnitudo e orientazione)
	cv::Mat magnitude, orientation;
	sobel(image, magnitude, orientation);

	// Contrast stretching [0; 255] per poter applicare la prossima soglia
	double minVal, maxVal;
	cv::minMaxLoc(magnitude, &minVal, &maxVal);
	std::cout << "pre: " << minVal << ", " << maxVal << std::endl;
	contrast_stretching(magnitude.clone(), magnitude, CV_32FC1);
	cv::minMaxLoc(magnitude, &minVal, &maxVal);
	std::cout << "post: " << minVal << ", " << maxVal << std::endl;

	// Non-maximum suppression della magnitudo
	cv::Mat non_maximum_suppression;
	findPeaks(magnitude, orientation, non_maximum_suppression, th);

	// Sogliatura con isteresi
	doubleTh(non_maximum_suppression, out, th1, th2);
}



int main(int argc, char **argv) {
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

		// leggo l'immagine di input in un cv::Mat, in toni di grigio
		cv::Mat input_img = cv::imread(frame_name, CV_8UC1);
		if(input_img.empty()) {
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		//////////////////////
		//processing code here

		std::cout << "\nProcessing...\n\n";

		/****************************
		 *********** ES1 ************
		 ***************************/
		// scelgo le dimensioni di size e stride per effettuare il max pooling
		int size_max_pooling = 5;
		int stride_max_pooling = 1;

		// dichiaro la matrice contenente il risultato del max pooling
		// (il suo dimensionamento è gestito direttamente nella funzione maxPooling())
		cv::Mat out_max_pooling;
		maxPooling(input_img, size_max_pooling, stride_max_pooling, out_max_pooling);

		// DEBUG
		// std::cout << "output max pooling: " << out_max_pooling.rows << " " << out_max_pooling.cols << std::endl;



		/****************************
		 *********** ES2 ************
		 ***************************/
		// scelgo le dimensioni di size e stride per effettuare l'average pooling
		int size_avg_pooling = 5;
		int stride_avg_pooling = 1;

		// dichiaro la matrice contenente il risultato dell'average pooling
		// (il suo dimensionamento è gestito direttamente nella funzione averagePooling())
		cv::Mat out_avg_pooling;
		averagePooling(input_img, size_avg_pooling, stride_avg_pooling, out_avg_pooling);

		// DEBUG
		// std::cout << "output avg pooling: " << out_avg_pooling.rows << " " << out_avg_pooling.cols << std::endl;



		// /****************************
		//  *********** ES3 ************
		//  ***************************/
		// int kernel_convfloat_rows = 3;
		// int kernel_convfloat_cols = 3;
		// float kernel_convfloat_data[] { 1, 0, -1,
		// 						 	    2, 0, -2,
		// 						 	    1, 0, -1 };

		// cv::Mat kernel_convfloat(kernel_convfloat_rows, kernel_convfloat_cols, CV_32FC1, kernel_convfloat_data);

		// int stride_convfloat = 1;

		// // dichiaro la matrice contenente il risultato della convFloat()
		// // (il suo dimensionamento è gestito direttamente nella funzione convFloat())
		// cv::Mat out_convfloat;
		// convFloat(input_img, kernel_convfloat, out_convfloat, stride_convfloat);



		// /***************************
		// *********** ES4 ************
		// ****************************/
		int kernel_conv_rows = 3;
		int kernel_conv_cols = 3;
		float kernel_conv_data[] { 1, 0, -1,
								   2, 0, -2,
								   1, 0, -1 };

		cv::Mat kernel_conv(kernel_conv_rows, kernel_conv_cols, CV_32FC1, kernel_conv_data);
	
		int stride_conv = 1;

		// dichiaro la matrice contenente il risultato della conv()
		// (il suo dimensionamento è gestito direttamente nella funzione conv())
		cv::Mat out_conv;
		conv(input_img, kernel_conv, out_conv, stride_conv);

		// DEBUG
		// std::cout << "output conv: " << out_conv.rows << " " << out_conv.cols << std::endl;

		// /***************************
		// *********** ES6 ************
		// ****************************/
		cv::Mat kernel_gauss_horizontal;
		cv::Mat kernel_gauss_vertical;
		int kernel_gauss_radius = 5;

		cv::Mat out_gauss_horizontal;
		cv::Mat out_gauss_vertical;
		cv::Mat out_gauss_2D;
		
		int stride_gauss = 1;

		gaussianKernel(20.0f, kernel_gauss_radius, kernel_gauss_horizontal);
		// std::cout << kernel_gauss_horizontal << std::endl;

		// avrò padding solo a sinistra e a destra
		conv(input_img, kernel_gauss_horizontal, out_gauss_horizontal, stride_gauss);

		cv::transpose(kernel_gauss_horizontal, kernel_gauss_vertical);
		// avrò padding solo in alto e in basso
		conv(input_img, kernel_gauss_vertical, out_gauss_vertical, stride_gauss);

		// per ottenere il blur gaussiano 2D, parto dal blur gaussiano orizzontale
		// e ne calcolo la convoluzione con il kernel gaussiano verticale
		// avrò padding su tutti e 4 i lati
		conv(out_gauss_horizontal, kernel_gauss_vertical, out_gauss_2D, stride_gauss);
		
		// DEBUG
		// std::cout << "output gaussians: " << std::endl;
		// std::cout << out_gauss_horizontal.rows << ", " << out_gauss_horizontal.cols << std::endl;
		// std::cout << out_gauss_vertical.rows << ", " << out_gauss_vertical.cols << std::endl;
		// std::cout << out_gauss_2D.rows << ", " << out_gauss_2D.cols << std::endl;



		/***************************
		*********** ES7 ************
		****************************/
		cv::Mat magnitude;
		cv::Mat orientation;

		sobel(input_img, magnitude, orientation);
		
		// contrast_stretching(magnitude.clone(), magnitude, 1.0f);

		/***************************
		*********** ES8 ************
		****************************/
		bilinear<uint8_t>(input_img, 27.8f, 11.4f);

		
		
		/***************************
		*********** ES9 ************
		****************************/
		int th = 30;
		cv::Mat non_max_suppression;
		findPeaks(magnitude, orientation, non_max_suppression, th);

		

		/***************************
		*********** ES11 ************
		****************************/
		int th1 = 120;
		int th2 = 60;
		cv::Mat out_canny;
		canny(input_img, out_canny, th, th1, th2);
		/////////////////////

		// display input_img
		cv::namedWindow("input_img", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img", input_img);

		// display out_max_pooling
		cv::namedWindow("out_max_pooling", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_max_pooling", out_max_pooling);
		// cv::imwrite("out_max_pooling.png", out_max_pooling);

		// display out_avg_pooling
		cv::namedWindow("out_avg_pooling", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_avg_pooling", out_avg_pooling);
		// cv::imwrite("out_avg_pooling.png", out_avg_pooling);

		// display out_convfloat
		// cv::namedWindow("out_convfloat", cv::WINDOW_AUTOSIZE);
		// cv::imshow("out_convfloat", out_convfloat);

		// display out_conv
		cv::namedWindow("out_conv", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_conv", out_conv);
		// cv::imwrite("out_conv.png", out_conv);

		// display out_gauss_horizontal
		cv::namedWindow("out_gauss_horizontal", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_gauss_horizontal", out_gauss_horizontal);
		// cv::imwrite("out_gauss_horizontal.png", out_gauss_horizontal);		

		// display out_gauss_vertical
		cv::namedWindow("out_gauss_vertical", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_gauss_vertical", out_gauss_vertical);
		// cv::imwrite("out_gauss_vertical.png", out_gauss_vertical);
		
		// display out_gauss_2D
		cv::namedWindow("out_gauss_2D", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_gauss_2D", out_gauss_2D);
		// cv::imwrite("out_gauss_2D.png", out_gauss_2D);

		// display magnitude
		// Effettuo un contrast stretching sulla magnitude per poterla visualizzare
		contrast_stretching(magnitude.clone(), magnitude, CV_8UC1);
		cv::namedWindow("magnitude", cv::WINDOW_AUTOSIZE);
		cv::imshow("magnitude", magnitude);
		// cv::imwrite("magnitude.png", magnitude);

		// // display orientation
		cv::Mat adjMap;
		cv::convertScaleAbs(orientation, adjMap, 255 / (2*M_PI));
		cv::Mat falseColorsMap;
		cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);
		cv::imshow("orientation", falseColorsMap);
		// cv::imwrite("orientation2.png", falseColorsMap);

		// display non_max_suppression
		// contrast stretching per visualizzazione
		contrast_stretching(non_max_suppression.clone(), non_max_suppression, CV_8UC1);
		cv::namedWindow("non_max_suppression", cv::WINDOW_AUTOSIZE);
		cv::imshow("non_max_suppression", non_max_suppression);
		// cv::imwrite("non_max_suppression.png", non_max_suppression);

		// display out_canny
		// contrast stretching per visualizzazione
		// contrast_stretching(out_canny.clone(), out_canny);
		cv::namedWindow("out_canny", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_canny", out_canny);
		// cv::imwrite("out_canny.png", out_canny);
		
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

		++frame_number;
	}

	return 0;
}