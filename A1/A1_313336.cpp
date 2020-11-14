/*******************
 * Francesco Vetere
 * Matricola 313336
 * Assegnamento 1
 *******************/

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



/*****************************************************************
******************************************************************
********************* FUNZIONI DI UTILITA' ***********************
******************************************************************
******************************************************************/



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
				float pixel_input = *((float*) &(input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()]));
				
				float stretched_pixel_input = a*pixel_input + b;
				
				// distinguo il modo in cui accedo alla matrice di output in base al suo tipo
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



/*****************************************************************
******************************************************************
********************** FUNZIONI RICHIESTE ************************
******************************************************************
******************************************************************/



/**
 * ES 1 - Max Pooling
 * 
 * Nota: La tecnica di padding da me scelta consiste nel calcolare l'immagine di output senza padding,
 * ottenendo dunque un' immagine di dimensione ridotta, e successivamente aggiungere un padding totale pari a input.rows - out.rows
 * In tal modo, avrò sempre dim output = dim input, il che è molto comodo per elaborazioni in cascata
 * Tale tecnica viene utilizzata anche nei successivi esercizi
 */
void maxPooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
	// inizialmente eseguo i calcoli senza padding
	int padding_size = 0;

	// calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - size)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - size)/stride + 1);

	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_rows, out_cols, image.type());
	
	// padding per riottenere le dimensioni di input
	padding_size = floor((image.rows - out.rows)/2);
	zero_padding(out.clone(), padding_size, padding_size, out);

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, di dimensione size x size,
	// su cui effettuare il max pooling
	std::vector<int> kernel;

	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {

			// Effettuo i calcoli solamente se, tenuto conto di size e stride, non fuoriesco dall'immagine
			if((r+size)*stride <= image.rows && (c+size)*stride <= image.cols) {
				for(int k = 0; k < out.channels(); ++k) {

					// 2 cicli per analizzare l'attuale kernel size x size
					for(int r_kernel = r; r_kernel < r+size; ++r_kernel) {
						for(int c_kernel = c; c_kernel < c+size; ++c_kernel) {
							// estraggo il pixel corrente sull'immagine
							int current_pixel = (int) image.data[((stride*r_kernel*image.cols + stride*c_kernel)*image.channels() + k)*image.elemSize1()];

							// inserisco il pixel corrente nel vettore che identifica il kernel attuale
							kernel.push_back(current_pixel);
						}
					}

					// In uscita dal doppio for del kernel, ho size x size valori nel vector
					// ora devo calcolarne il massimo e metterlo nel pixel corrente sull'immagine di output
					std::vector<int>::const_iterator max_val = std::max_element(kernel.begin(), kernel.end());
					
					// svuoto il vector per l'iterazione successiva 
					kernel.clear();
					
					// accedo al pixel di output partendo dal pixel corrente nell'input, e sommando il padding necessario
					out.data[(((r+padding_size)*out.cols + (c+padding_size))*out.channels() + k)*out.elemSize1()] = *max_val;
				}
			}
		}
	}
}



/**
 * ES 2 - Average Pooling
 */
void averagePooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
	// inizialmente eseguo i calcoli senza padding
	int padding_size = 0;

	// calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - size)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - size)/stride + 1);

	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_rows, out_cols, image.type());
	
	// padding per riottenere le dimensioni di input
	padding_size = floor((image.rows - out.rows)/2);
	zero_padding(out.clone(), padding_size, padding_size, out);

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, di dimensione size x size,
	// su cui effettuare il max pooling
	std::vector<int> kernel;

	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {

			// Effettuo i calcoli solamente se, tenuto conto di size e stride, non fuoriesco dall'immagine
			if((r+size)*stride <= image.rows && (c+size)*stride <= image.cols) {
				for(int k = 0; k < out.channels(); ++k) {

					// 2 cicli per analizzare l'attuale kernel size x size
					for(int r_kernel = r; r_kernel < r+size; ++r_kernel) {
						for(int c_kernel = c; c_kernel < c+size; ++c_kernel) {
							// estraggo il pixel corrente sull'immagine
							int current_pixel = (int) image.data[((stride*r_kernel*image.cols + stride*c_kernel)*image.channels() + k)*image.elemSize1()];

							// inserisco il pixel corrente nel vettore che identifica il kernel attuale
							kernel.push_back(current_pixel);
						}
					}

					// In uscita dal doppio for del kernel, ho size x size valori nel vector
					// ora devo calcolarne la media e metterla nel pixel corrente sull'immagine di output

					// sommo i valori del vector, con la funzione accumulate
					float sum_val = std::accumulate(kernel.begin(), kernel.end(), 0.0f);

					// divido quindi per il totale degli elementi ed ottengo il valor medio
					float avg_val = sum_val / kernel.size();

					// svuoto il vector per l'iterazione successiva 
					kernel.clear();
				
					// accedo al pixel di output partendo dal pixel corrente nell'input, e sommando il padding necessario
					out.data[(((r+padding_size)*out.cols + (c+padding_size))*out.channels() + k)*out.elemSize1()] = avg_val;
				}
			}
		}
	}
}



/**
 * ES 3 - Convoluzione float
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
 * ES 4 - Convoluzione intera
 */
void conv(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	// Richiamo la convoluzione float, e successivamente riporto i valori in un range [0; 255] con un contrast stretching
	// convertendo già a CV_8UC1 per ottenere un'immagine pronta da passare a imshow 
	cv::Mat convfloat_out;
	convFloat(image, kernel, convfloat_out, stride);

	contrast_stretching(convfloat_out, out, CV_8UC1);
}



/**
 * ES 5 - Kernel di un blur gaussiano orizzontale
 */
void gaussianKernel(float sigma, int radius, cv::Mat& kernel) {
	// dimensionamento del kernel orizzontale
	int kernel_rows = 1;
	int kernel_cols = (radius*2) + 1;
	
	kernel.create(kernel_rows, kernel_cols, CV_32FC1);
	
	// dichiaro un vettore che conterrà i valori della funzione gaussiana ad ogni iterazione 
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

			*current_magnitude_pixel = sqrt(pow(current_derivative_x, 2) + pow(current_derivative_y, 2));
			
			*current_orientation_pixel = atan2(current_derivative_y, current_derivative_x);

			// atan2 restituisce valori nel range [-M_PI; +M_PI]
			// sommando al valore calcolato M_PI, mi riporto correttamente nel range desiderato [0; 2*M_PI]
			*current_orientation_pixel += M_PI;
		}
	}
}



/**
 * ES 8 - Magnitudo e orientazione di Sobel 3x3
 * 
 * Nota: Uso un template di funzione, in quanto ho necessità di due versioni:
 * - uint_8, per l'ES 8
 * - float, per l'ES 9
 */
template <class T>
float bilinear(const cv::Mat& image, float r, float c) {
	// Date due coordinate r e c non discrete, calcolo un valore corrispondente ad esse
 	// interpolando il valore reale dei 4 vicini discreti, moltiplicati con opportuni pesi
	
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
	// Confronto ogni pixel dell'immagine magnitude coi vicini e1 ed e2 scelti a distanza 1.0 lungo la direzione del gradiente
	// Essi avranno coordinate non discrete, dunque sfrutto l'interpolazione bilineare per ricavarne un valore pesato
	// Applico poi un sistema di condizioni per sopprimere i non massimi
	// Se e1 o e2 sforano dall'immagine, pongo l'attuale pixel output a 0 (metodo 1 sulle slide) 

	out.create(magnitude.rows, magnitude.cols, magnitude.type());

	for(int r = 0; r < magnitude.rows; ++r) {
		for(int c = 0; c < magnitude.cols; ++c) {
			float theta = orientation.at<float>(r, c);

			// calcolo i vicini e1 ed e2, usando le formule date sulle slide
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
	
	// Sistema di condizioni per ottenere una sogliatura doppia
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

	// Contrast stretching [0; 255] (mantenendo invariato il tipo, ovvero CV_32FC1) per poter applicare la prossima soglia
	contrast_stretching(magnitude.clone(), magnitude, CV_32FC1);

	// Non-maximum suppression della magnitude
	cv::Mat non_maximum_suppression;
	findPeaks(magnitude, orientation, non_maximum_suppression, th);

	// Non necessario applicare uno stretching in vista delle prossime due soglie,
	// in quanto in uscita dalla findPeaks non avrò nuovi valori

	// Sogliatura doppia
	doubleTh(non_maximum_suppression, out, th1, th2);
}



/*****************************************************************
******************************************************************
***************************** MAIN *******************************
******************************************************************
******************************************************************/



int main(int argc, char** argv) {
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
			sprintf(frame_name, (const char*)(args.image_name.c_str()), frame_number);
		else //single frame case
			sprintf(frame_name, "%s",args.image_name.c_str());

		//opening file
		std::cout << "Opening " << frame_name << std::endl;

		cv::Mat input_img = cv::imread(frame_name, CV_8UC1);

		if(input_img.empty()) {
			std::cout << "Unable to open " << frame_name << std::endl;
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



		/****************************
		 *********** ES3 ************
		 ***************************/

		int kernel_convfloat_rows = 3;
		int kernel_convfloat_cols = 3;
		float kernel_convfloat_data[] { 1, 0, -1,
								 	    2, 0, -2,
								 	    1, 0, -1 };

		cv::Mat kernel_convfloat(kernel_convfloat_rows, kernel_convfloat_cols, CV_32FC1, kernel_convfloat_data);

		int stride_convfloat = 1;

		// dichiaro la matrice contenente il risultato della convFloat()
		// (il suo dimensionamento è gestito direttamente nella funzione convFloat())
		cv::Mat out_convfloat;
		convFloat(input_img, kernel_convfloat, out_convfloat, stride_convfloat);

		// Nota: non visualizzo l'immagine out_convfloat perchè serve ovviamente un contrast stretching, applicato nell'ES 4



		/***************************
		*********** ES4 ************
		****************************/
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



		/***************************
		*********** ES6 ************
		****************************/

		cv::Mat out_gauss_horizontal;
		cv::Mat out_gauss_vertical;
		cv::Mat out_gauss_2D;
		
		// --- gaussiana orizzontale
		// (avrò padding solo a sinistra e a destra)
		cv::Mat kernel_gauss_horizontal;
		float sigma_gauss = 20.0f;
		int kernel_gauss_radius = 5;
		int stride_gauss = 1;

		gaussianKernel(sigma_gauss, kernel_gauss_radius, kernel_gauss_horizontal);
		conv(input_img, kernel_gauss_horizontal, out_gauss_horizontal, stride_gauss);

		// --- gaussiana verticale 
		// ottenuta come convoluzione dell'input per il kernel gaussiano orizzontale trasposto
		// (avrò padding solo in alto e in basso)
		cv::Mat kernel_gauss_vertical;

		// Utilizzo una funzione di OpenCV built-in per la trasposizione di matrici
		cv::transpose(kernel_gauss_horizontal, kernel_gauss_vertical);
		conv(input_img, kernel_gauss_vertical, out_gauss_vertical, stride_gauss);

		// --- gaussiana 2D
		// ottenuta come convoluzione della gaussiana orizzontale calcolata precedentemente
		// ed kernel gaussiano verticale
		// (avrò padding su tutti e 4 i lati)
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



		/***************************
		*********** ES8 ************
		****************************/

		// Eseguo un test a solo titolo di esempio
		bilinear<uint8_t>(input_img, 27.8f, 11.4f);

		
		
		/***************************
		*********** ES9 ************
		****************************/
		// Nota: th, th1 e th2 sono soglie esprimibili con numeri nell'intervallo [0; 255]

		int th = 30;
		cv::Mat non_max_suppression;
		findPeaks(magnitude, orientation, non_max_suppression, th);

		

		/***************************
		*********** ES11 ***********
		****************************/

		int th1 = 120;
		int th2 = 60;
		cv::Mat out_canny;
		canny(input_img, out_canny, th, th1, th2);

		
		
		/////////////////////



		/******************************
		*********** DISPLAY ***********
		******************************/

		// display input_img
		cv::namedWindow("input_img", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img", input_img);

		// display out_max_pooling
		cv::namedWindow("out_max_pooling", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_max_pooling", out_max_pooling);

		// display out_avg_pooling
		cv::namedWindow("out_avg_pooling", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_avg_pooling", out_avg_pooling);

		// display out_conv
		cv::namedWindow("out_conv", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_conv", out_conv);

		// display out_gauss_horizontal
		cv::namedWindow("out_gauss_horizontal", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_gauss_horizontal", out_gauss_horizontal);	

		// display out_gauss_vertical
		cv::namedWindow("out_gauss_vertical", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_gauss_vertical", out_gauss_vertical);
		
		// display out_gauss_2D
		cv::namedWindow("out_gauss_2D", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_gauss_2D", out_gauss_2D);

		// display magnitude
		// contrast stretching per visualizzazione
		contrast_stretching(magnitude.clone(), magnitude, CV_8UC1);
		cv::namedWindow("magnitude", cv::WINDOW_AUTOSIZE);
		cv::imshow("magnitude", magnitude);

		// display orientation
		cv::Mat adjMap;
		cv::convertScaleAbs(orientation, adjMap, 255 / (2*M_PI));
		cv::Mat falseColorsMap;
		cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);
		cv::imshow("orientation", falseColorsMap);

		// display non_max_suppression
		// contrast stretching per visualizzazione
		contrast_stretching(non_max_suppression.clone(), non_max_suppression, CV_8UC1);
		cv::namedWindow("non_max_suppression", cv::WINDOW_AUTOSIZE);
		cv::imshow("non_max_suppression", non_max_suppression);

		// display out_canny
		cv::namedWindow("out_canny", cv::WINDOW_AUTOSIZE);
		cv::imshow("out_canny", out_canny);
		


		//wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout << "key " << int(key) << std::endl;

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