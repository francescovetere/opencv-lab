//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric> // std::accumulate()
#include <algorithm>

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


/*** FUNZIONI DI UTILITA'***/
void zero_padding(const cv::Mat& input, int padding_size, cv::Mat& output) {
	output = cv::Mat::zeros(input.rows + 2*padding_size, input.cols + 2*padding_size, input.type());

	for(int v = padding_size; v < output.rows-padding_size; ++v) {
		for(int u = padding_size; u < output.cols-padding_size; ++u) {
			for(int k = 0; k < output.channels(); ++k) {	
				output.data[((v*output.cols + u)*output.channels() + k)*output.elemSize1()]
				= input.data[(((v-padding_size)*input.cols + (u-padding_size))*input.channels() + k)*input.elemSize1()];
			}
		}
	}
}

/**
 * ES 1 - Max Pooling
 * Per risolvere il problema dei bordi, eseguo il max pooling 
 * partendo con il vertice top-left della finestra in (0,0)
 */
void maxPooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
	int padding_size = 0;

	// Calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - size)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - size)/stride + 1);

	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_rows, out_cols, image.type());

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, di dimensione size x size,
	// su cui effettuare il max pooling
	std::vector<int> kernel;

	// utilizzo due indici per capire dove mi trovo sull'immagine di output
	int current_out_r = -1;
	int current_out_c = -1;

	// 3 cicli per posizionarmi su ogni pixel dell'immagine di input, muovendomi di un passo pari alla stride
	for(int r = 0; r <= image.rows-std::max(size, stride); r+=stride) { // mi fermo quando arrivo a rows-stride o rows-size, dipende dal piu' grande
		// ad ogni riga dell'immagine di input, incremento la riga corrente sull'output, riazzerando la colonna corrente sull'output!
		++current_out_r;
		current_out_c = -1;
		for(int c = 0; c <= image.cols-std::max(size, stride); c+=stride) {
			// ad ogni colonna dell'immagine di input, incremento la colonna corrente sull'output
			++current_out_c;
			for(int k = 0; k < out.channels(); ++k) {
				// 2 cicli per analizzare ogni pixel dell'attuale kernel size x size
				for(int r_kernel = r; r_kernel < r+size; ++r_kernel) {
					for(int c_kernel = c; c_kernel < c+size; ++c_kernel) {
						// casto il puntatore al pixel corrente ad un int* 
						int current_pixel = (int) image.data[((r_kernel*image.cols + c_kernel)*image.channels() + k)*image.elemSize1()];

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
				out.data[((current_out_r*out.cols + current_out_c)*out.channels() + k)*out.elemSize1()] = *max_val;
			}
		}
	}
}

/**
 * ES 2 - Average Pooling
 * Per risolvere il problema dei bordi, eseguo l'average pooling 
 * partendo con il vertice top-left della finestra in (0,0)
 */
void averagePooling(const cv::Mat& image, int size, int stride, cv::Mat& out) {
	int padding_size = 0;

	// Calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - size)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - size)/stride + 1);

	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_rows, out_cols, image.type());

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, di dimensione size x size,
	// su cui effettuare l'average pooling
	std::vector<int> kernel;

	// utilizzo due indici per capire dove mi trovo sull'immagine di output
	int current_out_r = -1;
	int current_out_c = -1;

	// 3 cicli per posizionarmi su ogni pixel dell'immagine di input, muovendomi di un passo pari alla stride
	for(int r = 0; r <= image.rows-std::max(size, stride); r+=stride) { // mi fermo quando arrivo a rows-stride o rows-size, dipende dal piu' grande
		// ad ogni riga dell'immagine di input, incremento la riga corrente sull'output, riazzerando la colonna corrente sull'output!
		++current_out_r;
		current_out_c = -1;
		for(int c = 0; c <= image.cols-std::max(size, stride); c+=stride) {
			// ad ogni colonna dell'immagine di input, incremento la colonna corrente sull'output
			++current_out_c;
			for(int k = 0; k < out.channels(); ++k) {
				// 2 cicli per analizzare ogni pixel dell'attuale kernel size x size
				for(int r_kernel = r; r_kernel < r+size; ++r_kernel) {
					for(int c_kernel = c; c_kernel < c+size; ++c_kernel) {
						// casto il puntatore al pixel corrente ad un int* 
						int current_pixel = (int) image.data[((r_kernel*image.cols + c_kernel)*image.channels() + k)*image.elemSize1()];

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
				out.data[((current_out_r*out.cols + current_out_c)*out.channels() + k)*out.elemSize1()] = (int)avg_val;
			}
		}
	}
}

/**
 * ES 3 - Convoluzione float
 */
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	int padding_size = 0;

	// Calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - kernel.rows)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - kernel.cols)/stride + 1);

	out.create(out_rows, out_cols, CV_32FC1);

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, pesata coi corrispondenti valori del kernel
	// grazie a questo, calcolerò poi il risultato della convoluzione ad ogni iterazione
	// i valori saranno chiaramente float
	std::vector<float> convolution_window;
	
	// utilizzo due indici per capire dove mi trovo sull'immagine di output
	int current_out_r = -1;
	int current_out_c = -1;

	// 3 cicli per posizionarmi su ogni pixel dell'immagine di input, muovendomi di un passo pari alla stride
	for(int r = 0; r <= image.rows-std::max(kernel.rows, stride); r+=stride) {
		// ad ogni riga dell'immagine di input, incremento la riga corrente sull'output, riazzerando la colonna corrente sull'output!
		++current_out_r;
		current_out_c = -1;
		for(int c = 0; c <= image.cols-std::max(kernel.cols, stride); c+=stride) {
			// ad ogni colonna dell'immagine di input, incremento la colonna corrente sull'output
			++current_out_c;

			for(int k = 0; k < out.channels(); ++k) {
				// 2 cicli per analizzare ogni pixel dell'attuale kernel
				for(int r_kernel = 0; r_kernel < kernel.rows; ++r_kernel) {
					for(int c_kernel = 0; c_kernel < kernel.cols; ++c_kernel) {
				
						// eseguo la somma di prodotti
						float current_pixel = 
						(float)image.data[(((r+r_kernel)*image.cols + (c+c_kernel))*image.channels() + k)*image.elemSize1()]
						*
						(*((float*) &kernel.data[((r_kernel*kernel.cols + c_kernel)*kernel.channels() + k)*kernel.elemSize1()]));
						// (float)kernel.at<float>(r_kernel, c_kernel);

						// std::cout << (float)kernel.at<float>(r_kernel, c_kernel) << " * " <<
						// (float)image.data[(((r+r_kernel)*image.cols + (c+c_kernel))*image.channels() + k)*image.elemSize1()] << " = " <<
						// current_pixel << std::endl;
						
						// inserisco il pixel corrente nel vettore che identifica il kernel attuale
						// std::cout << current_pixel << " ";
						convolution_window.push_back(current_pixel);
					}
				}

				float sum_val = std::accumulate(convolution_window.begin(), convolution_window.end(), 0.0f);
				
				// std::cout << "\n\t";
				// std::for_each(convolution_window.begin(), convolution_window.end(), [](const float& x){std::cout << x << " ";});
				// std::cout << " = " << sum_val << "\n";

				// svuoto il vector per la window successiva 
				convolution_window.clear();

				// accedo all'output usando gli appositi indici, dichiarati prima dei for
				*((float*) &out.data[((current_out_r*out.cols + current_out_c)*out.channels() + k)*out.elemSize1()]) = sum_val;
				// out.at<float>(current_out_r, current_out_c) = sum_val;
				// std::cout << "out.data[" << (float)((current_out_r*out.cols + current_out_c)*out.channels() + k)*out.elemSize1() << "] = " << sum_val << std::endl;

			}
		}
	}

	// std::cout << out_rows << " " << out_cols << std::endl;
	// std::cout << current_out_r << " " << current_out_c << std::endl;
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

		/****************************
		 *********** ES1 ************
		 ***************************/
		// scelgo le dimensioni di size e stride per effettuare il max pooling
		int size_max_pooling = 6;
		int stride_max_pooling = 3;

		// dichiaro la matrice contenente il risultato del max pooling
		// (il suo dimensionamento è gestito direttamente nella funzione maxPooling())
		cv::Mat out_max_pooling;
		maxPooling(input_img, size_max_pooling, stride_max_pooling, out_max_pooling);

		
		
		/****************************
		 *********** ES2 ************
		 ***************************/
		// scelgo le dimensioni di size e stride per effettuare l'average pooling
		int size_avg_pooling = 6;
		int stride_avg_pooling = 3;

		// dichiaro la matrice contenente il risultato dell'average pooling
		// (il suo dimensionamento è gestito direttamente nella funzione averagePooling())
		cv::Mat out_avg_pooling;
		averagePooling(input_img, size_avg_pooling, stride_avg_pooling, out_avg_pooling);



		/****************************
		 *********** ES3 ************
		 ***************************/
		// scelgo le dimensioni di size e stride per effettuare l'average pooling
		int kernel_rows = 3;
		int kernel_cols = 3;
		float kernel_data[kernel_rows*kernel_cols] = {1, 0, -1,
													  2, 0, -2,
													  1, 0, -1};

		cv::Mat kernel(kernel_rows, kernel_cols, CV_32FC1, kernel_data);

		// std::cout << kernel << std::endl;


		// for(int r_kernel = 0; r_kernel < kernel.rows; ++r_kernel)
		// 		for(int c_kernel = 0; c_kernel < kernel.cols; ++c_kernel)
		// 			std::cout << (float)kernel.at<float>(r_kernel, c_kernel);
		// 			// std::cout << (float)kernel.data[(r_kernel*kernel.cols + c_kernel)*kernel.elemSize1()] << " ";
			
		int stride_conv_float = 3;

		// dichiaro la matrice contenente il risultato della convFloat()
		// (il suo dimensionamento è gestito direttamente nella funzione convFloat())
		cv::Mat out_conv_float;
		convFloat(input_img, kernel, out_conv_float, stride_conv_float);
		/////////////////////

		// display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);
		
		// display out_max_pooling
		cv::namedWindow("out_max_pooling", cv::WINDOW_NORMAL);
		cv::imshow("out_max_pooling", out_max_pooling);

		// display out_avg_pooling
		cv::namedWindow("out_avg_pooling", cv::WINDOW_NORMAL);
		cv::imshow("out_avg_pooling", out_avg_pooling);

		// display out_conv_float
		cv::namedWindow("out_conv_float", cv::WINDOW_NORMAL);
		cv::imshow("out_conv_float", out_conv_float);

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
