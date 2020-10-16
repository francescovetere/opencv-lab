/**
 * Caricare l’immagine organs.pgm e applicare una binarizzazione con soglia automatica:
 * 1. Provare tutte le soglie da 1 a 255 
 * 2. Scegliere quella che minimizzi la somma pesata delle varianza all’interno di ogni gruppo (metodo di otsu)
 * 
 * Attenzione che l’immagine non e’ esattamente bimodale
 * Abbiamo almeno 3 livelli: sfondo, organi e non organi (in realtà anche di piu’)
 * Escludere dal calcolo delle statistiche per la soglia ideale (medie, varianze, ecc.) i punti dello sfondo (<50).
 */

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <climits> // INT_MAX

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
 * Funzione per il calcolo della varianza degli elementi un array arr
 * partendo da begin e terminando in end
 * varianza di n dati: sigma^2(n) = sum{i=1, n} (x_i - media_x)^2 / n
 */
double variance(int* arr, int begin, int end){
	int n = (end - begin + 1);

	double avg = 0; // media aritmetica

	for(int i = begin; i < end; i++) {
		avg += arr[i];
	} 

	avg = avg / n;

	double sigma_quad = 0; // varianza (da restituire come risultato)
	
	for(int i = begin; i < end; i++) {
		sigma_quad += pow(arr[i] - avg, 2);
	} 

	sigma_quad = sigma_quad / n;
	
	return sigma_quad;
}

/** Funzione per il calcolo dei due pesi w1 e w2, in funzione della soglia th
 * w0(th) = sum{i=0, th-1} (histogram[i])
 * w1(th) = sum{i=th, levels-1} (histogram[i])
 */
int weight(int* histogram, int begin, int end) {
	int res = 0;

	for(int i = begin; i < end; ++i) {
		res += histogram[i];
	}

	return res;
}

/**
 * Funzione che riceve in input un istogramma(assunto bimodale) ed il suo numero di livelli,
 * e ritorna in output la soglia migliore tramite metodo di Otsu 
 */
int otsu_threshold(int* histogram, int levels) {
	int best_treshold = 0;
	
	// tmp è la funzione di otsu data la soglia th attuale (inizialmente th = 1):
	// voglio trovare il tmp minimo, e quindi la th migliore
	double tmp = variance(histogram, 0, 1)*weight(histogram, 0, 1) +
					variance(histogram, 1, levels)*weight(histogram, 1, levels);

	double best_value = tmp;

	// ho gia' calcolato tmp per th = 1, quindi parto da th = 2
	for(int th = 2; th < levels; ++th) {
		std::cout << "th: " << th << " ";

		// tmp è la funzione di otsu data la soglia th attuale: voglio trovare il tmp minimo, e quindi la th migliore
		tmp = variance(histogram, 0, th)*weight(histogram, 0, th) +
					variance(histogram, th, levels)*weight(histogram, th, levels);
		
		std::cout << "variance: " << tmp << std::endl;

		if(tmp < best_value) {
			best_value = tmp;
			best_treshold = th;
		}
		
	}

	return best_treshold;
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

		cv::Mat input_img = cv::imread(frame_name, CV_8U); /// toni di grigio
		if(input_img.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}

		cv::Mat output_img(input_img.rows,input_img.cols,CV_8U);

		// INITIALIZE VARIABLES

		int max_intensity = 256 ;
		int* histogram = new int[max_intensity];

		for (int i = 0; i< max_intensity; i++){
			histogram[i] = 0;
		}

		histogram = compute_histogram(input_img, max_intensity);

		// for(int i = 0; i < max_intensity; ++i){
		// 	std::cout << "histogram[" << i << "] = " << histogram[i] << std::endl;
		// }  
		
		//Rimuovo lo sfondo, ovvero i primi 50 livelli
		int background_levels = 50;
		int* histogram_foreground = new int[max_intensity - background_levels];

		for(int i = 0; i < max_intensity - background_levels; ++i) histogram_foreground[i] = histogram[i + background_levels];
		// for(int i = 0; i < max_intensity - background_levels; ++i) std::cout << "histogram_foreground[" << i << "] = " << histogram_foreground[i] << std::endl; 
		
		// Calcolo la soglia ideale
		int threshold = otsu_threshold(histogram_foreground, max_intensity - background_levels) + background_levels;

		std::cout << "\notsu threshold: " << threshold << std::endl;

		// Binarizzo l'immagine con la soglia trovata
		for(int v = 0; v < output_img.rows; ++v)
		{	
			for(int u = 0; u < output_img.cols; ++u)
			{
				if((int)input_img.data[(v*input_img.cols + u)] >= threshold)
					output_img.data[(v*output_img.cols + u)] = max_intensity-1;

				else output_img.data[(v*output_img.cols + u)] = 0;
			}
		}

		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);
	
		//display output_img
		cv::namedWindow("output_img", cv::WINDOW_NORMAL);
		cv::imshow("output_img", output_img);

		// save output_img
		cv::imwrite("../../images/binarized-organs.pbm", output_img);
		
		std::cout << "../../images/binarized-organs.pbm has been successfully saved" << std::endl;

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
