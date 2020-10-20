/**
 * Implementate i metodi di background subtraction di base:
 * - Frame precedente.
 * - Media a finestra mobile.
 * - Media esponenziale.
 * 
 * Come cambia il background al variare di k e alfa?
 * Visualizzate su una finestra il background calcolato.
 * Partire dall’esempio fornito utilizzando le immagini in Candela.zip:
 * >simple -i Candela_m1.10_%06d.pgm -t 500 
 */

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

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


/********************************************************************************************
 *********************** FUNZIONI DI UTILITA' ***********************************************
 ********************************************************************************************/

/** Funzione per il calcolo dell'istogramma 
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
 * Funzione per il calcolo della varianza degli elementi un array
 * (partendo da begin e terminando in end)
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
 * Funzione che riceve in input un istogramma (assunto bimodale) ed il suo numero di livelli,
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
		// std::cout << "th: " << th << " ";

		// tmp è la funzione di otsu data la soglia th attuale: voglio trovare il tmp minimo, e quindi la th migliore
		tmp = variance(histogram, 0, th)*weight(histogram, 0, th) +
					variance(histogram, th, levels)*weight(histogram, th, levels);
		
		// std::cout << "variance: " << tmp << std::endl;

		if(tmp < best_value) {
			best_value = tmp;
			best_treshold = th;
		}
		
	}

	return best_treshold;
}


/**
 * Data img di input e background, calcola foreground come img - background
 * (tenendo conto di una threshold)
 */
void compute_foreground(const cv::Mat& img, const cv::Mat& background, int threshold, cv::Mat& foreground) {
	for(unsigned int i = 0; i < foreground.rows*foreground.cols*foreground.elemSize(); i += foreground.elemSize()) {
		int diff = abs(((int)(img.data[i]) - (int)(background.data[i])));
		if(diff > threshold) foreground.data[i] = diff;
		else foreground.data[i] = 0;
	}
}

/********************************************************************************************/



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

	cv::Mat background;
	cv::Mat foreground;
	cv::Mat old_img;
	int threshold;

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

		cv::Mat input_img= cv::imread(frame_name, CV_8UC1);
		if(input_img.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}


		//////////////////////
		//processing code here

		/* Frame precedente */

		if(frame_number == 0) { // nel frame 0, mi limito a prendere come background un'immagine nera
			background = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
			foreground = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
		} 

		// dal frame 1 applico la formula: B(n) = Img(n-1) 
		else {
			background = old_img.clone();
		}

		old_img = input_img.clone();

		if(frame_number != 0) {
			threshold = 255 - otsu_threshold(compute_histogram(input_img, 256), 256);
			std::cout << threshold << std::endl;
			compute_foreground(input_img, background, threshold, foreground);
		}

		/////////////////////

		//display
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

		cv::namedWindow("background", cv::WINDOW_NORMAL);
		cv::imshow("background", background);

		cv::namedWindow("foreground", cv::WINDOW_NORMAL);
		cv::imshow("foreground", foreground);

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
