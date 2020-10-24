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




/* Metodo 1: Frame precedente */
void frame_precedente(const cv::Mat& input_img, cv::Mat& old_img, int frame_number, int threshold, cv::Mat& background, cv::Mat& foreground) {
	if(frame_number == 0) { // nel frame 0, mi limito a prendere come background un'immagine nera
			background = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
			foreground = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
	} 

	// dal frame 1 applico la formula: B(n) = Img(n-1) 
	else {
		background = old_img.clone();
		compute_foreground(input_img, background, threshold, foreground);
	}

	old_img = input_img.clone();
}

/* Metodo 2: Media a finestra mobile */
void media_finestra_mobile(const cv::Mat& input_img, int frame_number, int k, std::vector<cv::Mat>& prev_k_frames, int threshold, cv::Mat& background, cv::Mat& foreground) {
	if(frame_number < k) { // per i primi k frame, mi limito a prendere come background un'immagine nera
		background = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
		foreground = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
		// aggiungo in coda al buffer il nuovo frame, dato dal frame corrente, quindi buffer torna ad avere size = k
		prev_k_frames.push_back(input_img.clone());
	}

	// dal frame k applico la formula
	else {

		// calcolo la media dei precedenti k frame in una matrice temporanea
		cv::Mat tmp_avg = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type()); 

		for(int v = 0; v < tmp_avg.rows; ++v)
			for(int u = 0; u < tmp_avg.cols; ++u)
				for(unsigned int i = 0; i < prev_k_frames.size(); ++i)
					tmp_avg.data[u + v*tmp_avg.cols] += prev_k_frames[i].data[u + v*prev_k_frames[i].cols] / k;

		background = tmp_avg.clone();

		// aggiungo in coda al buffer il nuovo frame, dato dal frame corrente, quindi buffer torna ad avere size = k
		prev_k_frames.push_back(input_img.clone());

		// elimino in testa dal buffer il primo dei k frame
		prev_k_frames.erase(prev_k_frames.begin());

		compute_foreground(input_img, background, threshold, foreground);
	}
}

/* Metodo 3: Media esponenziale */
void media_esponenziale_mobile(const cv::Mat& input_img, int frame_number, float alpha, cv::Mat& old_img, cv::Mat& old_background, int threshold, cv::Mat& background, cv::Mat& foreground) {
	if(frame_number == 0) { // nel frame 0, mi limito a prendere come background un'immagine nera
		background = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
		foreground = cv::Mat::zeros(input_img.rows, input_img.cols, input_img.type());
	}

	// dal frame 1 applico la formula: B(n) = alpha*B(n-1) + (1-alpha)*Img(n-1) 
	else {
		for(int v = 0; v < background.rows; ++v)
			for(int u = 0; u < background.cols; ++u)
				background.data[u + v*background.cols] = 
				alpha*old_background.data[u + v*old_background.cols] + (1 - alpha)*old_img.data[u + v*old_img.cols];
		
		compute_foreground(input_img, background, threshold, foreground);
	}

	old_img = input_img.clone();
	old_background = background.clone();
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

	// dichiarazioni
	/* Metodo 1: Frame precedente */
	cv::Mat background_1;
	cv::Mat foreground_1;
	cv::Mat old_img_1;
	int threshold_1 = 50;

	/* Metodo 2: Media a finestra mobile */
	cv::Mat background_2;
	cv::Mat foreground_2;
	int k = 10;
	int threshold_2 = 20;
	std::vector<cv::Mat> prev_k_frames;

	/* Metodo 3: Media mobile esponenziale */
	cv::Mat background_3;
	cv::Mat foreground_3;
	float alpha = 0.5f;
	int threshold_3 = 5;
	cv::Mat old_img_3;
	cv::Mat old_background_3;

	while(!exit_loop) {
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

		/* Metodo 1: Frame precedente */
		frame_precedente(input_img, old_img_1, frame_number, threshold_1, background_1, foreground_1);

		/* Metodo 2: Media a finestra mobile */
		media_finestra_mobile(input_img, frame_number, k, prev_k_frames, threshold_2, background_2, foreground_2);
			
		/* Metodo 3: Media mobile esponenziale */
		media_esponenziale_mobile(input_img, frame_number, alpha, old_img_3, old_background_3, threshold_3, background_3, foreground_3);
		/////////////////////

		//display
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

		/* Metodo 1: Frame precedente */
		cv::namedWindow("background_1", cv::WINDOW_NORMAL);
		cv::imshow("background_1", background_1);

		cv::namedWindow("foreground_1", cv::WINDOW_NORMAL);
		cv::imshow("foreground_1", foreground_1);

		/* Metodo 2: Media a finestra mobile */
		cv::namedWindow("background_2", cv::WINDOW_NORMAL);
		cv::imshow("background_2", background_2);

		cv::namedWindow("foreground_2", cv::WINDOW_NORMAL);
		cv::imshow("foreground_2", foreground_2);

		/* Metodo 3: Media mobile esponenziale */
		cv::namedWindow("background_3", cv::WINDOW_NORMAL);
		cv::imshow("background_3", background_3);

		cv::namedWindow("foreground_3", cv::WINDOW_NORMAL);
		cv::imshow("foreground_3", foreground_3);

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
