//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

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
	int max_levels = 255;
	int** histograms = compute_histogram_rgb(input_img, max_levels);

	// Equalizzo i 3 istogrammi b, g, r
	int* equalized_hb = equalize_histogram(histograms[0], max_levels, input_img.rows*input_img.cols);
	int* equalized_hg = equalize_histogram(histograms[1], max_levels, input_img.rows*input_img.cols);
	int* equalized_hr = equalize_histogram(histograms[2], max_levels, input_img.rows*input_img.cols);

	for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << equalized_hb[i] << std::endl;
	for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << equalized_hg[i] << std::endl;
	for(int i = 0; i < max_levels; ++i) std::cout << i << ": " << equalized_hr[i] << std::endl;

	output_img.create(input_img.rows, input_img.cols, CV_8UC3);
	for(int r = 0; r < output_img.rows; ++r)
		for(int c = 0; c < output_img.cols; ++c) {
			// b
			output_img.data[(r*output_img.cols + c)*output_img.channels() + 0] =
			equalized_hb[input_img.data[(r*input_img.cols + c)*input_img.channels() + 0]];
			// output_img.at<cv::Vec3b>(r, c)[0] = equalized_hb[input_img.at<cv::Vec3b>(r, c)[0]];

			// g		
			output_img.data[(r*output_img.cols + c)*output_img.channels() + 1] =
			equalized_hg[input_img.data[(r*input_img.cols + c)*input_img.channels() + 1]];	
			// output_img.at<cv::Vec3b>(r, c)[1] = equalized_hg[input_img.at<cv::Vec3b>(r, c)[1]];

			// r
			output_img.data[(r*output_img.cols + c)*output_img.channels() + 2] =
			equalized_hr[input_img.data[(r*input_img.cols + c)*input_img.channels() + 2]];
			// output_img.at<cv::Vec3b>(r, c)[2] = equalized_hr[input_img.at<cv::Vec3b>(r, c)[2]];
		}
}


/**
 * Binarizzo un'immagine di input data una certa soglia
 */
void binarize(const cv::Mat& input_img, int threshold, cv::Mat& output_img) {
	int max_intensity = 255;
	output_img.create(input_img.rows, input_img.cols, input_img.type());

	for(int v = 0; v < output_img.rows; ++v) {	
		for(int u = 0; u < output_img.cols; ++u) {
				if((int)input_img.data[(v*input_img.cols + u)] >= threshold)
					output_img.data[(v*output_img.cols + u)] = max_intensity;
				else output_img.data[(v*output_img.cols + u)] = 0;
			}
		}
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

// OR fra 1-pixel dell'immagine di input e corrispondenti pixel dell'elemento strutturale
// N.B. cornice piu' esterna immagine di input esclusa
void dilation(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	for(int v = 0; v < output.rows; ++v)
	{
		for(int u = 0; u < output.cols; ++u)
		{
			// se siamo su un 1-pixel dell'imm di input, procedo con l'effettuare l'OR sul vicinato
			if(input.data[(v+1)*input.cols + (u+1)] == max_intensity) {
				// esamino tutti i 9 pixel dell'elemento strutturale, e ne faccio l'OR coi corrispondenti pixel sull'immagine
				for(int i = 0; i < structural_element.rows; ++i) {
					for(int j = 0; j < structural_element.cols; ++j) {
						if(input.data[(v+i)*input.cols + (u+j)] != 0 || structural_element.data[i*structural_element.cols + j] != 0)
							output.data[(v+i)*output.cols + (u+i)] = max_intensity;
						else output.data[(v+i)*output.cols + (u+i)] = 0;
					}
				}
			}
		}
	}
}

// AND fra tutti i pixel dell'immagine di input e corrispondenti pixel dell'elemento strutturale
// N.B. cornice piu' esterna immagine di input esclusa
void erosion(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	bool ok;

	for(int v = 0; v < output.rows; ++v)
	{	
		for(int u = 0; u < output.cols; ++u)
		{
			ok = true;
			
			// esamino tutti i 9 pixel dell'elemento strutturale, e ne faccio l'AND coi corrispondenti pixel sull'immagine
			for(int i = 0; i < structural_element.rows; ++i) {
				for(int j = 0; j < structural_element.cols; ++j) {
					if(input.data[(v+i)*input.cols + (u+j)] == 0 || structural_element.data[i*structural_element.cols + j] == 0) {
						ok = false;
					}
				}
			}
				
			if(ok) output.data[(v*output.cols + u)] = max_intensity;
			else output.data[(v*output.cols + u)] = 0;
		}
	}
}

void opening(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	cv::Mat tmp(input.rows-2, input.cols-2, input.type());
	erosion(input, structural_element, tmp);
	dilation(tmp, structural_element, output);
}

void closing(const cv::Mat& input, const cv::Mat& structural_element, cv::Mat& output) {
	cv::Mat tmp(input.rows-2, input.cols-2, input.type());
	dilation(input, structural_element, tmp);
	erosion(tmp, structural_element, output);
}


/**
 *  Sovrappone l'immagine foreground sull'immagine background
 *  L'immagine foreground è supposta essere ovunque 0 tranne nei pixel effettivamente rilevanti  
 *  (devono avere stessa dimensione e stesso tipo CV_8UC1, ovviamente)
 */
void overlap_images(const cv::Mat& foreground, const cv::Mat& background, cv::Mat& output) {
  output.create(foreground.rows, foreground.cols, foreground.type());

  for(int r = 0; r < foreground.rows; ++r) {
		for(int c = 0; c < foreground.cols; ++c) {
      int val;
      // Se il foreground ha un valore != 0, considero il suo valore
			if(foreground.at<u_int8_t>(r, c) != 0)
				val = foreground.at<u_int8_t>(r, c);
      // Altrimenti, considero il valore del background
			else
				val = background.at<u_int8_t>(r, c);

			output.at<u_int8_t>(r, c) = val;
		}
  }
}

/** 
 * Funzione che effettua la trasposta
 */
void transpose(const cv::Mat& input, cv::Mat& output) {
	output.create(input.cols, input.rows, input.type());
	for(int r = 0; r < input.rows; ++r)
		for(int c = 0; c < input.cols; ++c)
			output.at<float>(c, r) = input.at<float>(r, c);
}

/**
 * Funzione che ritorna true <==> image[r, c] è un massimo locale, rispetto ad una finestra w_size x w_size
 */
bool is_local_maximum(const cv::Mat& image, int r, int c, int w_size) {
  float val = image.at<float>(r, c);

  for(int rr = -(w_size/2); rr <= (w_size/2); ++rr)
    for(int cc = -(w_size/2); cc <= (w_size/2); ++cc)
      if(image.at<float>(r+rr, c+cc) > val)
        return false;

  return true;
}

/** 
 * Funzione che trova i massimi locali e li mette in un vector
 */
void find_local_maxs(const cv::Mat& image, std::vector<float>& maxs, int w_size) {
	// scorro lungo tutta l'immagine (escludendo i bordi)
	for(int r = w_size/2; r < image.rows - w_size/2; ++r) {
		for(int c = w_size/2; c < image.cols - w_size/2; ++c) {
			// se il pixel è > di tutti i pixel della finestra, lo aggiungo al vector di output
			if(is_local_maximum(image, r, c, w_size)) 
				maxs.push_back(image.at<float>(r, c));	
		}
	}
}

/**
 * Funzione per la stampa delle immagini
 */
void display(std::string name, cv::Mat image) {
	int type = image.type();

	if(type == CV_32FC1)
    	contrast_stretching(image.clone(), image, CV_8UC1, 255.0f);
  
  	cv::namedWindow(name, cv::WINDOW_NORMAL);
	cv::imshow(name, image);
}

/**
 * Max Pooling
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
 * Average Pooling
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
 * Kernel di un blur gaussiano orizzontale
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
 * Magnitudo e orientazione di Sobel 3x3
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
 * Derivata x con gradiente 1x3
 */
void grad_x(const cv::Mat& image, cv::Mat& derivative_x) {
	// applico una convoluzione di image con gradiente orizzontale

	// creo il filtro gradiente
	float grad_x_data[] {-1, 0, 1};

	cv::Mat kernel_grad_x(1, 3, CV_32FC1, grad_x_data);
  
	// applico le convoluzioni
	conv(image, kernel_grad_x, derivative_x);
}

/**
 * Derivata y con gradiente 3x1
 */
void grad_y(const cv::Mat& image, cv::Mat& derivative_y) {
	// applico una convoluzione di image con gradiente verticale

	// creo il filtro gradiente
	float grad_x_data[] {1, 0, -1};

	cv::Mat kernel_grad_y(3, 1, CV_32FC1, grad_x_data);
  
	// applico le convoluzioni
	conv(image, kernel_grad_y, derivative_y);
}

/**
 * Date due coordinate r e c non discrete, calcolo un valore corrispondente ad esse
 * interpolando il valore reale dei 4 vicini discreti, moltiplicati con opportuni pesi
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
 * Non-maximum suppression
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



int main(int argc, char **argv) {
	//////////////////////
	//processing code here
		
	float kernel_convfloat_data[] { 1, 0, -1,
							 	    2, 0, -2,
							 	    1, 0, -1 };

	float kernel_for_local_maxes[] { 1, 0, -1, 2, 5,
							 	     2, 0, -2, 3, 4,
							 	     1, 12, -1, 8, 1, 
									 1, 7,  0, 2, 5,
									 1, 2,  1, 2, 5,
								   };

	cv::Mat M1(5, 5, CV_32FC1, kernel_for_local_maxes);
	std::cout << "M1: " << M1 << std::endl;
	
	std::vector<float> maxs;
	find_local_maxs(M1, maxs, 3);

	std::cout << "Local maxs: [";
	for(int i = 0; i < maxs.size(); ++i) std::cout << maxs[i] << " ";
	std::cout << "]\n";
	/////////////////////

	//display image
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", M1);

	return 0;
}
