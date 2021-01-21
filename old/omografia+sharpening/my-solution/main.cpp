//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <numeric>
#include <cmath>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

void myFindHomographySVD(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, cv::Mat & H) {
  // Creazione matrici A e H
  cv::Mat A(points1.size()*2, 9, CV_64FC1, cv::Scalar(0));
  H.create(3, 3, CV_64FC1);

  // Costruzione di A, per coppie di righe
  for(int i = 0; i < A.rows; i+=2) {
    float x1 = points1[i/2].x;
    float y1 = points1[i/2].y;
    float x2 = points0[i/2].x;
    float y2 = points0[i/2].y;

    // prima riga
    A.at<double>(i, 0) = -x1; A.at<double>(i, 1) = -y1; A.at<double>(i, 2) = -1;
    A.at<double>(i, 3) = 0; A.at<double>(i, 4) = 0; A.at<double>(i, 5) = 0;
    A.at<double>(i, 6) = x1*x2; A.at<double>(i, 7) = y1*x2; A.at<double>(i, 8) = x2;

    // seconda riga
    A.at<double>(i + 1, 0) = 0; A.at<double>(i + 1, 1) = 0; A.at<double>(i + 1, 2) = 0;
    A.at<double>(i + 1, 3) = -x1; A.at<double>(i + 1, 4) = -y1; A.at<double>(i + 1, 5) = -1;
    A.at<double>(i + 1, 6) = x1*y2; A.at<double>(i + 1, 7) = y1*y2; A.at<double>(i + 1, 8) = y2;
  }

  // Calcolo della decomposizione A = UDVt
  cv::Mat D, U, Vt, V;
  cv::SVD::compute(A, D, U, Vt, cv::SVD::Flags::FULL_UV);

  cv::transpose(Vt, V);

  // L'ultima colonna di V contiene il vettore soluzione di 9 elementi, che assegno alla matrice H
  for(int r = 0; r < H.rows; ++r)
    for(int c = 0; c < H.cols; ++c)
      H.at<double>(r, c) = V.at<double>(r*H.rows + c, V.cols - 1);
  
  // Normalizzo per ottenere H.at<double>(2, 2) = 1
  H /= H.at<double>(2,2);

  // std::cout<<"myH"<<std::endl<<H<<std::endl;
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


//////////////////////////////////////////////
/// EX1
//
// Nota la posizione dei 4 angoli della copertina del libro nell'immagine "input.jpg"
// generare la corrispondente immagine vista dall'alto, senza prospettiva.
//
// Si tratta di trovare l'opportuna trasformazione che fa corrispondere la patch di immagine
// input.jpg corrispondente alla copertina del libro con la vista dall'alto della stessa.
//
// Che tipo di trasformazione e'? Come si puo' calcolare con i dati forniti?
//
// E' possibile utilizzare alcune funzioni di OpenCV
//
void WarpBookCover(const cv::Mat & image, cv::Mat & output, const std::vector<cv::Point2f> & corners_src) {
	std::vector<cv::Point2f> corners_out;

	/*
	* YOUR CODE HERE
	*
	*
	*/

    std::vector<cv::Point2f> corners_dst = { cv::Point2f(0, 0), //top left
    		                                 cv::Point2f(output.cols-1, 0), //top right
    									     cv::Point2f(0, output.rows-1), //bottom left
										     cv::Point2f(output.cols-1, output.rows-1)};//bottom right

    // Calcolo l'omografia H
    cv::Mat H;
    myFindHomographySVD(corners_dst, corners_src, H);
    // H = cv::findHomography(corners_src, corners_dst, cv::RANSAC);

    std::cout << "H:" << H << std::endl;
    // cv::Mat Hinv = H.inv();

    for(int r = 0; r < output.rows; ++r) {
        for(int c = 0; c < output.cols; ++c) {
            // Calcolo la destinazione finale di ciascun punto della nuova cover, grazie ad H
            cv::Mat curr_point(3, 1, CV_64FC1);
            curr_point.at<double>(0, 0) = c;
            curr_point.at<double>(1, 0) = r;
            curr_point.at<double>(2, 0) = 1;

            cv::Mat transformed_point = H*curr_point;
            
            double x = transformed_point.at<double>(0, 0) / transformed_point.at<double>(2, 0);
            double y = transformed_point.at<double>(1, 0) / transformed_point.at<double>(2, 0);
			if(x >= 0 && x <= image.cols - 1 && y >= 0 && y <= image.rows - 1) {
        		output.at<cv::Vec3b>(r, c)[0] = image.at<cv::Vec3b>(y, x)[0];
				output.at<cv::Vec3b>(r, c)[1] = image.at<cv::Vec3b>(y, x)[1];
				output.at<cv::Vec3b>(r, c)[2] = image.at<cv::Vec3b>(y, x)[2];
      		}
        }
    }
    
	// cv::warpPerspective(image, output, H, output.size());
}
/////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////
/// EX2
//
// Applicare il filtro di sharpening visto a lezione
//
// Per le convoluzioni potete usare le funzioni sviluppate per il primo assegnamento
//
//
void sharpening(const cv::Mat & image, cv::Mat & output, float alpha) {
	output = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));

    cv::Mat LoG_conv_b, LoG_conv_g, LoG_conv_r;

	/*
	* YOUR CODE HERE
	*
	*
	*/
    float LoG_data[] { 0, 1, 0,
					   1, -4, 1,
					   0, 1, 0};

	cv::Mat LoG_kernel(3, 3, CV_32FC1, LoG_data);
  
	// Creo le 3 immagini per i 3 canali
    cv::Mat blue(image.rows, image.cols, CV_8UC1);
	cv::Mat green(image.rows, image.cols, CV_8UC1);
	cv::Mat red(image.rows, image.cols, CV_8UC1);
    
    for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {
			blue.at<uint8_t>(r, c) = image.at<cv::Vec3b>(r, c)[0];
			green.at<uint8_t>(r, c) = image.at<cv::Vec3b>(r, c)[1];
			red.at<uint8_t>(r, c) = image.at<cv::Vec3b>(r, c)[2];
		}
	}

    // Per ogni immagine, eseguo la convoluzione con LoG
	convFloat(blue, LoG_kernel, LoG_conv_b);
    convFloat(green, LoG_kernel, LoG_conv_g);
    convFloat(red, LoG_kernel, LoG_conv_r);

    for(int r = 0; r < output.rows; ++r) {
        for(int c = 0; c < output.cols; ++c) {
            int val_b = std::fabs(blue.at<uint8_t>(r, c) - alpha*(LoG_conv_b.at<float>(r, c)));
            if(val_b > 255) val_b = 255; // clipping

            int val_g = std::fabs(green.at<uint8_t>(r, c) - alpha*(LoG_conv_g.at<float>(r, c)));
            if(val_g > 255) val_g = 255; // clipping
            
            int val_r = std::fabs(red.at<uint8_t>(r, c) - alpha*(LoG_conv_r.at<float>(r, c)));
            if(val_r > 255) val_r = 255; // clipping

            output.at<cv::Vec3b>(r, c)[0] = val_b;
            output.at<cv::Vec3b>(r, c)[1] = val_g;
            output.at<cv::Vec3b>(r, c)[2] = val_r;
        }
    }

}
//////////////////////////////////////////////


int main(int argc, char **argv) {
    
    if (argc != 2)
    {
        std::cerr << "Usage ./prova <image_filename>" << std::endl;
        return 0;
    }
    
    //images
    cv::Mat input;

    // load image from file
    input = cv::imread(argv[1]);
	if(input.empty())
	{
		std::cout<<"Error loading input image "<<argv[1]<<std::endl;
		return 1;
	}

    //////////////////////////////////////////////
    /// EX1
    //
    // Creare un'immagine contenente la copertina del libro input come vista "dall'alto" (senza prospettiva)
    //
    //
	//

	// Dimensioni note e fissate dell'immagine di uscita (vista dall'alto):
	constexpr int outwidth = 431;
	constexpr int outheight = 574;
	cv::Mat outwarp(outheight, outwidth, input.type(), cv::Scalar(0));

	//posizioni note e fissate dei quattro corner della copertina nell'immagine input
    std::vector<cv::Point2f> pts_src = { cv::Point2f(274,189), //top left
    		                             cv::Point2f(631,56), //top right
										 cv::Point2f(722,764),//bottom left
                                         cv::Point2f(1042,457)}; //bottom right

    WarpBookCover(input, outwarp, pts_src);
    //////////////////////////////////////////////







    //////////////////////////////////////////////
    /// EX2
    //
    // Applicare uno sharpening all'immagine cover
    //
    // Immagine = Immagine - alfa(LoG * Immagine)
    //
    //
    // alfa e' una costante float, utilizziamo 0.5
    //
    //
    // LoG e' il Laplaciano del Gaussiano. Utilizziamo l'approssimazione 3x3 vista a lezione
    //
    //
    // In questo caso serve fare il contrast stratching nelle convoluzioni?
    //
    //

    //immagine di uscita sharpened
	cv::Mat sharpened(input.rows, input.cols, CV_8UC3);

	//convertiamo l'immagine della copertina a toni di grigio, per semplicita'
	cv::Mat inputgray(input.rows, input.cols, CV_8UC3);
	cv::cvtColor(input, inputgray, cv::COLOR_BGR2GRAY);

	sharpening(input, sharpened, 0.8);
    //////////////////////////////////////////////






    ////////////////////////////////////////////
    /// WINDOWS
    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input", input);
    
    cv::Mat outimage_win(std::max(input.rows, outwarp.rows), input.cols+outwarp.cols, input.type(), cv::Scalar(0));
    input.copyTo(outimage_win(cv::Rect(0,0,input.cols, input.rows)));
    outwarp.copyTo(outimage_win(cv::Rect(input.cols,0,outwarp.cols, outwarp.rows)));

    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::imshow("Output", outimage_win);

    cv::namedWindow("Input Gray", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray", inputgray);

    cv::namedWindow("Input Gray Sharpened", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray Sharpened", sharpened);

    cv::waitKey();

    return 0;
}