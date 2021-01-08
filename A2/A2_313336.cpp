/*******************
 * Francesco Vetere
 * Matricola 313336
 * Assegnamento 2
 * 
 * Esempio di esecuzione:
 * ./A2_313336 ../images/input1.jpg ../images/book.jpg ../images/cover2.jpg
 *******************/

//OpenCV
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
#include <algorithm> /* find */
#include <random>
#include <iterator>
#include <cstdlib>   /* srand, rand */

// eigen
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Dense>

/***************************************************************************************/
/***************************************************************************************/
/******************************** FUNZIONI DI UTILITA' *********************************/
/***************************************************************************************/
/***************************************************************************************/

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
 * Derivata con Sobel 3x3
 */
void sobel(const cv::Mat& image, cv::Mat& derivative_x, cv::Mat& derivative_y) {
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
	convFloat(image, kernel_sobel_x, derivative_x);
	convFloat(image, kernel_sobel_y, derivative_y);
}

/**
 *  Sovrappone l'immagine foreground sull'immagine background 
 *  (devono avere stessa dimensione e stesso tipo CV_8UC1, ovviamente)
 */
void overlap_images(const cv::Mat& foreground, const cv::Mat& background, cv::Mat& output) {
  output.create(foreground.rows, foreground.cols, foreground.type());

  for(int r = 0; r < foreground.rows; r++) {
		for(int c = 0; c < foreground.cols; c++) {
      int val;
			if(background.at<u_int8_t>(r, c) != 0)
				val = background.at<u_int8_t>(r, c);
			else
				val = foreground.at<u_int8_t>(r, c);

			output.at<u_int8_t>(r, c) = val;
		}
  }
}

/**
 * Funzione che ritorna true <==> image[r, c] è un massimo locale, rispetto ad una finestra 3x3
 */
bool is_local_maximum(const cv::Mat& image, int r, int c) {
  float val = image.at<float>(r, c);

  for(int rr = -1; rr <= 1; ++rr)
    for(int cc = -1; cc <= 1; ++cc)
      if(image.at<float>(r+rr, c+cc) > val)
        return false;

  return true;
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

/***************************************************************************************/
/***************************************************************************************/
/**************************** FUNZIONI DELL'ASSEGNAMENTO *******************************/
/***************************************************************************************/
/***************************************************************************************/

void myHarrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh, std::string img_name) {
  // Calcolo le 2 derivate lungo x e lungo y
  cv::Mat Ix, Iy;

  // Prima di effettuare la derivata, eseguo un blur gaussiano per migliorarne l'effetto
  cv::Mat blurred_image;
  cv::GaussianBlur(image, blurred_image, cv::Size(3, 3), 1.0, 1.0);

  // Effettuo la derivata vera e propria
  sobel(blurred_image, Ix, Iy);

  // Altrimenti, con la funzione built-in di OpenCV:
  // cv::Sobel(blurred_image, Ix, CV_32FC1, 1, 0);
  // cv::Sobel(blurred_image, Iy, CV_32FC1, 0, 1);

  // Calcolo le 3 sottomatrici che compongono la matrice M
  cv::Mat Ix2, Iy2, IxIy;
  Ix2.create(Ix.rows, Ix.cols, CV_32FC1);
  Iy2.create(Iy.rows, Iy.cols, CV_32FC1);
  IxIy.create(Ix.rows, Ix.cols, CV_32FC1);

  Ix2 = Ix.mul(Ix);
  Iy2 = Iy.mul(Iy);
  IxIy = Ix.mul(Iy);

  // Calcolo la cornerness function con l'approssimazione vista sulle slide
  cv::Mat gIx2, gIy2, gIxIy;
  gIx2.create(Ix2.rows, Ix2.cols, CV_32FC1);
  gIy2.create(Iy2.rows, Iy2.cols, CV_32FC1);
  gIxIy.create(IxIy.rows, IxIy.cols, CV_32FC1);
  
  cv::GaussianBlur(Ix2, gIx2, cv::Size(3, 3), 1.0, 1.0);
  cv::GaussianBlur(Iy2, gIy2, cv::Size(3, 3), 1.0, 1.0);
  cv::GaussianBlur(IxIy, gIxIy, cv::Size(3, 3), 1.0, 1.0);

  cv::Mat response;
  response.create(gIx2.rows, gIx2.cols, CV_32FC1);
	response = gIx2.mul(gIy2) - gIxIy.mul(gIxIy) - alpha * (gIx2 + gIy2).mul(gIx2 + gIy2);

	// Infine, eseguo una non-maximum suppression 
  // (escludo chiaramente i bordi, su cui non potrei effettuare il controllo sul massimo locale)
	for (int r = 1; r < response.rows - 1; ++r)
		for (int c = 1; c < response.cols - 1; ++c)
				if(response.at<float>(r,c) > harrisTh && is_local_maximum(response, r, c))
					keypoints0.push_back(cv::KeyPoint((float)c, (float)r, 1.0f));

  // Visualizzazione passaggi intermedi
  /*
  display("Ix", Ix);
  display("Iy", Iy);
  display("Ix2", Ix2);
  display("Iy2", Iy2);
  display("IxIy", IxIy);

  display("gIx2", gIx2);
  display("gIy2", gIy2);
  display("gIxIy", gIxIy);

  // Per la response di Harris
  cv::Mat adjMap;
  cv::Mat falseColorsMap;
  double minr,maxr;
  cv::minMaxLoc(response, &minr, &maxr);
  cv::convertScaleAbs(response, adjMap, 255 / (maxr-minr));
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  cv::namedWindow("response", cv::WINDOW_NORMAL);
  cv::imshow("response", falseColorsMap);
  */

  // HARRIS CORNER END
  ////////////////////////////////////////////////////////
}

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

void myFindHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, const std::vector<cv::DMatch> & matches, int N, float epsilon, int sample_size, cv::Mat & H, std::vector<cv::DMatch> & matchesInlierBest) {
  // Inizializzo un seme per gli indici randomici
  srand(time(NULL));

  // vector contenenti i 4 match casuali presi da points0 e points1
  std::vector<cv::Point2f> sample0, sample1;

  // vector contenenti gli inliers trovati in ogni iterazione ransac
  std::vector<cv::Point2f> currInliers0, currInliers1;

  // vector contenenti i migliori inliers trovati in uscita dal ciclo di ransac
  std::vector<cv::Point2f> bestInliers0, bestInliers1;

  // Ciclo ransac
  for(int ransac_iteration = 0; ransac_iteration < N; ++ransac_iteration) {

    // Seleziono i 4 match casuali
    for(int i = 0; i < sample_size; ++i) {
      // Scelgo un indice random, col quale prelevo i punti dai 2 vector points0 e points1
      // I punti però non vengono inseriti se risultano duplicati, altrimenti chiaramente l'omografia non andrebbe a buon fine
      int index;
      cv::Point2f val0, val1;

      do {
        index = rand() % points1.size();

        val0 = points0[index];
        val1 = points1[index];

      }while(
             std::find(sample0.begin(), sample0.end(), val0) != sample0.end() ||
             std::find(sample1.begin(), sample1.end(), val1) != sample1.end()
            );

      // Una volta uscito dal do-while, sono certo di poter inserire i punti nei vettori sample0 e sample1
      sample0.push_back(val0);
      sample1.push_back(val1);
    }

    // Calcolo l'omografia coi samples appena scelti
    myFindHomographySVD(sample1, sample0, H);

    // Controllo quanti inliers rispettano l'omografia
    for(unsigned int i = 0; i < points0.size(); ++i) {
        cv::Mat p0_euclidean(2, 1, CV_64FC1);
        cv::Mat p1_homogeneus(3, 1, CV_64FC1);
        cv::Mat p1_euclidean(2, 1, CV_64FC1);

        p0_euclidean.at<double>(0,0) = points0[i].x;
		    p0_euclidean.at<double>(1,0) = points0[i].y;
        
        // Trasformo in coordinate omogenee
        p1_homogeneus.at<double>(0,0) = points1[i].x;
		    p1_homogeneus.at<double>(1,0) = points1[i].y;
		    p1_homogeneus.at<double>(2,0) = 1;

        cv::Mat Hp1_homogeneus = H*p1_homogeneus;
        cv::Mat Hp1_euclidean(2, 1, CV_64FC1);

        // Torno in coordinate euclidee
        Hp1_euclidean.at<double>(0, 0) = Hp1_homogeneus.at<double>(0, 0) / Hp1_homogeneus.at<double>(2, 0);
        Hp1_euclidean.at<double>(1, 0) = Hp1_homogeneus.at<double>(1, 0) / Hp1_homogeneus.at<double>(2, 0);
        
        // DEBUG
        // std::cout << ransac_iteration << ") p0, Hp1: " << p0_euclidean << "\n" << Hp1_euclidean << "\n";

        // Se |p0, Hp1| < epsilon ==> p0 e p1 sono inliers
        if(cv::norm(cv::Mat(p0_euclidean), cv::Mat(Hp1_euclidean)) < epsilon) {
          currInliers0.push_back(points0[i]);
          currInliers1.push_back(points1[i]);
        }
    }

    // Aggiorno i vector dei migliori inliers, nel caso in cui abbia ottenuto piu' inliers rispetto alle iterazioni precedenti
	  if(currInliers0.size() > bestInliers0.size()) {
			bestInliers0.clear(); bestInliers0 = currInliers0;
			bestInliers1.clear(); bestInliers1 = currInliers1;
		}

	  // Ad ogni iterazione, azzero i vector
		sample0.clear();
		sample1.clear();
		currInliers0.clear();
		currInliers1.clear();
  } // fine ciclo for ransac

  // Ricalcolo H coi migliori inliers trovati da ransac
  myFindHomographySVD(bestInliers1, bestInliers0, H);

  // Costruisco il vector matchesInlierBest, da restituire in output
	for(unsigned int i = 0; i < bestInliers0.size(); ++i)
		for(unsigned int j = 0; j < points0.size(); ++j)
			if(bestInliers0[i] == points0[j] && bestInliers1[i] == points1[j])
				matchesInlierBest.push_back(matches[j]);
}



/***************************************************************************************/
/***************************************************************************************/
/**************************************** MAIN *****************************************/
/***************************************************************************************/
/***************************************************************************************/

int main(int argc, char **argv) {

  if(argc < 4) {
    std::cerr << "Usage prova <image_filename> <book_filename> <new_cover_filename>" << std::endl;
    return 0;
  }

  // images
  cv::Mat input, cover, new_cover;

  // load image from file
  input = cv::imread(argv[1], CV_8UC1);
  if(input.empty()) {
    std::cout << "Error loading input image " << argv[1] << std::endl;
    return 1;
  }

  // load cover from file
  cover = cv::imread(argv[2], CV_8UC1);
  if(cover.empty()) {
    std::cout << "Error loading cover image " << argv[2] << std::endl;
    return 1;
  }

  // load new cover from file
  new_cover = cv::imread(argv[3], CV_8UC1);
  if(new_cover.empty()) {
    std::cout << "Error loading newcover image " << argv[3] << std::endl;
    return 1;
  }

  // Resize della nuova cover per assicurarsi che abbia le stesse dimensioni della cover originale
  cv::resize(new_cover.clone(), new_cover, cover.size());

  ////////////////////////////////////////////////////////
  /// HARRIS CORNER
  //
  float alpha = 0.04;
  float harrisTh = 500000*200;    //da impostare in base alla propria implementazione!!!!!

  std::vector<cv::KeyPoint> keypoints0, keypoints1;

  // FASE 1
  //
  // Qui sotto trovate i corner di Harris di OpenCV
  //
  // Da commentare e sostituire con la propria implementazione
  //
  // {
  //   std::vector<cv::Point2f> corners;
  //   int maxCorners = 0;
  //   double qualityLevel = 0.01;
  //   double minDistance = 10;
  //   int blockSize = 3;
  //   bool useHarrisDetector = true;
  //   double k = 0.04;

  //   cv::goodFeaturesToTrack( input,corners,maxCorners,qualityLevel,minDistance,cv::noArray(),blockSize,useHarrisDetector,k ); // estrae strong feature (k -> alpha)
  //   std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints0), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} ); // applica funzione a range vector e memorizza in altro range 3->size del keypoint

  //   corners.clear();
  //   cv::goodFeaturesToTrack( cover, corners, maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, k );
  //   std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints1), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} );
  // }
  //
  //
  //
  // Abilitare il proprio detector una volta implementato
  //
  //
  std::cout << "Computing harris corners..." << std::endl;

  // (Aggiungo un parametro finale per stampare a video il nome dell'immagine nei risultati temporanei)
  myHarrisCornerDetector(input, keypoints0, alpha, harrisTh, "input");
  myHarrisCornerDetector(cover, keypoints1, alpha, harrisTh, "cover");
  //
  //
  //


  std::cout << "keypoints0 " << keypoints0.size() << std::endl;
  std::cout << "keypoints1 " << keypoints1.size() << std::endl;
  //
  //
  ////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////
  /// CALCOLO DESCRITTORI E MATCHES
  //
  std::cout << "\nComputing descriptors..." << std::endl;
  int briThreshl = 30;
  int briOctaves = 3;
  int briPatternScales = 1.0;
  cv::Mat descriptors0, descriptors1;

  //dichiariamo un estrattore di features di tipo BRISK
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
  //calcoliamo il descrittore di ogni keypoint
  extractor->compute(input, keypoints0, descriptors0);
  extractor->compute(cover, keypoints1, descriptors1);

  //associamo i descrittori tra le due immagini
  std::vector<std::vector<cv::DMatch> > matches;
  std::vector<cv::DMatch> matchesDraw;
  cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING); // brute force matcher (TODO aggiornare con .create()), usa hamming distance tra vettori
  //matcher.radiusMatch(descriptors0, descriptors1, matches, input.cols*2.0);
  matcher.match(descriptors0, descriptors1, matchesDraw);

  //copio i match dentro a dei semplici vettori Point2f
  std::vector<cv::Point2f> points[2];
  for(unsigned int i=0; i<matchesDraw.size(); ++i) {
    points[0].push_back(keypoints0.at(matchesDraw.at(i).queryIdx).pt);
    points[1].push_back(keypoints1.at(matchesDraw.at(i).trainIdx).pt);
  }
  ////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////
  // CALCOLO OMOGRAFIA
  //
  //
  // E' obbligatorio implementare RANSAC.
  //
  // Per testare i corner di Harris inizialmente potete utilizzare findHomography di opencv, che include gia' RANSAC
  //
  // Una volta che avete verificato che i corner funzionano, passate alla vostra implementazione di RANSAC
  //
  //
  cv::Mat H;                                  //omografia finale
  std::vector<cv::DMatch> matchesInliersBest; //match corrispondenti agli inliers trovati
  std::vector<cv::Point2f> corners_cover;     //coordinate dei vertici della cover sull'immagine di input
  bool have_match=false;                      //verra' messo a true in caso ti match

  //
  // Verifichiamo di avere almeno 4 inlier per costruire l'omografia
  //
  //
  if(points[0].size() >= 4) {
    //
    // Soglie RANSAC
    //
    // Piuttosto critiche, da adattare in base alla propria implementazione
    //
    int N=5000;             //numero di iterazioni di RANSAC
    float epsilon = 3;      //distanza per il calcolo degli inliers


    // Dimensione del sample per RANSAC, questo e' fissato
    //
    int sample_size = 4;    //dimensione del sample di RANSAC

    //////////
    // FASE 2
    //
    //
    //
    // Inizialmente utilizzare questa chiamata OpenCV, che utilizza RANSAC, per verificare i vostri corner di Harris
    //
    //
    // cv::Mat mask;
    // H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), cv::RANSAC, 3, mask);
    // for(std::size_t i=0;i<matchesDraw.size();++i)
    //   if(mask.at<uchar>(0,i) == 1) matchesInliersBest.push_back(matchesDraw[i]);
    //
    //
    //
    // Una volta che i vostri corner di Harris sono funzionanti, commentare il blocco sopra e abilitare la vostra myFindHomographyRansac
    //

    std::cout << "\nExecuting ransac iterations..." << std::endl;
    myFindHomographyRansac(points[1], points[0], matchesDraw, N, epsilon, sample_size, H, matchesInliersBest);
    //
    //
    //

    std::cout<<std::endl<<"Ransac results: "<<std::endl;
    std::cout<<"Num inliers / total matches  "<<matchesInliersBest.size()<<" / "<<matchesDraw.size()<<std::endl;

    std::cout<<"H"<<std::endl<<H<<std::endl;

    //
    // Facciamo un minimo di controllo sul numero di inlier trovati
    //
    //
    float match_kpoints_H_th = 0.1;
    if(matchesInliersBest.size() > matchesDraw.size()*match_kpoints_H_th) {
      std::cout<<"MATCH!"<<std::endl;
      have_match = true;


      // Calcoliamo i bordi della cover nell'immagine di input, partendo dai corrispondenti nell'immagine target
      //
      //
      cv::Mat p  = (cv::Mat_<double>(3, 1) << 0, 0, 1);
      cv::Mat pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	      corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << cover.cols-1, 0, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	      corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << cover.cols-1, cover.rows-1, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	      corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << 0,cover.rows-1, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	      corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }
    }

    else {
      std::cout<<"Pochi inliers! "<<matchesInliersBest.size()<<"/"<<matchesDraw.size()<<std::endl;
    }


  }

  else {
    std::cout<<"Pochi match! "<<points[0].size()<<"/"<<keypoints0.size()<<std::endl;
  }
  ////////////////////////////////////////////////////////

  ////////////////////////////////////////////
  /// WINDOWS
  cv::Mat inputKeypoints;
  cv::Mat coverKeypoints;
  cv::Mat outMatches;
  cv::Mat outInliers;

  cv::drawKeypoints(input, keypoints0, inputKeypoints);
  cv::drawKeypoints(cover, keypoints1, coverKeypoints);

  cv::drawMatches(input, keypoints0, cover, keypoints1, matchesDraw, outMatches);
  cv::drawMatches(input, keypoints0, cover, keypoints1, matchesInliersBest, outInliers);

  // Sostituzione della nuova cover sulla cover originale 
  // Utilizzo la funzione cv::warpPerspective per ottenere il risultato migliore possibile
  cv::Mat transformed_cover;
	cv::warpPerspective(new_cover, transformed_cover, H, input.size());

  // Sovrappongo la nuova cover trasformata sull'immagine di input originale
  cv::Mat overlapped_input;
	overlap_images(input, transformed_cover, overlapped_input);

  /*** In alternativa, si puo' effettuare la trasformazione di ogni punto manualmente ***/
  /*** Il risultato è però un po' peggiore, in quanto cv::warpPerspective() effettua un'interpolazione ottimale ***/
  // for(int r = 0; r < new_cover.rows; ++r) {
  //   for(int c = 0; c < new_cover.cols; ++c) {
      // // Calcolo la destinazione finale di ciascun punto della nuova cover, grazie ad H
      // cv::Mat curr_point(3, 1, CV_64FC1);
      // curr_point.at<double>(0, 0) = c;
      // curr_point.at<double>(1, 0) = r;
      // curr_point.at<double>(2, 0) = 1;

      // cv::Mat transformed_point = H*curr_point;
      
      // double x = transformed_point.at<double>(0, 0) / transformed_point.at<double>(2, 0);
      // double y = transformed_point.at<double>(1, 0) / transformed_point.at<double>(2, 0);

      // if(x >= 0 && x <= input.cols - 1 && y >= 0 && y <= input.rows - 1) {
      //   input.at<uint8_t>(y, x) = new_cover.at<uint8_t>(r, c);
      // }
  //   }
  // }

  // Se abbiamo un match, disegniamo sull'immagine di input i contorni della cover
  if(have_match) {
    for(unsigned int i = 0;i<corners_cover.size();++i) {
      cv::line(input, cv::Point(corners_cover[i].x , corners_cover[i].y ), cv::Point(corners_cover[(i+1)%corners_cover.size()].x , corners_cover[(i+1)%corners_cover.size()].y ), cv::Scalar(255), 2, 8, 0);
    }
  }

  display("Input", input);
  display("BookCover", cover);
  display("InputKeypoints", inputKeypoints);
  display("coverKeypoints", coverKeypoints);
  display("Matches", outMatches);
  display("MatchesInliers", outInliers);
  display("NewInput", overlapped_input);
  
  cv::waitKey();

  return 0;
}
