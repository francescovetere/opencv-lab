/*******************
 * Francesco Vetere
 * Matricola 313336
 * Assegnamento 2
 * 
 * Esempio di comando per l'esecuzione:
 * ./A2_313336 ../images/input1.jpg ../images/book.jpg ../images/cover2.jpg
 *******************/

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
#include <algorithm>
#include <random>
#include <iterator>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

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
 * ES 7 (modificato per ottenere solo le derivate) - Derivata con Sobel 3x3
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


// Aggiungo un parametro finale per le stampe dei risultati intermedi
void myHarrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh, std::string img_name) {
  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   *
   *
   * E' ovviamente vietato utilizzare un detector di OpenCv....
   *
   */
  ////////////////////////////////////////////////////////
  /// HARRIS CORNER
  //

  cv::Mat derivative_x, derivative_y;
  sobel(image, derivative_x, derivative_y);

  // Visualizzazione passaggi intermedi
  contrast_stretching(derivative_x.clone(), derivative_x, CV_8UC1, 255.0f);
  cv::namedWindow(img_name + "derivative_x", cv::WINDOW_AUTOSIZE);
  cv::imshow(img_name + "derivative_x", derivative_x);

  contrast_stretching(derivative_y.clone(), derivative_y, CV_8UC1, 255.0f);
  cv::namedWindow(img_name + "derivative_y", cv::WINDOW_AUTOSIZE);
  cv::imshow(img_name + "derivative_y", derivative_y);

  // Disegnate tutti i risultati intermedi per capire se le cose funzionano
  //
  // Per la response di Harris:
  //    cv::Mat adjMap;
  //    cv::Mat falseColorsMap;
  //    double minr,maxr;
  //
  //    cv::minMaxLoc(response1(roi), &minr, &maxr);
  //    cv::convertScaleAbs(response1(roi), adjMap, 255 / (maxr-minr));
  //    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  //    cv::namedWindow("response1", cv::WINDOW_NORMAL);
  //    cv::imshow("response1", falseColorsMap);

  // HARRIS CORNER END
  ////////////////////////////////////////////////////////
}

void myFindHomographySVD(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, cv::Mat & H) {
  cv::Mat A(points1.size()*2,9, CV_64FC1, cv::Scalar(0));

  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   *
   * Utilizzate la funzione:
   * cv::SVD::compute(A,D, U, Vt);
   *
   * In pratica dovete costruire la matrice A opportunamente e poi prendere l'ultima colonna di V
   *
   */

  // ricordatevi di normalizzare alla fine
  H/=H.at<double>(2,2);

  //std::cout<<"myH"<<std::endl<<H<<std::endl;
}

void myFindHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, const std::vector<cv::DMatch> & matches, int N, float epsilon, int sample_size, cv::Mat & H, std::vector<cv::DMatch> & matchesInlierBest) {
  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   * Implementare il calcolo dell'omografia con un loop RANSAC
   *
   *
   * E' vietato utilizzare:
   * 		cv::findHomography(sample1, sample0, CV_RANSAC)
   *
   *
   *
   * Inizialmente utilizzare la findHomografy di OpenCV dentro al vostro loop RANSAC
   *
   *      cv::findHomography(sample1, sample0, 0)
   *
   *
   * Una volta verificato che il loop RANSAC funziona, sostituire la findHomography di OpenCV con la vostra
   *      cv::Mat HR;
   *      myFindHomographySVD( cv::Mat(sample[1]), cv::Mat(sample[0]), HR);
   *
   */
}


int main(int argc, char **argv) {

  if(argc < 4) {
    std::cerr << "Usage prova <image_filename> <book_filename> <alternative_cover_filename>" << std::endl;
    return 0;
  }

  // images
  cv::Mat input, cover, newcover;

  // load image from file
  input = cv::imread(argv[1], CV_8UC1);
  if(input.empty()) {
    std::cout << "Error loading input image " << argv[1] << std::endl;
    return 1;
  }

  // load image from file
  cover = cv::imread(argv[2], CV_8UC1);
  if(cover.empty()) {
    std::cout << "Error loading book image " << argv[2] << std::endl;
    return 1;
  }

  ////////////////////////////////////////////////////////
  /// HARRIS CORNER
  //
  float alpha = 0.04;
  float harrisTh = 500000;    //da impostare in base alla propria implementazione!!!!!

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
  // (Aggiungo un parametro finale per stampare a video il nome dell'immagine nei risultati temporanei)
  myHarrisCornerDetector(input, keypoints0, alpha, harrisTh, "input");
  myHarrisCornerDetector(cover, keypoints1, alpha, harrisTh, "cover");
  //
  //
  //


  std::cout<<"keypoints0 "<<keypoints0.size()<<std::endl;
  std::cout<<"keypoints1 "<<keypoints1.size()<<std::endl;
  //
  //
  ////////////////////////////////////////////////////////

  /* Questa parte non va toccata */
  ////////////////////////////////////////////////////////
  /// CALCOLO DESCRITTORI E MATCHES
  //
  int briThreshl = 30;
  int briOctaves = 3;
  int briPatternScales = 1.0;
  cv::Mat descriptors0, descriptors1;

  //dichiariamo un estrattore di features di tipo BRISK
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
  //calcoliamo il descrittore di ogni keypoint
  extractor->compute(input, keypoints0, descriptors0);
  extractor->compute(cover, keypoints1, descriptors1);

  //associamo i descrittori tra me due immagini
  std::vector<std::vector<cv::DMatch> > matches;
  std::vector<cv::DMatch> matchesDraw;
  cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING); // brute force matcher (TODO aggiornare con .create()), usa hamming distance tra vettori
  //matcher.radiusMatch(descriptors0, descriptors1, matches, input.cols*2.0);
  matcher.match(descriptors0, descriptors1, matchesDraw);

  //copio i match dentro a dei semplici vettori oint2f
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
    int N=50000;            //numero di iterazioni di RANSAC
    float epsilon = 3;      //distanza per il calcolo degli inliers


    // Dimensione del sample per RANSAC, quiesto e' fissato
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
    cv::Mat mask;
    H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), cv::RANSAC, 3, mask);
    for(std::size_t i=0;i<matchesDraw.size();++i)
      if(mask.at<uchar>(0,i) == 1) matchesInliersBest.push_back(matchesDraw[i]);
    //
    //
    //
    // Una volta che i vostri corner di Harris sono funzionanti, commentare il blocco sopra e abilitare la vostra myFindHomographyRansac
    //
    //myFindHomographyRansac(points[1], points[0], matchesDraw, N, epsilon, sample_size, H, matchesInliersBest);
    //
    //
    //

    std::cout<<std::endl<<"Risultati Ransac: "<<std::endl;
    std::cout<<"Num inliers / match totali  "<<matchesInliersBest.size()<<" / "<<matchesDraw.size()<<std::endl;

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


  // se abbiamo un match, disegniamo sull'immagine di input i contorni della cover
  if(have_match) {
    for(unsigned int i = 0;i<corners_cover.size();++i) {
      cv::line(input, cv::Point(corners_cover[i].x , corners_cover[i].y ), cv::Point(corners_cover[(i+1)%corners_cover.size()].x , corners_cover[(i+1)%corners_cover.size()].y ), cv::Scalar(255), 2, 8, 0);
    }
  }

  // cv::namedWindow("Input", cv::WINDOW_AUTOSIZE); // Noi dobbiamo fare passettino in più nel passo 5: prendere la cover e inserirla qui: doppio ciclo in cui spalmo la cover qui, usando H
  // cv::imshow("Input", input);

  // cv::namedWindow("BookCover", cv::WINDOW_AUTOSIZE);
  // cv::imshow("BookCover", cover);

  cv::namedWindow("inputKeypoints", cv::WINDOW_AUTOSIZE);
  cv::imshow("inputKeypoints", inputKeypoints);

  cv::namedWindow("coverKeypoints", cv::WINDOW_AUTOSIZE);
  cv::imshow("coverKeypoints", coverKeypoints);

  // cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE); // tutti i match, sia sensati che non
  // cv::imshow("Matches", outMatches);

  // cv::namedWindow("Matches Inliers", cv::WINDOW_AUTOSIZE); // solo i match sensati
  // cv::imshow("Matches Inliers", outInliers);

  cv::waitKey();

  return 0;
}




