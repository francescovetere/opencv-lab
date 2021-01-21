//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
// eigen
// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Dense>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> 	// std::find, std::min_element
#include <numeric> 		// std::accumulate
#include <cmath> 		// std::abs
#include <cstdlib>   	// srand, rand

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


void myHarrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh, std::string img_name) {
  // Calcolo le 2 derivate lungo x e lungo y
  cv::Mat Ix, Iy;

  // Prima di effettuare la derivata, eseguo un blur gaussiano per migliorarne l'effetto
  cv::Mat blurred_image;
  cv::GaussianBlur(image, blurred_image, cv::Size(3, 3), 1.0, 1.0);

  // Effettuo la derivata vera e propria
  // sobel(blurred_image, Ix, Iy);

  // Altrimenti, con la funzione built-in di OpenCV:
  cv::Sobel(blurred_image, Ix, CV_32FC1, 1, 0);
  cv::Sobel(blurred_image, Iy, CV_32FC1, 0, 1);

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
				if(response.at<float>(r,c) > harrisTh && is_local_maximum(response, r, c, 3))
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


struct ArgumentList {
	std::string input_img_1;
	std::string input_img_2;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 3;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <input_img_1> <input_img_2>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i") {
			++i;
			args.input_img_1 = std::string(argv[i]);
			args.input_img_2 = std::string(argv[i+1]);
		}

		++i;
	}

	return true;
}

int main(int argc, char **argv) {
	int frame_number = 0;
	bool exit_loop = false;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop) {
		cv::Mat input_img_1 = cv::imread(args.input_img_1, CV_8UC1);
		if(input_img_1.empty()) {
			std::cout << "Error loading input_img_1: " << argv[2] << std::endl;
    		return 1;
		}

		cv::Mat input_img_2 = cv::imread(args.input_img_2, CV_8UC1);
		if(input_img_1.empty()) {
			std::cout << "Error loading input_img_2: " << argv[3] << std::endl;
    		return 1;
		}
		//////////////////////
			//processing code here
					////////////////////////////////////////////////////////
		/// HARRIS CORNER
		//
		float alpha = 0.04;
		float harrisTh = 500000*200;    //da impostare in base alla propria implementazione!!!!!

		std::vector<cv::KeyPoint> keypoints0, keypoints1;

		// FASE 1

		std::cout << "Computing harris corners..." << std::endl;

		// (Aggiungo un parametro finale per stampare a video il nome dell'immagine nei risultati temporanei)
		myHarrisCornerDetector(input_img_1, keypoints0, alpha, harrisTh, "input");
		myHarrisCornerDetector(input_img_2, keypoints1, alpha, harrisTh, "cover");

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
		extractor->compute(input_img_1, keypoints0, descriptors0);
		extractor->compute(input_img_2, keypoints1, descriptors1);

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
			myFindHomographyRansac(points[0], points[1], matchesDraw, N, epsilon, sample_size, H, matchesInliersBest);
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

				/*
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
			*/
			}
			
			else {
			std::cout<<"Pochi inliers! "<<matchesInliersBest.size()<<"/"<<matchesDraw.size()<<std::endl;
			}


		}

		else {
			std::cout<<"Pochi match! "<<points[0].size()<<"/"<<keypoints0.size()<<std::endl;
		}
		////////////////////////////////////////////////////////
		cv::Mat stitched;

		cv::Mat padding = cv::Mat::zeros(cv::Size(input_img_1.cols/2 - 100, input_img_1.rows), CV_8UC1);
		cv::hconcat(input_img_1, padding, stitched);
		// cv::Mat stitched = cv::Mat::zeros(cv::Size(input_img_1.cols + 300, input_img_1.rows + 100), CV_8UC1);
		// input_img_1.copyTo(stitched(cv::Rect(100, 0, input_img_1.cols, input_img_1.rows)));
		
		// cv::Mat warped;
		
		cv::Mat tmp;
		cv::warpPerspective(input_img_2, tmp, H.inv(), stitched.size());

		// input_img_1.copyTo(stitched(cv::Rect(0, 0, input_img_1.cols, input_img_1.rows)));
		overlap_images(stitched.clone(), tmp, stitched);
		// H = H.inv();
		// for(int r = 0; r < input_img_2.rows; ++r) {
		// 	for(int c = 0; c < input_img_2.cols; ++c) {
		// 	// Calcolo la destinazione finale di ciascun punto della nuova cover, grazie ad H
		// 	cv::Mat curr_point(3, 1, CV_64FC1);
		// 	curr_point.at<double>(0, 0) = c;
		// 	curr_point.at<double>(1, 0) = r;
		// 	curr_point.at<double>(2, 0) = 1;

		// 	cv::Mat transformed_point = H*curr_point;
			
		// 	double x = transformed_point.at<double>(0, 0) / transformed_point.at<double>(2, 0);
		// 	double y = transformed_point.at<double>(1, 0) / transformed_point.at<double>(2, 0);

		// 		if(x >= 0 && x <= stitched.cols - 1 && y >= 0 && y <= stitched.rows - 1) {
		// 				stitched.at<uint8_t>(y, x) = input_img_2.at<uint8_t>(r, c);
		// 		}
		// 	}
		// }
		// /////////////////////

		//display images
		cv::namedWindow("input_img_1", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img_1", input_img_1);

		cv::namedWindow("input_img_2", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img_2", input_img_2);

		// cv::namedWindow("warped", cv::WINDOW_AUTOSIZE);
		// cv::imshow("warped", warped);

		cv::namedWindow("stitched", cv::WINDOW_AUTOSIZE);
		cv::imshow("stitched", stitched);		

		//wait for key or timeout
		unsigned char key = cv::waitKey(0);
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
