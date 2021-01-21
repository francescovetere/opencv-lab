/**
 * Francesco Vetere
 * Matricola 313336
 * Esempio di esecuzione: ./skel -l ../images/left.png -r ../images/right.png
 */
//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <stdint.h>

#define MAX_DISPARITY (128)

/****************************/
/*** FUNZIONI DI UTILITA' ***/
/****************************/

/**
 * Date due coordinate r e c non discrete, calcolo un valore corrispondente ad esse
 * interpolando il valore reale dei 4 vicini discreti, moltiplicati con opportuni pesi
 */
template <class T>
T bilinear(const cv::Mat& image, float r, float c) {

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
 * Date le immagini stereo L e R, la window e la massima disparità, calcola l'immagine disparità
 */
void SAD_Disparity(const cv::Mat& L, const cv::Mat& R, unsigned short w_size, int max_range, cv::Mat& out) {
	out = cv::Mat::zeros(L.rows, L.cols, CV_8UC1);

	int current_SAD; // SAD corrente calcolata
	int best_SAD;    // la migliore SAD calcolata fin' ora per (row_L, col_L)
	int best_offset; // la migliore disparità calcolata fin' ora per (row_L, col_L)

	for(int row_L = 0; row_L < L.rows - w_size; ++row_L) { 		// tengo conto della window
		for(int col_L = 0; col_L < L.cols - w_size; ++col_L) {  // tengo conto della window e dell'offset massimo
			best_SAD = std::numeric_limits<int>::max();
			best_offset = 0;

			for(int offset = 0; offset < max_range; ++offset) {
				if(col_L - offset - w_size >= 0) { // verifico di non uscire dall'immagine destra
					current_SAD = 0;

					for(int row_window = 0; row_window < w_size; ++row_window) {
						for(int col_window = 0; col_window < w_size; ++col_window) {
								current_SAD += std::abs(
									(int)L.at<uint8_t>(row_L + row_window, col_L + col_window) -
									// faccio la ricerca sulla stessa riga!
									(int)R.at<uint8_t>(row_L + row_window, col_L - offset + col_window)
								);
						}
					}

					if(current_SAD < best_SAD) {
						best_SAD = current_SAD; // aggiorno l'attuale valore migliore di disparità
						best_offset = offset;
					}
				}
			}

			// Terminati i confronti a tutti gli offset possibili, ho in best_offset il valore da assegnare a out(row_L, col_L)
			out.at<uint8_t>(row_L, col_L) = best_offset;
		}
	}
}

struct ArgumentList {
  std::string left_image;
  std::string right_image;
  unsigned short max_d;
  unsigned short w;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  int c;
  args.max_d=MAX_DISPARITY;
  args.w=7;

  while ((c = getopt (argc, argv, "hl:r:d:w:")) != -1)
    switch (c)
    {
      case 'l':
	args.left_image = optarg;
	break;
      case 'r':
	args.right_image = optarg;
	break;
      case 'w':
	args.w = atoi(optarg);
	if(!(args.w%2) || !args.w)
	{
	  std::cerr << "Disparity window size must be an odd number>0" << std::endl;
	  exit(EXIT_FAILURE);
	}
	break;
      case 'd':
	args.max_d = atoi(optarg);
	break;
      case 'h':
      default:
	std::cout<<"usage: " << argv[0] << " -i <image_name>"<<std::endl;
	std::cout<<"exit:  type q"<<std::endl<<std::endl;
	std::cout<<"Allowed options:"<<std::endl<<
	  "   -h                       produce help message"<<std::endl<<
	  "   -r arg                   right image name"<<std::endl<<
	  "   -l arg                   left image name"<<std::endl<<
	  "   -w arg                   disparity window size [default = 7]"<<std::endl<<
	  "   -d arg                   max disparity [default = " << MAX_DISPARITY << "]"<<std::endl<<std::endl<<std::endl;
	return false;
    }
  return true;
}


/******************************/
/*** FUNZIONI DELLA TRACCIA ***/
/******************************/
void mysample(const cv::Mat& in, const float factor, cv::Mat& out) {  
    out.create(in.rows/factor, in.cols/factor, in.type());

    for(int r = 0; r < out.rows; ++r) {	
			for(int c = 0; c < out.cols; ++c) {
          // Ho due coordinate float: faccio un'interpolazione bilineare
          int val = bilinear<uint8_t>(in, r*factor, c*factor);
          out.at<uint8_t>(r, c) = val;
			}
		}
}

void mycrop(const cv::Mat& in, const cv::Rect ROI, cv::Mat& out) {

  	unsigned int top_left_row = ROI.x;
		unsigned int top_left_col = ROI.y;
		unsigned int width = ROI.width;
		unsigned int height = ROI.height;

		out.create(height - top_left_row, width - top_left_col, in.type());
		// std::cout << out.rows << " " << out.cols << std::endl;

		/* Accesso riga/colonna con data */
		for(int v = 0; v < out.rows; ++v) {
			for(int u = 0;u < out.cols; ++u) {
					out.data[(v*out.cols + u)] 
					= in.data[(((top_left_row + v)*in.cols) + top_left_col + u)];
			}
		}
}

/****************************/
/********** MAIN ************/
/****************************/
int main(int argc, char **argv) {
  //////////////////////
  //parse argument list:
  //////////////////////
  ArgumentList args;
  if(!ParseInputs(args, argc, argv)) {
    return 1;
  }

  if(args.right_image.empty())
  {
    std::cerr << "Missing right filename,  please use -r option" << std::endl;
    return 1;
  }
  if(args.left_image.empty()) 
  {
    std::cerr << "Missing left filename,  please use -l option" << std::endl;
    return 1;
  }

  //opening file
  std::cout<<"Opening left image" << args.left_image <<std::endl;
  cv::Mat left = cv::imread(args.left_image.c_str(), CV_8UC1);
  if(left.empty())
  {
    std::cerr << "Unable to open " << args.left_image << std::endl;
    return 1;
  }

  //opening file
  std::cout<<"Opening right image" << args.right_image <<std::endl;
  cv::Mat right = cv::imread(args.right_image.c_str(), CV_8UC1);
  if(right.empty())
  {
    std::cerr << "Unable to open " << args.right_image << std::endl;
    return 1;
  }

  if(right.size() != left.size())
  {
    std::cerr << "The left and right images have different sizes" << std::endl;
    return 1;
  }

  /**********/
  /** ES 1 **/
  /**********/

  float factor = std::sqrt(2);

  // down_1 = primo downsample (fattore sqrt(2))
  // down_2 = secondo downsample (fattore 2)
  cv::Mat left_down1, right_down1;
  cv::Mat left_down2, right_down2;

  mysample(left, factor, left_down1);
  mysample(right, factor, right_down1);
  mysample(left, factor*factor, left_down2);
  mysample(right, factor*factor, right_down2);
  
  // DEBUG
  std::cout << "left: " << left.size() << std::endl;
  std::cout << "left_down1: " << left_down1.size() << std::endl;
  std::cout << "left_down2: " << left_down2.size() << std::endl;
  std::cout << "right: " << right.size() << std::endl;
  std::cout << "right_down1: " << right_down1.size() << std::endl;
  std::cout << "right_down2: " << right_down2.size() << std::endl;

  cv::namedWindow("right", cv::WINDOW_AUTOSIZE);
  cv::imshow("right", right);

  cv::namedWindow("left", cv::WINDOW_AUTOSIZE);
  cv::imshow("left", left);

  cv::namedWindow("left_down1", cv::WINDOW_AUTOSIZE);
  cv::imshow("left_down1", left_down1);
 
  cv::namedWindow("right_down1", cv::WINDOW_AUTOSIZE);
  cv::imshow("right_down1", right_down1);
  
  cv::namedWindow("left_down2", cv::WINDOW_AUTOSIZE);
  cv::imshow("left_down2", left_down2);
  
  cv::namedWindow("right_down2", cv::WINDOW_AUTOSIZE);
  cv::imshow("right_down2", right_down2);

  cv::waitKey(0);

  cv::destroyWindow("left");
  cv::destroyWindow("right");
  cv::destroyWindow("left_down1");
  cv::destroyWindow("right_down1");
  cv::destroyWindow("left_down2");
  cv::destroyWindow("right_down2");

  /**********/
  /** ES 2 **/
  /**********/
  // Definisco le regioni di crop in modo opportuno
  cv::Rect ROI_L1(left.rows/2 - left_down2.rows/2, left.cols/2 - left_down2.cols/2, left.cols/2 - left_down2.cols/2 + left_down2.cols, left.rows/2 - left_down2.rows/2 + left_down2.rows);
  cv::Rect ROI_R1(right.rows/2 - right_down2.rows/2, right.cols/2 - right_down2.cols/2, right.cols/2 - right_down2.cols/2 + right_down2.cols, right.rows/2 - right_down2.rows/2 + right_down2.rows);

  cv::Rect ROI_L2(left_down1.rows/2 - left_down2.rows/2, left_down1.cols/2 - left_down2.cols/2, left_down1.cols/2 - left_down2.cols/2 + left_down2.cols, left_down1.rows/2 - left_down2.rows/2 + left_down2.rows);
  cv::Rect ROI_R2(right_down1.rows/2 - right_down2.rows/2, right_down1.cols/2 - right_down2.cols/2, right_down1.cols/2 - right_down2.cols/2 + right_down2.cols, right_down1.rows/2 - right_down2.rows/2 + right_down2.rows);

  cv::Mat left_cropped, right_cropped;
  cv::Mat left_down1_cropped, right_down1_cropped;

  // Creo le nuove coppie di immagini cropped 
  mycrop(left, ROI_L1, left_cropped);
  mycrop(right, ROI_R1, right_cropped);

  mycrop(left_down1, ROI_L2, left_down1_cropped);
  mycrop(right_down1, ROI_R2, right_down1_cropped);

  cv::namedWindow("left_cropped", cv::WINDOW_AUTOSIZE);
  cv::imshow("left_cropped", left_cropped);
  
  cv::namedWindow("right_cropped", cv::WINDOW_AUTOSIZE);
  cv::imshow("right_cropped", right_cropped);
  
  cv::namedWindow("left_down1_cropped", cv::WINDOW_AUTOSIZE);
  cv::imshow("left_down1_cropped", left_down1_cropped);

  cv::namedWindow("right_down1_cropped", cv::WINDOW_AUTOSIZE);
  cv::imshow("right_down1_cropped", right_down1_cropped);

  // DEBUG
  std::cout << "\nleft_cropped: " << left_cropped.size() << std::endl;
  std::cout << "right_cropped: " << right_cropped.size() << std::endl;
  std::cout << "left_down1_cropped: " << left_down1_cropped.size() << std::endl;
  std::cout << "right_down1_cropped: " << right_down1_cropped.size() << std::endl;

  cv::waitKey(0);
  cv::destroyWindow("left_cropped");
  cv::destroyWindow("right_cropped");
  cv::destroyWindow("left_down1_cropped");
  cv::destroyWindow("right_down1_cropped");

  /**********/
  /** ES 3 **/
  /**********/
  cv::Mat SAD_original;
  cv::Mat SAD_down1;
  cv::Mat SAD_down2;

  unsigned short w_size = args.w;
  // Nota: per velocizzare l'esecuzione, pongo comunque la w_size a 2
  w_size = 2;

  int max_range = args.max_d;

  std::cout << "Computing SAD...\n\n" << std::endl;
  SAD_Disparity(left_cropped, right_cropped, w_size, max_range, SAD_original);
  SAD_Disparity(left_down1_cropped, right_down1_cropped, w_size, max_range, SAD_down1);
  SAD_Disparity(left_down2, right_down2, w_size, max_range, SAD_down2);

  cv::namedWindow("SAD_original", cv::WINDOW_AUTOSIZE);
  cv::imshow("SAD_original", SAD_original);
  
  cv::namedWindow("SAD_down1", cv::WINDOW_AUTOSIZE);
  cv::imshow("SAD_down1", SAD_down1);

  cv::namedWindow("SAD_down2", cv::WINDOW_AUTOSIZE);
  cv::imshow("SAD_down2", SAD_down2);
  
  cv::waitKey(0);
  cv::destroyWindow("SAD_original");
  cv::destroyWindow("SAD_down1");
  cv::destroyWindow("SAD_down2");

  /**********/
  /** ES 4 **/
  /**********/
  std::cout << "Merging disparities...\n\n" << std::endl;
  
  // Faccio degli upsample sulle immagini su cui ho fatto un downsample
  cv::Mat tmp1;
  mysample(SAD_down1, 1/factor, tmp1);
  SAD_down1 = tmp1.clone();

  cv::Mat tmp2;
  mysample(SAD_down2, 1/(factor*factor), tmp2);
  SAD_down2 = tmp2.clone();

  // La SAD finale avrà dimensioni pari a quelle dell'input iniziale
  cv::Mat merged_SAD(left.rows, left.cols, left.type());

  // Leggo un intero che codifica la scelta tra la media o la disparità del sottocampionamento maggiore
  int scelta;
  std::cout << "Inserire scelta per il merge:\n1) Media\n2) Sottocampionamento maggiore" << std::endl;
  std::cin >> scelta;

  // Riempo l'immagine merged_SAD in base alla scelta fatta dall'utente
  for(int r = 0; r < merged_SAD.rows; ++r) {
    for(int c = 0; c < merged_SAD.cols; ++c) {
      int val = 0;

      // MEDIA
      if(scelta == 1) {
        // count mi dice in quante immagini ricade il pixel
        // servirà per effettuare la media aritmetica

        int count = 0;
        
        // guardo se pixel cade in SAD_original
        if(r > r+ROI_L1.x && r < ROI_L1.height && c > c+ROI_L1.y && c < ROI_L1.width) {
          val += SAD_original.at<uint8_t>(r, c);
          ++count;
        }
        
        // guardo se pixel cade in SAD_down1
        else if (r > r+ROI_L2.x && r < ROI_L2.height && c > c+ROI_L2.y && c < ROI_L2.width) {
          val += SAD_down1.at<uint8_t>(r, c);
          ++count;
        }
        
        // altrimenti, cadrà in SAD_down2
        else {
          val += SAD_down2.at<uint8_t>(r, c);
          ++count;
        }

        // std::cout << "val: " << val << std::endl;

        val = val/count;
      }

      // SOTTOCAMPIONAMENTO MAGGIORE
      else if(scelta == 2) {
        // guardo se pixel cade in SAD_original
        if(r > r+ROI_L1.x && r < ROI_L1.height && c > c+ROI_L1.y && c < ROI_L1.width) 
          val = SAD_original.at<uint8_t>(r, c);
        // guardo se pixel cade in SAD_down1
        else if (r > r+ROI_L2.x && r < ROI_L2.height && c > c+ROI_L2.y && c < ROI_L2.width) 
          val = SAD_down1.at<uint8_t>(r, c);
        // altrimenti, cadrà in SAD_down2
        else val = SAD_down2.at<uint8_t>(r, c);
      }

      // Ora inserisco il valore calcolato in merged_SAD
      merged_SAD.at<uint8_t>(r, c) = val;
    }
  }

  cv::namedWindow("merged_SAD", cv::WINDOW_AUTOSIZE);
  cv::imshow("merged_SAD", merged_SAD);

  cv::waitKey(0);
  cv::destroyWindow("merged_SAD");

  return 0;
}

