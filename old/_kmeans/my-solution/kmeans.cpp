//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

/////////////////////////////
/// Struttura per rappresentate un cluster
/// Da eventualmente modificare a piacere
///
struct cluster {
	int num_pixels=0;    //number of pixel in cluster, inizialmente facoltativo

	float ur = 0;       //center of mass R
	float ug = 0;       //center of mass G
	float ub = 0;       //center of mass B

	//costruttore
	cluster(int _num_pixels=0,float _ur=0, float _ug=0, float _ub=0):num_pixels(_num_pixels), ur(_ur), ug(_ug), ub(_ub) {}

	// comparison operator
	//
	// in questo caso li confrontiamo in base al numero di elementi
	// un altro criterio possible potrebbe essere la posizione del baricentro dei pixel
	//
	bool operator<(const cluster& a) {
		return num_pixels < a.num_pixels;
	}

};
/////////////////////////////



int main(int argc, char **argv) {
	//VOSTRO path in cui si trovano le immagini
	std::string path("../images/");

	//Immagine originale di partenza
	std::string original("pond_1.ppm");

	//Database di immagini globale
	std::vector<std::string> compare = {"beach_1.ppm", "beach_2.ppm", "beach_3.ppm", "beach_4.ppm", "boat_1.ppm", "boat_2.ppm", "boat_3.ppm", "stHelens_1.ppm", "stHelens_2.ppm", "stHelens_3.ppm", "crater_1.ppm", "crater_2.ppm", "crater_3.ppm", "pond_1.ppm", "pond_2.ppm", "pond_3.ppm", "sunset1_1.ppm", "sunset1_2.ppm", "sunset1_3.ppm", "sunset2_1.ppm", "sunset2_2.ppm", "sunset2_3.ppm"};

	//////////////////////
	//Variabili K-MEANS
	int cluster_count = 5;

	// cv::namedWindow("App", cv::WINDOW_NORMAL);
	// cv::createTrackbar("Cluster Count", "App", &cluster_count, 50, 0);
	/////////////////////



	//ESECUZIONE


	std::cout<<"Simple program."<<std::endl;
	//opening file
	std::cout<<"Opening original image"<<path+original<<std::endl;

	cv::Mat image = cv::imread(path+original, CV_32FC3);
	if(image.empty()) {
		std::cout<<"Unable to open "<<path+original<<std::endl;
		return 1;
	}

	while(1) {
		//lista cluster dell'immagine originale
		std::vector<cluster> cluster_list_original(cluster_count);
		//immagine dei cluster
		cv::Mat new_image( image.size(), image.type() );


		/////////////////////////////////////////////////////////////////
		/// FASE 1: K MEANS immagine originale
		//
		//
		//
		//OPENCV
		//
		//
		// YOUR CODE HERE
		//

		// Definisco la matrice dei samples di input, da passare a cv::kmeans
		// Avrò tante righe quanti sono i pixel di input, ma solo 3 colonne (r g b)
		// quindi ogni riga -> 1 pixel
		cv::Mat samples(image.rows * image.cols, 3, CV_32F);
		for(int r = 0; r < image.rows; ++r)
			for(int c = 0; c < image.cols; ++c)
				for(int z = 0; z < 3; ++z)
					samples.at<float>(r + c*image.rows, z) = image.at<cv::Vec3b>(r, c)[z];

		// Dichiaro la matrice che cv::kmeans riempirà: conterrà per ogni pixel un valore [0...cluster_count]
		// Ovvero, per ogni pixel mi dice a quale cluster esso appartiene
		cv::Mat labels;
		int attempts = 5;
		cv::Mat centers;

		cv::kmeans(samples, cluster_count, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 10000, 0.0001), attempts, cv::KmeansFlags::KMEANS_PP_CENTERS, centers);
		
		std::cout << "samples: " << samples.rows << "x" << samples.cols << std::endl; // 76800 x 3
		std::cout << "labels: " << labels.rows << "x" << labels.cols << std::endl; 	  // 76800 x 1
		std::cout << "centers: " << centers.rows << "x" << centers.cols << std::endl; // 5 x 3

		std::cout << "centers\n" << centers << std::endl;

		// Riempimento di new_image coi valori dei centri dei cluster
		for(int r = 0; r < image.rows; ++r)
			for(int c = 0; c < image.cols; ++c)
			{
				// Per ogni pixel, recupero la label che gli è stata associata
				int cluster_idx = labels.at<int>(r + c*image.rows, 0); // matrice labels ha 1 sola colonna

				// Recupero il corrispondente centro, e lo assegno all'attuale pixel di new_image
				// matrice centers ha 3 colonne: ciascuna, mi dà il valore r, g o b del centro i-esimo
				new_image.at<cv::Vec3b>(r,c)[0] = centers.at<float>(cluster_idx, 0);
				new_image.at<cv::Vec3b>(r,c)[1] = centers.at<float>(cluster_idx, 1);
				new_image.at<cv::Vec3b>(r,c)[2] = centers.at<float>(cluster_idx, 2);

				// Se e' la prima volta che incontro questo cluster, aggiorno i centri di massa
				if(cluster_list_original[cluster_idx].num_pixels == 0) {
					cluster_list_original[cluster_idx].ur = centers.at<float>(cluster_idx, 0);
					cluster_list_original[cluster_idx].ug = centers.at<float>(cluster_idx, 1);
					cluster_list_original[cluster_idx].ub = centers.at<float>(cluster_idx, 2);
				}

				// In ogni caso, aggiorno il numero di pixel appartenenti al cluster cluster_idx
				++cluster_list_original[cluster_idx].num_pixels;
			}

		///
		/// Ordinamento
		//
		//  Per migliorare le performance di associazione, un semplice espediente e' quello di
		//  ordinare i cluster per numero di pixel.
		//
		//  In questo modo andro' a confrontare cluster che, con maggiore probabilita', sono anche simili di dimensione
		//
		//  Si veda il documento pdf per dettagli
		std::sort(cluster_list_original.begin(), cluster_list_original.end());

		/////////////////////////////////////////////////////////////////



		///////////////////////////////////////////////////////////////
		/// FASE 2: confronto con immagini database
		//
		//
		float best_error=std::numeric_limits<float>::max();
		std::string best_name("undefined");
		cv::Mat best_image_cluster( image.size(), image.type() );

		//scorro le immagini di database
		for (const auto & s: compare)
		{
			//evito di confrontare con l'immagine originale
			if(s==original)
				continue;

			//apro l'immagine di database
			cv::Mat image_compare = cv::imread(path+s, CV_32FC3);
			if(image_compare.empty())
			{
				std::cout<<"Unable to open "<<path+s<<std::endl;
				return 1;
			}

			std::cout<<"compare with "<<s<<std::endl;

			//lista cluster dell'immagine di database
			std::vector<cluster> cluster_list_compare(cluster_count);
			//immagine dei cluster
			cv::Mat new_image_compare( image.size(), image.type() );

			///////////////////////////////////////////
			///
			/// YOUR CODE HERE
			///
			/// 1) kmeans per immagine di database s
			/// 2) confronto tra i cluster originali e quelli di s, calcolo error
			
			// Definisco la matrice dei samples di input, da passare a cv::kmeans
			// Avrò tante righe quanti sono i pixel di input, ma solo 3 colonne (r g b)
			// quindi ogni riga -> 1 pixel
			cv::Mat samples_compare(image_compare.rows * image_compare.cols, 3, CV_32F);
			for(int r = 0; r < image_compare.rows; ++r)
				for(int c = 0; c < image_compare.cols; ++c)
					for(int z = 0; z < 3; ++z)
						samples_compare.at<float>(r + c*image_compare.rows, z) = image_compare.at<cv::Vec3b>(r, c)[z];

			// Dichiaro la matrice che cv::kmeans riempirà: conterrà per ogni pixel un valore [0...cluster_count]
			// Ovvero, per ogni pixel mi dice a quale cluster esso appartiene
			cv::Mat labels_compare;
			int attempts = 5;
			cv::Mat centers_compare;

			cv::kmeans(samples_compare, cluster_count, labels_compare, cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 10000, 0.0001), attempts, cv::KmeansFlags::KMEANS_PP_CENTERS, centers_compare);
			
			// std::cout << "samples: " << samples.rows << "x" << samples.cols << std::endl; // 76800 x 3
			// std::cout << "labels: " << labels.rows << "x" << labels.cols << std::endl; 	  // 76800 x 1
			// std::cout << "centers: " << centers.rows << "x" << centers.cols << std::endl; // 5 x 3

			// std::cout << "centers\n" << centers << std::endl;

			// Riempimento di new_image coi valori dei centri dei cluster
			for(int r = 0; r < image_compare.rows; ++r)
				for(int c = 0; c < image_compare.cols; ++c)
				{
					// Per ogni pixel, recupero la label che gli è stata associata
					int cluster_idx = labels_compare.at<int>(r + c*image_compare.rows, 0); // matrice labels ha 1 sola colonna

					// Recupero il corrispondente centro, e lo assegno all'attuale pixel di new_image
					// matrice centers ha 3 colonne: ciascuna, mi dà il valore r, g o b del centro i-esimo
					new_image_compare.at<cv::Vec3b>(r,c)[0] = centers_compare.at<float>(cluster_idx, 0);
					new_image_compare.at<cv::Vec3b>(r,c)[1] = centers_compare.at<float>(cluster_idx, 1);
					new_image_compare.at<cv::Vec3b>(r,c)[2] = centers_compare.at<float>(cluster_idx, 2);

					// Se e' la prima volta che incontro questo cluster, aggiorno i centri di massa
					if(cluster_list_compare[cluster_idx].num_pixels == 0) {
						cluster_list_compare[cluster_idx].ur = centers_compare.at<float>(cluster_idx, 0);
						cluster_list_compare[cluster_idx].ug = centers_compare.at<float>(cluster_idx, 1);
						cluster_list_compare[cluster_idx].ub = centers_compare.at<float>(cluster_idx, 2);
					}

					// In ogni caso, aggiorno il numero di pixel appartenenti al cluster cluster_idx
					++cluster_list_compare[cluster_idx].num_pixels;
				}

			std::sort(cluster_list_compare.begin(), cluster_list_compare.end());

			//ad esempio...
			//			std::cout<<"error :"<<error<<std::endl;
			//			if(error<best_error)
			//			{
			//				best_error = error;
			//				best_name = s;
			//
			//				best_image_cluster = new_image_compare.clone();
			//			}



			///
			/// Confronto tra cluster
			//
			//  Per semplicta' facciamo un confronto 1-1 nell'ordine in cui si trovano nei rispettivi vettori di cluster
			//
			//  Come spiegato nel documento pdf, questa non e' la soluzione ottimale in generale, ma andrebbe fatto un confronto
			//  tutti-a-tutti tenendo la soluzione migliore per ogni immagine. Sarebbe stato troppo lungo per un esame di due ore.
			//
			float error=0.0f;
			for(int i=0; i < cluster_count; ++i) {
				cv::Point3f center_original(cluster_list_original[i].ur, cluster_list_original[i].ug, cluster_list_original[i].ub);
				cv::Point3f center_compare(cluster_list_compare[i].ur, cluster_list_compare[i].ug, cluster_list_compare[i].ub);

				error += cv::norm(center_original - center_compare);

				// Altro modo:
				// error+=fabs(cluster_list_compare[i].ur - cluster_list_original[i].ur);
				// error+=fabs(cluster_list_compare[i].ug - cluster_list_original[i].ug);
				// error+=fabs(cluster_list_compare[i].ub - cluster_list_original[i].ub);
			}
			std::cout<<"compare with "<<s<<":"<<error<<std::endl;
			if(error<best_error)
			{
				//aggiorno best error
				best_error = error;
				//mi salvo il nome dell'attuale best match
				best_name = s;

				//mi salvo l'attuale best match clusterizzato
				best_image_cluster = new_image_compare.clone();
			}
		}
		///////////////////////////////////////////////////////////////



		///////////////////////////////////////////////////////////////
		/// FASE 3: OUTPUT
		///
		//
		// immagine originale
		cv::namedWindow("image", cv::WINDOW_NORMAL);
		cv::imshow("image", image);

		cv::namedWindow("clustered image", cv::WINDOW_NORMAL);
		cv::imshow( "clustered image", new_image );

		// best match
		std::cout<<"best match "<<best_name<<std::endl;

		// cv::Mat image_best_match = cv::imread(path+best_name);
		// if(image_best_match.empty())
		// {
		// 	std::cout<<"Unable to open "<<path+best_name<<std::endl;
		// 	return 1;
		// }
		// cv::namedWindow("best match", cv::WINDOW_NORMAL);
		// cv::imshow("best match", image_best_match );

		cv::namedWindow("clustered best match", cv::WINDOW_NORMAL);
		cv::imshow( "clustered best match", best_image_cluster );
		///////////////////////////////////////////////////////////////////


		//wait for key
		cv::waitKey();
	}


	return 0;
}
