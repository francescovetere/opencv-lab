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
typedef struct _cluster
{
	int num_pixels=0;    //number of pixel in cluster, inizialmente facoltativo

	float ur = 0;       //center of mass R
	float ug = 0;       //center of mass G
	float ub = 0;       //center of mass B

	//costruttore
	_cluster(int _num_pixels=0,float _ur=0, float _ug=0, float _ub=0):num_pixels(_num_pixels), ur(_ur), ug(_ug), ub(_ub) {}

	// comparison operator
	//
	// in questo caso li confrontiamo in base al numero di elementi
	// un altro criterio possible potrebbe essere la posizione del baricentro dei pixel
	//
	bool operator<(const _cluster & a)
	{
		return num_pixels < a.num_pixels;
	}

}cluster;
/////////////////////////////


int main(int argc, char **argv)
{
	//VOSTRO path in cui si trovano le immagini
	std::string path("/home/scattani/workspace/visione/pratico_20190619/images/");

	//Immagine originale di partenza
	std::string original("pond_1.ppm");

	//Database di immagini globale
	std::vector<std::string> compare = {"beach_1.ppm", "beach_2.ppm", "beach_3.ppm", "beach_4.ppm", "boat_1.ppm", "boat_2.ppm", "boat_3.ppm", "stHelens_1.ppm", "stHelens_2.ppm", "stHelens_3.ppm", "crater_1.ppm", "crater_2.ppm", "crater_3.ppm", "pond_1.ppm", "pond_2.ppm", "pond_3.ppm", "sunset1_1.ppm", "sunset1_2.ppm", "sunset1_3.ppm", "sunset2_1.ppm", "sunset2_2.ppm", "sunset2_3.ppm"};

	//////////////////////
	//Variabili K-MENAS
	int cluster_count = 4;

	cv::namedWindow("App",CV_WINDOW_NORMAL);
	cv::createTrackbar("Cluster Count", "App", &cluster_count, 50, 0);
	/////////////////////



	//ESECUZIONE


	std::cout<<"Simple program."<<std::endl;
	//opening file
	std::cout<<"Opening original image"<<path+original<<std::endl;

	cv::Mat image = cv::imread(path+original);
	if(image.empty())
	{
		std::cout<<"Unable to open "<<path+original<<std::endl;
		return 1;
	}

	while(1)
	{
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
		cv::Mat samples(image.rows * image.cols, 3, CV_32F);
		for( int y = 0; y < image.rows; y++ )
			for( int x = 0; x < image.cols; x++ )
				for( int z = 0; z < 3; z++)
					samples.at<float>(y + x*image.rows, z) = image.at<cv::Vec3b>(y,x)[z];

		cv::Mat labels;
		int attempts = 5;
		cv::Mat centers;
		//
		//
		cv::kmeans(samples, (cluster_count==0)?1:cluster_count, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, cv::KMEANS_RANDOM_CENTERS, centers );
		//
		//
		for( int y = 0; y < image.rows; y++ )
			for( int x = 0; x < image.cols; x++ )
			{
				int cluster_idx = labels.at<int>(y + x*image.rows,0);
				new_image.at<cv::Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
				new_image.at<cv::Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
				new_image.at<cv::Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);

				//aggiorniamo il numero di pixel appartenenti al cluster cluster_idx
				//se e' la prima volta che incontro questo cluster, aggiorno anche i centri di massa
				if(cluster_list_original[cluster_idx].num_pixels++ == 0)
				{
					cluster_list_original[cluster_idx].ur = centers.at<float>(cluster_idx, 0);
					cluster_list_original[cluster_idx].ug = centers.at<float>(cluster_idx, 1);
					cluster_list_original[cluster_idx].ub = centers.at<float>(cluster_idx, 2);
				}
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
			cv::Mat image_compare = cv::imread(path+s);
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

			//OPENCV
			cv::Mat samples_compare(image_compare.rows * image_compare.cols, 3, CV_32F);
			for( int y = 0; y < image_compare.rows; y++ )
				for( int x = 0; x < image_compare.cols; x++ )
					for( int z = 0; z < 3; z++)
						samples_compare.at<float>(y + x*image_compare.rows, z) = image_compare.at<cv::Vec3b>(y,x)[z];

			cv::Mat labels_compare;
			int attempts = 5;
			cv::Mat centers_compare;
			//
			//
			cv::kmeans(samples_compare, (cluster_count==0)?1:cluster_count, labels_compare, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, cv::KMEANS_RANDOM_CENTERS, centers_compare );
			//
			//

			for( int y = 0; y < image.rows; y++ )
				for( int x = 0; x < image.cols; x++ )
				{
					int cluster_idx = labels_compare.at<int>(y + x*image.rows,0);
					new_image_compare.at<cv::Vec3b>(y,x)[0] = centers_compare.at<float>(cluster_idx, 0);
					new_image_compare.at<cv::Vec3b>(y,x)[1] = centers_compare.at<float>(cluster_idx, 1);
					new_image_compare.at<cv::Vec3b>(y,x)[2] = centers_compare.at<float>(cluster_idx, 2);

					//aggiorniamo il numero di pixel appartenenti al cluster cluster_idx
					//se e' la prima volta che incontro questo cluster, aggiorno anche i centri di massa
					if(cluster_list_compare[cluster_idx].num_pixels++ == 0)
					{
						cluster_list_compare[cluster_idx].ur = centers_compare.at<float>(cluster_idx, 0);
						cluster_list_compare[cluster_idx].ug = centers_compare.at<float>(cluster_idx, 1);
						cluster_list_compare[cluster_idx].ub = centers_compare.at<float>(cluster_idx, 2);
					}
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
			std::sort(cluster_list_compare.begin(), cluster_list_compare.end());


			///
			/// Confronto tra cluster
			//
			//  Per semplicta' facciamo un confronto 1-1 nell'ordine in cui si trovano nei rispettivi vettori di cluster
			//
			//  Come spiegato nel documento pdf, questa non e' la soluzione ottimale in generale, ma andrebbe fatto un confronto
			//  tutti-a-tutti tenendo la soluzione migliore per ogni immagine. Sarebbe stato troppo lungo per un esame di due ore.
			//
			float error=0;
			for(int i=0;i<cluster_count;++i)
			{
				error+=fabs(cluster_list_compare[i].ur - cluster_list_original[i].ur);
				error+=fabs(cluster_list_compare[i].ug - cluster_list_original[i].ug);
				error+=fabs(cluster_list_compare[i].ub - cluster_list_original[i].ub);
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
		//////////////////////////////////////////


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

		cv::Mat image_best_match = cv::imread(path+best_name);
		if(image_best_match.empty())
		{
			std::cout<<"Unable to open "<<path+best_name<<std::endl;
			return 1;
		}
		cv::namedWindow("best match", cv::WINDOW_NORMAL);
		cv::imshow("best match", image_best_match );

		cv::namedWindow("clustered best match", cv::WINDOW_NORMAL);
		cv::imshow( "clustered best match", best_image_cluster );
		///////////////////////////////////////////////////////////////////


		//wait for key
		cv::waitKey();
	}


	return 0;
}
