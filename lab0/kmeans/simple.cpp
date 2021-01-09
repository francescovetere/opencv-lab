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
	int cluster_count = 5;

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

			//////////////////////////////////////////

			//ad esempio...
//			std::cout<<"error :"<<error<<std::endl;
//			if(error<best_error)
//			{
//				best_error = error;
//				best_name = s;
//
//				best_image_cluster = new_image_compare.clone();
//			}
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
