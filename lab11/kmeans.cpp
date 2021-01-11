//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

struct ArgumentList {
	std::string image;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 3;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <image>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i") {
			++i;
			args.image = std::string(argv[i]);
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
		int m_clusterCount = 1;
      	int m_centerMethod = 0;
  		
		cv::Mat src = cv::imread(args.image, CV_32FC1);
		cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
  		cv::imshow("image", src);
 
		cv::Mat samples(src.rows * src.cols, 3, CV_32F);
		for( int y = 0; y < src.rows; y++ )
			for( int x = 0; x < src.cols; x++ )
			for( int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<cv::Vec3b>(y,x)[z];

		cv::Mat labels;
		int attempts = 5;
		cv::Mat centers;
		// kmeans(samples, (m_clusterCount==0)?1:m_clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers );
		cv::kmeans(samples, (m_clusterCount==0)?1:m_clusterCount, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers );

		cv::Mat new_image( src.size(), src.type() );
		for( int y = 0; y < src.rows; y++ )
			for( int x = 0; x < src.cols; x++ )
			{ 
			int cluster_idx = labels.at<int>(y + x*src.rows,0);
			new_image.at<cv::Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<cv::Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<cv::Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
			}
		cv::imshow( "clustered image OpenCV", new_image );

		/*
		* K-Means grey scale
		*/
		cv::Mat grayscale(src.rows, src.cols, CV_8UC1);
		cv::cvtColor(src, grayscale, cv::COLOR_BGR2GRAY);

		cv::Mat clusters(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
		cv::Mat mapp(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
		int nk = m_clusterCount;
		std::vector<int> u(nk);
		std::vector<int> nu(nk);
		std::vector<int> nuc(nk);
		float error = 0;
		float oerror= 0;
		float error_ratio = 0.01;
		float error_th = 0.1;
		int max_iteration = 1000;
		int j = 0;

		switch (m_centerMethod) {
		case 0:
			//centri equidistribuiti
			for(int k=0;k<nk;++k)
			{
				u[k] = ((1+k)*255)/nk;
			}
			break;

		case 1:
			//centri random
			srand (time(NULL));
			std::cout<<"centri random: ";
			for(int k=0;k<nk;++k)
			{
				u[k] = rand()%255;
				//verifichiamo che sia unico
				for(int kk=0;kk<k;++kk)
				{
					if(u[k] == u[kk])
					{
						u[k] = rand()%nk;
						kk=0;
						continue;
					}
				}
				std::cout<<u[k]<<" ";
			}
			std::cout<<std::endl;
			break;

		case 2:
			//istogramma
			std::vector<unsigned int> istrogramma(255,0);
			for(int v = 0; v < src.rows; v+=1)
			{
				for(int u = 0; u < src.cols; u+=1)
				{
					istrogramma[(*(grayscale.data + (v)* src.cols+ u))]++;
				}
			}
			//ricerca massimi locali
			u.clear();
			nk=0;
			for(int i = 0; i < 255; ++i)
			{
				bool max = true;
				for(int u = -8; u <= 8; u+=1)
				{
					if(i+u>=0 && i+u<255 && u!=0)
					{
						if(istrogramma[i] <= istrogramma[i+u])
						{
							max = false;
							break;
						}
					}
				}
				if(max)
				{
					u.push_back(i);
					nk++;
				}
			}
			std::cout<<"centri dai massimi istogramma ("<<nk<<"): ";
			for(std::vector<int>::iterator p=u.begin();p!=u.end();++p)
				std::cout<<*p<<" ";
			std::cout<<std::endl;

			if(nk == 0)
			{
				nk++;
				u.push_back(rand()%255);
				std::cout<<"Non ci sono massimi, uso un centro massimo random "<<u.front()<<std::endl;
			}
			nu = std::vector<int>(nk);
			nuc = std::vector<int>(nk);
			break;
		}

		while(1)
		{
			for(int k=0;k<nk;++k)
			{
				nu[k]=nuc[k]=0;
			}
			for(int i = 0; i<grayscale.rows*grayscale.cols;++i )
			{
				std::vector<int> distance(nk);
				for(int k=0;k<nk;++k)
				{
					distance[k] = abs(u[k] - *(grayscale.data + i));
				}

				std::vector<int>::iterator mine = std::min_element(distance.begin(), distance.end());

				*(mapp.data + i) = mine - distance.begin();
				nu[*(mapp.data + i)]+=*(grayscale.data + i);
				nuc[*(mapp.data + i)]++;
			}

			for(int k=0;k<nk;++k)
			{
				nu[k]/=nuc[k]>0 ? nuc[k] : 1;
			}

			std::swap(u,nu);

			oerror = error;
			error = 0;
			for(int i = 0; i<grayscale.rows*grayscale.cols;++i )
			{
				error+=(u[*(mapp.data + i)] - int(*(grayscale.data + i)))*(u[*(mapp.data + i)] - int(*(grayscale.data + i)));
			}
			error/=grayscale.rows*grayscale.cols;

			if( (j>0 && oerror>error && (oerror-error)/oerror < error_ratio) || j==max_iteration || (j>0 && error<error_th))
				break;

			++j;
		}
		std::cout<<"Esco all'iterazione "<<j<<" con errore "<<error<<" "<<(oerror-error)/oerror<<std::endl;

			for(int i = 0; i<grayscale.rows*grayscale.cols;++i )
			{
				*(clusters.data + i) = u[*(mapp.data + i)];
			}

		cv::namedWindow("Clusters Grey",cv::WINDOW_AUTOSIZE);
		cv::imshow("Clusters Grey", clusters);
  

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
