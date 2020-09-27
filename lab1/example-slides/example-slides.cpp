//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList {
	std::string image_name;		    //!< image file name
	int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-i") {
			args.image_name = std::string(argv[++i]);
		}

		if(std::string(argv[i]) == "-t") {
			args.wait_t = atoi(argv[++i]);
		}
		else
			args.wait_t = 0;

		++i;
	}

	return true;
}

int main(int argc, char **argv)
{
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	std::cout<<"Simple program."<<std::endl;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	while(!exit_loop)
	{
		//generating file name
		//
		//multi frame case
		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //single frame case
			sprintf(frame_name,"%s",args.image_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		//Dichiariamo una variabile immagine in OpenCv
		cv::Mat M;
		
		//Leggiamo un’immagine da file
		//Attenzione! Di default OpenCV apre le immagini in formato RGB!
		//Se vogliamo aprire un’immagine gray scale, dobbiamo specificarlo:
		M = cv::imread(frame_name, CV_8UC1);
		if(M.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		} 
		
		//Stampiamo sul terminale l'immagine
		// std::cout << "M = "<< std::endl << " " << M << std::endl << std::endl;

		
		//////////////////////
		//processing code here

		/* Immagine come array semplice di unsigned char */
		// for(unsigned int i = 0; i < M.rows * M.cols * M.elemSize(); ++i)
		// 	M.data[i] = i;
		
		/* Immagine come array di pixel a 3 canali di 1 byte ciascuno, RGB ad esempio */
		// for(unsigned int i = 0; i<M.rows*M.cols*M.elemSize(); i+=M.elemSize())
		// {
		// 	M.data[i] = i; //B
		// 	M.data[i+M.elemSize1()] = i + 1; //G
		// 	M.data[i+M.elemSize1()+M.elemSize1()] = i + 2; //R
		// }
 
		/* Accesso riga/colonna per immagine a 3 canali di 1 byte ciascuno, RGB ad esempio */
		// for(int v =0;v<M.rows;++v)
		// {
		// 	for(int u=0;u<M.cols;++u)
		// 	{
		// 		M.data[ (u + v*M.cols)*3] = u; //B
		// 		M.data[ (u + v*M.cols)*3 + 1] = u+1; //G
		// 		M.data[ (u + v*M.cols)*3 + 2] = u+2; //R
		// 	}
		// }

		/* Accesso riga/colonna per immagine a multi-canale di 1 byte ciascuno 
		   (metodo generale)
		*/
		// for(int v =0;v<M.rows;++v)
		// {
		// 	for(int u=0;u<M.cols;++u)
		// 	{
		// 		for(int k=0;k<M.channels();++k)
		// 			M.data[ (u + v*M.cols)*M.channels() + k] = u + k;
		// 	}
		// }

		/* Accesso riga/colonna per immagine a multi-canale di 1 byte ciascuno 
		   (con at())
		*/
		// std::cout << M.channels() << std::endl;
		// for(int v =0;v<M.rows;++v)
		// {
		// 	for(int u=0;u<M.cols;++u)
		// 	{
		// 		for(int k=0;k<M.channels();++k)
		// 			M.at<cv::Vec3b>(v, u)[k-1] = u + k;
		// 	}
		// }

		/////////////////////

		//Creiamo una finestra che chiamiamo “test”
		cv::namedWindow("test", cv::WINDOW_NORMAL);

		//Visualizziamo nella finestra l'immagine
		// cv::imshow("test", cv::imread(M)); --- // imread() legge una string, ma noi gli stiamo passando un Mat!? 
		cv::imshow("test", M);

		//wait for key or timeout
		unsigned char key = cv::waitKey(args.wait_t);
		std::cout<<"key "<<int(key)<<std::endl;

		//here you can implement some looping logic using key value:
		// - pause
		// - stop
		// - step back
		// - step forward
		// - loop on the same frame

		if(key == 'q')
			exit_loop = true;

		frame_number++;
	}

	return 0;
}
