//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

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

/* Funzione di convoluzione */
void convInt(const Mat& image, const Mat& kernel, Mat& output)
{
		/* controllo se il kernel che abbiamo inserito è simmetrico */
		if(kernel.rows % 2 == 0 && kernel.cols == 0 )
		{
			  cout << "Il kernel non è dispari e quindi non simmetrico" << endl;
				//return -1;
		}

		/* Parametri per constant stretching */
		float max = -1000.0f;
		float min = 1000.0f;

		int padding_riga;
		int padding_colonna;
		int pad_r;
		int pad_c;

		if (kernel.rows > kernel.cols)
		 {
		    	padding_riga = kernel.rows - 1;
		    	padding_colonna = 0;
		 }
		 else if (kernel.cols > kernel.rows)
		 {
		    	padding_colonna = kernel.cols - 1;
		    	padding_riga = 0;
		 }
		 else
		 {
		    	padding_colonna = kernel.cols - 1;
		    	padding_riga = kernel.rows - 1;
		 }

		/* padding immagine */
		pad_r= floor(padding_riga / 2);
		pad_c = floor(padding_colonna / 2);


		Mat pad_image;
		pad_image = Mat::zeros(image.rows + 2*pad_r,image.cols+2*pad_c , CV_8UC1);

		for(int r = pad_r; r < pad_image.rows - pad_r; r++)
		{
			for(int c=pad_c; c < pad_image.cols -pad_c ; c++)
			{
					pad_image.at<uchar>(r,c)= image.at<uchar>(r-pad_r,c-pad_c);
			}
		}

		//cout << pad_image << endl;
		Mat conversione;
		conversione=Mat(image.rows, image.cols, CV_32FC1); //la matrice di output ha le stesse dimensioni dell'image di input


		int somma;
		for(int i=pad_r; i < pad_image.rows - pad_r; i++ )
		{
			for(int j=pad_c; j < pad_image.cols - pad_c; j++)
			{
				 somma = 0;

				 //adesso inserisco i cicli per scorrere sul kernel
				 for(int l=0; l < kernel.rows; l ++)
				 {
					 for(int k=0; k < kernel.cols; k++)
					 {
						   somma += (float) (pad_image.at<uchar> (i-pad_r+l, j-pad_c+k) * kernel.at<float>(l,k));
					 }
				 }

				/*aggiorno max e min per poter applicare il constant stretching */

				if (somma > max)
 	 				max = somma;
 	 			if(somma < min)
 	 				min = somma;

				 conversione.at <float> (i-pad_r,j-pad_c) = somma;

			}
		}
		output = Mat(image.rows, image.cols, CV_8UC1);

		/*riscalo i valori */
		conversione -= min;
	  conversione = conversione * (255.0/(max-min));
		conversione.convertTo(output, CV_8UC1);
}



int convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out)
{
    out = cv::Mat(image.rows, image.cols, CV_32FC1, cv::Scalar(0));

    int h_kh = (int) std::floor(kernel.rows/ 2);
    int h_kw = (int) std::floor(kernel.cols/ 2);

    for (int r = h_kh; r < out.rows - h_kh; r++)
    {
        for (int c = h_kw; c < out.cols - h_kw; c++)
        {
            for (int rr = -h_kh; rr <= h_kh; rr++)
            {
                for (int cc = -h_kw; cc <= h_kw; cc++)
                {
                    *((float*) &out.data[(c + r * out.cols) * out.elemSize()]) +=
                            image.data[(c + cc + (r + rr) * image.cols) * image.elemSize()] *
                            *((float*) &kernel.data[(cc + h_kw + (rr + h_kh) * kernel.cols) * kernel.elemSize()]);

                }
            }
        }
    }

    return 0;
}


///////////////////////////////////////////
// PROCESSING CODE HERE
// Scrivere un programma C/C++ che crei una nuova immagine contenente
// unicamente il lato corrispondente all’ultimio numero della matricola
// (eventualmente in modulo 8).
//
// HINT 1: usarei grandienti.
// HINT 2: non e' necessario fare smoothing o noise reduction in questo caso
//
void selectLine(const cv::Mat & image, int numero, cv::Mat & result)
{
	//impostiamo lo sfondo a bianco
	result.setTo(255);

	if(numero<0)
	{
		std::cout<<"IMPOSTARE IL PARAMETRO numero CON IL MODULO 8 DELL'ULTIMO NUMERO DELLA MATRICOLA!"<<std::endl;
		return;
	}
	
	/* Calcoliamo i gradienti */ 
	Mat g_x= Mat(image.rows,image.cols, CV_32FC1);
	Mat g_y = Mat(image.rows,image.cols, CV_32FC1);
	Mat monodimensionale = (Mat_<float>(1,3) << -1, 0, 1);
	convFloat(image, monodimensionale, g_x);
	convFloat(image, monodimensionale.t(), g_y);
	
	
	/* stampo gx e gy */
	cv::namedWindow("gx", cv::WINDOW_NORMAL);
	cv::imshow("gx", g_x);
	cv::namedWindow("gy", cv::WINDOW_NORMAL);
	cv::imshow("gy", g_y);
	
	if(numero > 8 )
	{
		numero = numero % 8;
	}
	
	for(int i = 0; i < result.rows; i++)
	{
		for(int j=0; j < result.cols; j++)
		{
			if( numero ==0 && g_x.at<float>(i,j) ==0 && g_y.at<float>(i,j)  < 0 )
			{
				//lato 0
				result.at<uchar>(i,j)=0;
			}
			else if(numero ==1 &&  g_x.at<float>(i,j) > 0 && g_y.at<float>(i,j)  < 0 )
			{
				//lato 1
				result.at<uchar>(i,j)=0;
			}
			else if(numero ==2 && g_x.at<float>(i,j) > 0 && g_y.at<float>(i,j) ==0 )
			{
				//lato 2
				result.at<uchar>(i,j)=0;
			}
			else if(numero ==3 && g_x.at<float>(i,j)  > 0 && g_y.at<float>(i,j)  < 0 )
			{
				//lato 3
				result.at<uchar>(i,j)=0;
			}
			else if(numero ==4 && g_x.at<float>(i,j) ==0 && g_y.at<float>(i,j)  > 0)
			{
				//lato 4
				result.at<uchar>(i,j)=0;
			}
			else if(numero ==5 &&  g_x.at<float>(i,j) < 0 && g_y.at<float>(i,j)  > 0 )
			{
				//lato 5
				result.at<uchar>(i,j)=0;
			}
			else if(numero ==6 &&  g_x.at<float>(i,j) < 0 && g_y.at<float>(i,j)  == 0 )
			{
				//lato 6
			}
			else if(numero ==7 &&  g_x.at<float>(i,j) < 0 && g_y.at<float>(i,j)  < 0 )
			{
				//lato 7
				result.at<uchar>(i,j)=0;
			}
		}
	}
	
	

}
///////////////////////////////////////////

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
		sprintf(frame_name,"%s",args.image_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat image = cv::imread(frame_name, CV_8UC1);
		if(image.empty())
		{
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}


		////////////////////////////////////
		//PROCESSING
		//

		// immagine di output
		cv::Mat result(image.rows, image.cols, CV_8UC1);

		//
		//ultimo numero della propria matricola in modulo 8
		//
		int numero = 9;

		selectLine(image, numero, result);
		/////////////////////////////////

		//display image
		cv::namedWindow("ottagono", cv::WINDOW_NORMAL);
		cv::imshow("ottagono", image);

		cv::namedWindow("linea", cv::WINDOW_NORMAL);
		cv::imshow("linea", result);

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
