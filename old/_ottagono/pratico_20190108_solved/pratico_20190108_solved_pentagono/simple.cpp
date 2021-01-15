//OpneCV
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

///////////////////////////////////////////
// PROCESSING CODE HERE
// Scrivere un programma C/C++ che crei una nuova immagine contenente
// unicamente il lato corrispondente all’ultimio numero della matricola
// (eventualmente in modulo 8).
//
// HINT 1: usarei grandienti.
// HINT 2: non e' necessario fare smoothing o noise reduction in questo caso
//

template <class T>
int contrastStretching(cv::Mat& image)
{
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();

    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            for (int k = 0; k < image.channels(); k++)
            {
                T value = *((T*) &image.data[(c + r * image.cols) * image.elemSize() + k * image.elemSize1()]);
                if (value < min)
                {
                    min = value;
                }
                if (value > max)
                {
                    max = value;
                }
            }
        }
    }

    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            for (int k = 0; k < image.channels(); k++)
            {
                *((T*) &image.data[(c + r * image.cols) * image.elemSize() + k * image.elemSize1()]) =
                        (*((T*) &image.data[(c + r * image.cols) * image.elemSize() + k * image.elemSize1()]) - min) *
                        (255.0f / (max - min));
            }
        }
    }

    min = std::numeric_limits<T>::max();
    max = std::numeric_limits<T>::min();

    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            for (int k = 0; k < image.channels(); k++)
            {
                T value = *((T*) &image.data[(c + r * image.cols) * image.elemSize() + k * image.elemSize1()]);
                if (value < min)
                {
                    min = value;
                }
                if (value > max)
                {
                    max = value;
                }
            }
        }
    }

    return 0;
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

int convInt(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out)
{
	cv::Mat outFloat;
    convFloat(image, kernel, outFloat);
    contrastStretching<float>(outFloat);

    out = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(0));

    for (int r = 0; r < out.rows; r++)
    {
        for (int c = 0; c < out.cols; c++)
        {
            out.data[c + r * out.cols] = (uchar) *((float*) &outFloat.data[(c + r * outFloat.cols) * outFloat.elemSize()]);
        }
    }

    return 0;
}

void selectLine(const cv::Mat & image, int numero, cv::Mat & result)
{
	//impostiamo lo sfondo a bianco
	result.setTo(255);

	if(numero<0)
	{
		std::cout<<"IMPOSTARE IL PARAMETRO numero CON IL MODULO 8 DELL'ULTIMO NUMERO DELLA MATRICOLA!"<<std::endl;
		return;
	}

	//gradiente CON SEGNO
	cv::Mat gxFloat;
    cv::Mat dh = (cv::Mat_<float>(1, 3) << -1, 0, 1);
	convFloat(image, dh, gxFloat);

	//gradiente CON SEGNO
	cv::Mat gyFloat;
    cv::Mat dv = (cv::Mat_<float>(3, 1) << -1, 0, 1);
	convFloat(image, dv, gyFloat);

	bool lati[8][4] = {
	//       Tabella di condizioni, una per ogni lato
	//
	//       gy>0   gy<0   gx>0   gx<0
			{false, true,  false, false}, //gy<0 , gx==0
			{false, true,  true,  false}, //gy<0 , gx>0
			{false, false, true,  false}, //gy==0, gx>0
			{true,  false, true,  false}, //gy>0 , gx>0
			{true,  false, false, false}, //gy>0 , gx==0
			{true,  false, false, true }, //gy>0 , gx<0
			{false, false, false, true }, //gy==0, gx<0
			{false, true,  false, true}   //gy<0 , gx<0
	};

	int epsilon = 1;
	for(int r=0;r<image.rows;++r)
	{
		for(int c=0;c<image.cols;++c)
		{
			bool c0 = gyFloat.at<float>(r,c) >  epsilon;  // gy>0
			bool c1 = gyFloat.at<float>(r,c) < -epsilon;  // gy<0
			bool c2 = gxFloat.at<float>(r,c) >  epsilon;  // gx>0
			bool c3 = gxFloat.at<float>(r,c) < -epsilon;  // gx<0

			//verifichiamo che se il pixel corrente ha il pattern di segni dei grandenti cercato
			if(lati[numero][0] == c0 &&
			   lati[numero][1] == c1 &&
			   lati[numero][2] == c2 &&
			   lati[numero][3] == c3)
				result.at<unsigned char>(r,c) = 0; //se il pixel verifica il patterne cercato, imposto il valore a nero (0)
		}
	}


	//////////////////////////
	// Per puro debug
	//
	//
	// Visualizziamo i gradienti come unsigned char dopo aver fatto un contrast stratching
	//
	// Questo passaggio NON e' necessario per la soluzione, per la quale e' sufficiente il calcolo
	// fatto in precedenza con valori puramente float con segno
	//
	// Questa visualizzazione aiuta pero' a capire come variano i segni del gradiente in base al lato
	//
	cv::Mat gx;
	convInt(image, dh, gx);
	cv::namedWindow("Gx", cv::WINDOW_NORMAL);
	cv::imshow("Gx",gx);

	cv::Mat gy;
	convInt(image, dv, gy);
	cv::namedWindow("Gy", cv::WINDOW_NORMAL);
	cv::imshow("Gy",gy);
}
///////////////////////////////////////////



/////////////////////////////////////////
// EX2.
//
// In un caso generale, in cui la forma sia sempre CONVESSA ma abbia piu’ di 8 lati,
// come potrei risolvere questo problema? Come potrei individuare ogni orientazione nel caso generale?
//
//
// La soluzione piu' generale consiste nel calcolare l'ORIENTAZIONE DEL GRADIENTE: se la forma in input e' convessa,
// e' garantito che ad ogni lato corrisponda esattamente UN solo angolo di orientazione del gradiente possibile.
// Per cui, se avessimo un decagono (10 lati), avremmo esattente 10 possibili orientazioni del gradiente, una per ogni lato appunto
//
// Nel caso semlice di una forma equilatera di N lati, ogni lato sara' quindi caratterizzato da un orientazione del gradiente pari a:
//
//      (2 * M_PI * i)/N
//
// dove i e' nel range 0..N-1
//
void selectLine_ex2(const cv::Mat & image, int numero, int N, cv::Mat & result)
{
	//impostiamo lo sfondo a bianco
	result.setTo(255);

	if(numero<0)
	{
		std::cout<<"IMPOSTARE IL PARAMETRO numero CON IL MODULO 8 DELL'ULTIMO NUMERO DELLA MATRICOLA!"<<std::endl;
		return;
	}


	//gradiente CON SEGNO
	cv::Mat gxFloat;
    cv::Mat dh = (cv::Mat_<float>(1, 3) << -1, 0, 1);
	convFloat(image, dh, gxFloat);

	//gradiente CON SEGNO
	cv::Mat gyFloat;
    cv::Mat dv = (cv::Mat_<float>(3, 1) << -1, 0, 1);
	convFloat(image, dv, gyFloat);


	//calcolo dell'orientazione del gradiente
	cv::Mat orientation(image.rows, image.cols, CV_32FC1, cv::Scalar(0));
	for(int r=0;r<image.rows;++r)
	{
		for(int c=0;c<image.cols;++c)
		{
			//atan2 restituisce un angolo compreso tra -M_PI,M_PI
			orientation.at<float>(r,c) = float(atan2(gyFloat.at<float>(r,c), gxFloat.at<float>(r,c)));

			//mi riporto nel range 0,2*M_PI
			orientation.at<float>(r,c) += orientation.at<float>(r,c)<0 ? 2*M_PI :0;
		}
	}

	/////////////////////
	// Questo e' necessario perche' ho *infelicemente* deciso di numerare i lati partendo da quello in alto.
	// Se pero' prendiamo in considerazione gli assi x/y (o righe/colonne), vediamo che l'angolo 0 corrisponde, in realta', al lato numero 2
	// Aggiungo quindi 6 alla numerazione per fare in modo che i risulti dei due metodi siano identici
	//
	// naturalmente questo dettaglio verra' ignorato nelle soluzioni presentate durante il compito
	//
	numero = (numero+6)%N;
	///////////////////////

	float target_angle = 2.0*M_PI*float(numero)/float(N);
	std::cout<<"target_angle numero/2*M_PI*N = "<<target_angle<<std::endl;

	float epsilon = 0.1;
	for(int r=0;r<image.rows;++r)
	{
		for(int c=0;c<image.cols;++c)
		{
			//l'orientazione del gradiente deve essere simile a quella cercata
			//dobbiamo pero' escludere tutte le zone con gradiente nullo!!
			//qualunque sia l'orientazione del lato, una gradiente diverso da zero deve esistere sempre, altrimenti siamo in una zona omogena
			if( (fabs(gyFloat.at<float>(r,c))>epsilon*10 || fabs(gxFloat.at<float>(r,c))>epsilon*10) && fabs(target_angle-orientation.at<float>(r,c)) < epsilon)
				result.at<unsigned char>(r,c) = 0;
		}
	}


	//////////////////////////
	// Per puro debug
	//
	//
	// Visualizziamo i gradienti come unsigned char dopo aver fatto un contrast stratching
	//
	// Questo passaggio NON e' necessario per la soluzione, per la quale e' sufficiente il calcolo
	// fatto in precedenza con valori puramente float con segno
	//
	// Questa visualizzazione aiuta pero' a capire come variano i segni del gradiente in base al lato
	//
	cv::Mat gx;
	convInt(image, dh, gx);
	cv::namedWindow("Gx", cv::WINDOW_NORMAL);
	cv::imshow("Gx",gx);
	cv::imwrite("Gx.pgm",gx);

	cv::Mat gy;
	convInt(image, dv, gy);
	cv::namedWindow("Gy", cv::WINDOW_NORMAL);
	cv::imshow("Gy",gy);
	cv::imwrite("Gy.pgm",gy);

    cv::Mat adjMap;
    cv::convertScaleAbs(orientation, adjMap, 255 / (2.0 * M_PI));
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
    cv::imshow("Out", falseColorsMap);
	cv::imwrite("otient.pgm",falseColorsMap);
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
		int numero = 6;

		selectLine(image, numero, result);

		cv::Mat result2(image.rows, image.cols, CV_8UC1);
		selectLine_ex2(image, numero, 8 ,result2);
		/////////////////////////////////

		//display image
		cv::namedWindow("ottagono", cv::WINDOW_NORMAL);
		cv::imshow("ottagono", image);

		cv::namedWindow("linea", cv::WINDOW_NORMAL);
		cv::imshow("linea", result);

		cv::namedWindow("linea ex2", cv::WINDOW_NORMAL);
		cv::imshow("linea ex2", result2);

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
