//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


//////////////////////////////////////////////
/// EX1
//
// Nota la posizione dei 4 angoli della copertina del libro nell'immagine "input.jpg"
// generare la corrispondente immagine vista dall'alto, senza prospettiva.
//
// Si tratta di trovare l'opportuna trasformazione che fa corrispondere la patch di immagine
// input.jpg corrispondente alla copertina del libro con la vista dall'alto della stessa.
//
// Che tipo di trasformazione e'? Come si puo' calcolare con i dati forniti?
//
// E' possibile utilizzare alcune funzioni di OpenCV
//
void WarpBookCover(const cv::Mat & image, cv::Mat & output, const std::vector<cv::Point2f> & corners_src)
{
	std::vector<cv::Point2f> corners_out;

	/*
	* YOUR CODE HERE
	*
	*
	*/
}
/////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////
/// EX2
//
// Applicare il filtro di sharpening visto a lezione
//
// Per le convoluzioni potete usare le funzioni sviluppate per il primo assegnamento
//
//
void sharpening(const cv::Mat & image, cv::Mat & output, float alpha)
{
	output = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));

    cv::Mat LoG_conv_I;

	/*
	* YOUR CODE HERE
	*
	*
	*/
}
//////////////////////////////////////////////


int main(int argc, char **argv) {
    
    if (argc != 2)
    {
        std::cerr << "Usage ./prova <image_filename>" << std::endl;
        return 0;
    }
    
    //images
    cv::Mat input;

    // load image from file
    input = cv::imread(argv[1]);
	if(input.empty())
	{
		std::cout<<"Error loading input image "<<argv[1]<<std::endl;
		return 1;
	}





    //////////////////////////////////////////////
    /// EX1
    //
    // Creare un'immagine contenente la copertina del libro input come vista "dall'alto" (senza prospettiva)
    //
    //
	//

	// Dimensioni note e fissate dell'immagine di uscita (vista dall'alto):
	constexpr int outwidth = 431;
	constexpr int outheight = 574;
	cv::Mat outwarp(outheight, outwidth, input.type(), cv::Scalar(0));

	//posizioni note e fissate dei quattro corner della copertina nell'immagine input
    std::vector<cv::Point2f> pts_src = { cv::Point2f(274,189), //top left
    		                             cv::Point2f(631,56), //top right
    									 cv::Point2f(1042,457), //bottom right
										 cv::Point2f(722,764)};//bottom left

    WarpBookCover(input, outwarp, pts_src);
    //////////////////////////////////////////////







    //////////////////////////////////////////////
    /// EX2
    //
    // Applicare uno sharpening all'immagine cover
    //
    // Immagine = Immagine - alfa(LoG * Immagine)
    //
    //
    // alfa e' una costante float, utilizziamo 0.5
    //
    //
    // LoG e' il Laplaciano del Gaussiano. Utilizziamo l'approssimazione 3x3 vista a lezione
    //
    //
    // In questo caso serve fare il contrast stratching nelle convoluzioni?
    //
    //

    //immagine di uscita sharpened
	cv::Mat sharpened(input.rows, input.cols, CV_8UC1);

	//convertiamo l'immagine della copertina a toni di grigio, per semplicita'
	cv::Mat inputgray(input.rows, input.cols, CV_8UC1);
	cv::cvtColor(input, inputgray, cv::COLOR_BGR2GRAY);

	sharpening(inputgray, sharpened, 0.8);
    //////////////////////////////////////////////






    ////////////////////////////////////////////
    /// WINDOWS
    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input", input);
    
    cv::Mat outimage_win(std::max(input.rows, outwarp.rows), input.cols+outwarp.cols, input.type(), cv::Scalar(0));
    input.copyTo(outimage_win(cv::Rect(0,0,input.cols, input.rows)));
    outwarp.copyTo(outimage_win(cv::Rect(input.cols,0,outwarp.cols, outwarp.rows)));

    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", outimage_win);

    cv::namedWindow("Input Gray", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray", inputgray);

    cv::namedWindow("Input Gray Sharpened", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray Sharpened", sharpened);

    cv::waitKey();

    return 0;
}





