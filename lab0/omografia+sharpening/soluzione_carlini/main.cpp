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

#include "opencv2/calib3d/calib3d.hpp"


// MIE FUNZIONI
// convoluzione float input CV_8UC1 o CV_32FC1
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1)
{

	// controllo sul kernel simmetrico (dimensioni dispari)
	if (kernel.rows % 2 != 1 || kernel.cols % 2 != 1)
	{
		std::cerr << "- Kernel non simmetrico -";
		exit(1);
	}

	// dimensionamento output
	int padding = 0;
	int outRows = floor( (image.rows + 2*padding - kernel.rows)/stride  + 1 );
	int outCols = floor( (image.cols + 2*padding - kernel.cols)/stride  + 1 );
	out = cv::Mat(outRows, outCols, CV_32FC1);


	// indici di riga e colonna per l'output11
	int rOut = 0;
	int cOut = 0;

	float pxConv = 0.0;
	int kernelRadiusWidth = floor(kernel.cols/2);
	int kernelRadiusHeight = floor(kernel.rows/2);


	// CV_8UC1
	if (image.type() == CV_8UC1)
	{

		for(int r = 0; r < image.rows - kernelRadiusHeight; r += stride)
		{
			for(int c = 0; c < image.cols - kernelRadiusWidth; c += stride)
			{
				// applichiamo il filtro solo dove si sovrappone completamente all'immagine
				if ((r - kernelRadiusHeight >= 0) && (r + kernelRadiusHeight <= image.rows)
					&& (c - kernelRadiusWidth >= 0) && (c + kernelRadiusWidth <= image.cols))
				{
					// calcolo del valore ottenuto applicando il kernel sul'immagine
					for (int kRow = -kernelRadiusHeight; kRow <= kernelRadiusHeight;  ++kRow)
					{
						for (int kCol = -kernelRadiusWidth; kCol <= kernelRadiusWidth; ++kCol)
						{
							float kValue = *((float*)&kernel.data[((kRow + kernelRadiusHeight) * kernel.cols + (kCol + kernelRadiusWidth)) * kernel.elemSize()]);
					    float imageValue = (float)image.data[(r+kRow) * image.cols + (c + kCol)];
							pxConv += (kValue) * (imageValue);
						}
					}


				  *((float *) &out.data[(rOut * out.cols + cOut)*out.elemSize()]) = pxConv;
					pxConv = 0.0;

					// aggiornamento indici di scorrimento dell'output
					cOut++;
				}

			}

			// aggiornamento indici di scorrimento dell'output
			if (cOut == outCols - 1)
			{
				rOut++;
				cOut = 0;
		  }
		}
	}


	// CV_32FC1
	if (image.type() == CV_32FC1)
	{

		for(int r = 0; r < image.rows - kernelRadiusHeight; r += stride)
		{
			for(int c = 0; c < image.cols - kernelRadiusWidth; c += stride)
			{
				// applichiamo il filtro solo dove si sovrappone completamente all'immagine
				if ((r - kernelRadiusHeight >= 0) && (r + kernelRadiusHeight <= image.rows)
					&& (c - kernelRadiusWidth >= 0) && (c + kernelRadiusWidth <= image.cols))
				{
					// calcolo del valore ottenuto applicando il kernel sul'immagine
					for (int kRow = -kernelRadiusHeight; kRow <= kernelRadiusHeight;  ++kRow)
					{
						for (int kCol = -kernelRadiusWidth; kCol <= kernelRadiusWidth; ++kCol)
						{
							float kValue = kernel.at<float>(kRow + kernelRadiusHeight, kCol + kernelRadiusWidth);
							float imageValue = image.at<float>(r+kRow, c + kCol);
							pxConv += (kValue) * (imageValue);
						}
					}


				  *((float *) &out.data[(rOut * out.cols + cOut)*out.elemSize()]) = pxConv;
					pxConv = 0.0;

					// aggiornamento indici di scorrimento dell'output
					cOut++;
				}

			}

			// aggiornamento indici di scorrimento dell'output
			if (cOut == outCols - 1)
			{
				rOut++;
				cOut = 0;
		  }
		}
	}

}
//////////////////////////////////


// generatore Kernel gaussiano
void gaussianKernel(float sigma, int radius, cv::Mat& kernel)
{
	// kernel 1-D orizzontale, singolo canale float 32
	int kernelWidth = (2 * radius) + 1;
  kernel = cv::Mat(1, kernelWidth, CV_32FC1);

	float gaussianValue, normalizedGaussianValue, distributionSum = 0.0;

	// calcolo iniziale non normalizzato dei valori della distribuzione guaussiana e calcolo della somma
	for(int i = 0; i < kernelWidth; ++i)
	{
		gaussianValue = exp(-((i - radius) *  (i - radius)) / (2 * sigma * sigma)) / (2 * M_PI * (sigma * sigma));
		distributionSum += gaussianValue;
		*((float *) &kernel.data[i * kernel.elemSize()]) = gaussianValue;
	}


  // normalizzazione della distribuzione gaussiana
	for(int i = 0; i < kernelWidth; ++i)
	{
		normalizedGaussianValue = *((float *) &kernel.data[i * kernel.elemSize()]) / distributionSum;
		*((float *) &kernel.data[i * kernel.elemSize()]) = normalizedGaussianValue;
	}

}
//////////////////////////////////


// Smoothing
void gaussianBlur2D(cv::Mat& image, cv::Mat& kernel, cv::Mat& outBlur2D)
{

	// blur orizzontale
	cv::Mat outBlurOrizz;
	convFloat(image, kernel, outBlurOrizz);

	cv::Mat transposedKernel;
	cv::transpose(kernel, transposedKernel);

	// blur 2d
	convFloat(outBlurOrizz, transposedKernel, outBlur2D);
}

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

	corners_out.push_back(cv::Point2f(0.0, 0.0)); //top left
	corners_out.push_back(cv::Point2f(output.cols - 1, 0.0)); //top right
	corners_out.push_back(cv::Point2f(output.cols - 1, output.rows - 1)); //bottom right
	corners_out.push_back(cv::Point2f(0.0, output.rows - 1)); //bottom left

	cv::Mat H = cv::findHomography(corners_out, corners_src, 0);


	cv::Point2f pDest;
	cv::Point3f pSource;

	for(int r = 0; r < output.rows; r++)
		for(int c = 0; c < output.cols; c++)
		{
			pSource.x = c;
			pSource.y = r;
			pSource.z = 1;

			pSource = H * pSource;

			pDest.x = pSource.x;
			pDest.y = pSource.y;

			output.at<float>(r,c) = image.at<float>((int)pDest.y, (int)pDest.x);
		}


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
	cv::Mat alphaImage;

	float LoG_Conv_Data[] = {0,1,0,
										       1,-4,1,
										       0,1,0};

	cv::Mat LoG_conv_I(3, 3, CV_32FC1, LoG_Conv_Data);


	convFloat(image, LoG_conv_I, alphaImage);

	alphaImage *= alpha;

	//cv::Mat convertedAlphaIMage;
	//alphaImage.convertTo(convertedAlphaIMage, CV_8UC1);

	for(int r = 0; r < image.rows; r++)
		for(int c = 0; c < image.cols; c++)
		{

			float px2 = alphaImage.at<float>(r,c);
			float px1 = (float)image.at<u_int8_t>(r,c);

			float px = px1 - px2;

			u_int8_t satPx;

			// saturazione
			if (px > 255.0)
				px = 255.0;
			else if (px < 0)
				px = 0.0;

			output.at<u_int8_t>(r,c) = (u_int8_t)px;
		}



}
//////////////////////////////////////////////



// MAIN
int main(int argc, char **argv)
{

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
    // alfa e' una costante float, utilizziamo 0.5
    //
    // LoG e' il Laplaciano del Gaussiano. Utilizziamo l'approssimazione 3x3 vista a lezione
    //
    // In questo caso serve fare il contrast stratching nelle convoluzioni?
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


    cv::namedWindow("Output Warp", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output Warp", outimage_win);

    cv::namedWindow("Input Gray", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray", inputgray);

    cv::namedWindow("Input Gray Sharpened", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray Sharpened", sharpened);


    cv::waitKey();

    return 0;
}
