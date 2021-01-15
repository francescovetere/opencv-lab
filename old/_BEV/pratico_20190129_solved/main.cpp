// std
#include <iostream>
#include <fstream>

// opencv
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

// eigen
#include <eigen3/Eigen/Core>

// utils
#include "utils.h"


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

//////////////////////////////////////////////
/// EX1
//
// Creare una vista dall'alto (Bird Eye View) a partire da image e dai
// corrispondenti parametri di calibrazione
//
// L'immagine di uscita sara' formata da 400 righe e 400 colonne
//
// L'immagine di uscita rappresenta un'area di 20m x 20m di fronte alla camera
//
// In particolare abbiamo che:
//   angolo in alto a sinistra:  x=-10, z=20
//   angolo in alto a destra:    x=+10, z=20
//   angolo in basso a sinistra: x=-10, z=0
//   angolo in basso a destra:   x=+10, z=0
//   la y=0 sempre
//
// Quindi esiste una mappatura 1:1 tra pixel di output (r,c) e posizioni nel mondo (x,y,z)
// Cioe', ogni pixel dell'immagine output corrispinde ad un preciso punto del mondo,
// che dipende dalla riga e dalla colonna
//
// Dato un punto (x,y,z) nel mondo, come faccio a sapere a quale pixel di image corrisponde?
// Tramite la matrice di priezione prospettica M=...
//
//
// In altre parole:
//   1) per ogni pixel (r_out,c_out) di output, calcolare il corrispondente punto (x,y,z) mondo
//   2) per ogni (x,y,z) mondo, calcolare il pixel corrispondente (r_in,c_in) su image, tramite M
//   3) copiare il pixel (r_in,c_in) di image dentro il pixel (r_out,c_out) di output
//
void BEV(const cv::Mat & image, const CameraParams& params, cv::Mat & output)
{
	output = cv::Mat(400, 400, CV_8UC1, cv::Scalar(0));

	//matrice di roto-traslazione
    Eigen::Matrix<float, 4, 4> RT;
    cv::Affine3f RT_inv = params.RT.inv();
    RT << RT_inv.matrix(0,0), RT_inv.matrix(0,1), RT_inv.matrix(0,2), RT_inv.matrix(0,3),
          RT_inv.matrix(1,0), RT_inv.matrix(1,1), RT_inv.matrix(1,2), RT_inv.matrix(1,3),
          RT_inv.matrix(2,0), RT_inv.matrix(2,1), RT_inv.matrix(2,2), RT_inv.matrix(2,3),
                           0,                  0,                  0,                  1;

    //matrice degli intrinseci
    Eigen::Matrix<float, 3, 4> K;
    K << params.ku,         0, params.u0, 0,
                 0, params.kv, params.v0, 0,
                 0,         0,         1, 0;

    //Limiti dell'area di interesse nel mondo:
    float xmax =  10.0;
    float xmin = -10.0;
    float zmin =   0.0;
    float zmax =  20.0;

    for(int r_out = 0; r_out<output.rows;++r_out)
    {
    	for(int c_out = 0; c_out<output.cols;++c_out)
    	{
    	    /**
    	     * YOUR CODE HERE:
    	     *
    	     * M = ...?
    	     *
    	     * (u,v) <- M*(x,y,z)
    	     * ...
    	     */

    		Eigen::Vector4f point;
    		point.x() = xmin + c_out*((xmax - xmin)/float(output.cols));
    		point.y() = 0;
    		point.z() = zmax - r_out*(zmax/float(output.rows));
    		point.w() = 1.0;

    		Eigen::Vector3f uv_point;
    		uv_point = K * RT * point;

    		double u = uv_point.x() / uv_point.z();
    		double v = uv_point.y() / uv_point.z();

    		if(u>=0 && u<=(image.cols-1) && v>=0 && v<=(image.rows-1))
    			output.data[r_out*output.cols +c_out] = image.data[int(v)*image.cols + int(u)];
    	}
    }
}
/////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////
/// EX2
//
// Tramite un'oppurtuna operazione di convoluzione, evidenziare le linee di demarcazione della strada
//
// In particolare, utilizzare un kernel 1x5 (orizzontale)
//
// Come dovrebbe essere fatto questo kernel per individuare un pattern basso-alto-basso?
// Cioe' un pattern scuro(asfalto), chiaro(linea), scuro(asfalto)?
//
// cv::Mat hlh = (cv::Mat_<float>(1, 5) << ?, ?, ?, ?, ?);
//
//
// HINT: Applicare una binarizzazione dopo la convoluzione
//
void FindLines(const cv::Mat & bev, cv::Mat & output)
{
	output = cv::Mat(400, 400, CV_8UC1, cv::Scalar(0));

    /**
     * YOUR CODE HERE:
     *
     * 1) Convoluzione con kernel (1,5) opportuno
     * 2) Binarizzazione
     */
    cv::Mat hlh = (cv::Mat_<float>(1, 5) << -1, 0, 2, 0, -1);
	convInt(bev, hlh, output);

	for(int i=0;i<output.rows*output.cols;++i)
		if(output.data[i] < 190)
			output.data[i] = 0;
}
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    
    if (argc < 3) 
    {
        std::cerr << "Usage prova <image_filename> <camera_params_filename>" << std::endl;
        return 0;
    }
    
    
    // images
    cv::Mat input,bev_image, lines_image;

    // load image from file
    input = cv::imread(argv[1], CV_8UC1);

    // load camera params from file
    CameraParams params;
    LoadCameraParams(argv[2], params);
            
    
    //////////////////////////////////////////////
    /// EX1
    //
    // funzione che restituisce una Bird Eye View dell'immagine iniziale
    //
    BEV(input, params, bev_image);
    //////////////////////////////////////////////

    //////////////////////////////////////////////
    /// EX2
    //
    // funzione che evidenzia le linee della carreggiata
    //
    FindLines(bev_image, lines_image);
    //////////////////////////////////////////////
    
    ////////////////////////////////////////////
    /// WINDOWS
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", input);
    
    cv::namedWindow("BEV", cv::WINDOW_AUTOSIZE);
    cv::imshow("BEV", bev_image);
    
    cv::namedWindow("Lines", cv::WINDOW_AUTOSIZE);
    cv::imshow("Lines", lines_image);

    cv::waitKey();

    return 0;
}





