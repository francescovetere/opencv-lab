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

using namespace std;
using namespace cv;
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
    		point[0] = xmin + c_out*((xmax - xmin)/float(output.cols));
    		point[1] = 0;
    		point[2] = zmax - r_out*(zmax/float(output.rows));
    		point[3] = 1.0;

    		Eigen::Vector3f risultato;
    		risultato = K * RT * point;

    		double u = risultato[0] / risultato[2];
    		double v = risultato[1] / risultato[2];

    		if(u>=0 && u<=(image.cols-1) && v>=0 && v<=(image.rows-1))
    			
				output.at<uchar> (r_out,c_out) = image.at<uchar> (v,u);
			
			
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
}
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    
    if (argc < 3) 
    {
        std::cerr << "Usage pratico <image_filename> <camera_params_filename>" << std::endl;
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





