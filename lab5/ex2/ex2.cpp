// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

/**
 * Struct per contenere i parametri della camera
 */
struct CameraParams {
    // size
    int w, h;

    // intrinsics
    float ku, kv;
    float u0, v0;

    // estrinsics
    cv::Affine3f RT;
};

/**
 * Funzione per la costruzione della matrice degli estrinseci RT
 * a partire dai 3 parametri di rotazione e i 3 parametri di traslazione
 * La RT risultante viene inserita nel parametro affine, ritornato per riferimento
 */
void PoseToAffine(float rx, float ry, float rz, float tx, float ty, float tz, cv::Affine3f& affine) {
    cv::Mat world_RvecX_cam = cv::Mat(1,3,CV_32F);
    world_RvecX_cam.at<float>(0,0) = rx;
    world_RvecX_cam.at<float>(0,1) = 0.0;
    world_RvecX_cam.at<float>(0,2) = 0.0;
    cv::Mat world_Rx_cam;
    cv::Rodrigues(world_RvecX_cam, world_Rx_cam); // Converts a rotation matrix to a rotation vector or vice versa. 

/*
void cv::Rodrigues  (  
  InputArray   src,
  OutputArray   dst,
  OutputArray   jacobian = noArray()  // 3x9 o or 9x3, which is a matrix of partial derivatives of the output array components with respect to the input array components.
 )   
*/
    
    cv::Mat world_RvecY_cam = cv::Mat(1,3,CV_32F);
    world_RvecY_cam.at<float>(0,0) = 0.0;
    world_RvecY_cam.at<float>(0,1) = ry;
    world_RvecY_cam.at<float>(0,2) = 0.0;
    cv::Mat world_Ry_cam;
    cv::Rodrigues(world_RvecY_cam, world_Ry_cam); // Converts a rotation matrix to a rotation vector or vice versa.
    
    cv::Mat world_RvecZ_cam = cv::Mat(1,3,CV_32F);
    world_RvecZ_cam.at<float>(0,0) = 0.0;
    world_RvecZ_cam.at<float>(0,1) = 0.0;
    world_RvecZ_cam.at<float>(0,2) = rz;
    cv::Mat world_Rz_cam;
    cv::Rodrigues(world_RvecZ_cam, world_Rz_cam); // Converts a rotation matrix to a rotation vector or vice versa.
    
    cv::Mat world_R_cam = world_Rx_cam*world_Ry_cam*world_Rz_cam; // Multiplication order is important (it depends on how the rotation is built)
    
    cv::Mat world_t_cam = cv::Mat(1,3,CV_32F);
    world_t_cam.at<float>(0,0) = tx;
    world_t_cam.at<float>(0,1) = ty;
    world_t_cam.at<float>(0,2) = tz;
    
    /// Data una matrice di rotazione e un vettore di traslazione, restituisce la matrice di rototraslazione
    affine = cv::Affine3f(world_R_cam, world_t_cam); // costruttore, Affine transform. It represents a 4x4 homogeneous transformation matrix T
}

/**
 * Funzione per la lettura di parametri della camera da un file
 * I parametri vengono poi inseriti in un CameraParams, ritornato per riferimento
 */
void LoadCameraParams(const std::string& filename, CameraParams& params) {
    std::ifstream file;
    file.open(filename.c_str());
    
    file >> params.w >> params.h;
    
    file >> params.ku >> params.kv;
    file >> params.u0 >> params.v0;
    
    float rx, ry, rz, tx, ty, tz;
    file >> rx >> ry >> rz;
    file >> tx >> ty >> tz;
    
    PoseToAffine(rx, ry, rz, tx, ty, tz, params.RT);
}


void BEV(const cv::Mat& image, const CameraParams& params, cv::Mat & output) {
	output = cv::Mat(400, 400, CV_8UC3, cv::Scalar(0));

	// Matrice degli estrinseci
    Eigen::Matrix<float, 4, 4> RT;
    cv::Affine3f RT_inv = params.RT.inv();
    RT << RT_inv.matrix(0,0), RT_inv.matrix(0,1), RT_inv.matrix(0,2), RT_inv.matrix(0,3),
          RT_inv.matrix(1,0), RT_inv.matrix(1,1), RT_inv.matrix(1,2), RT_inv.matrix(1,3),
          RT_inv.matrix(2,0), RT_inv.matrix(2,1), RT_inv.matrix(2,2), RT_inv.matrix(2,3),
                           0,                  0,                  0,                  1;

    // Matrice degli intrinseci
    Eigen::Matrix<float, 3, 4> K;
    K << params.ku,         0, params.u0, 0,
                 0, params.kv, params.v0, 0,
                 0,         0,         1, 0;

	// M
    Eigen::Matrix<float, 3, 4> M;
    M = K*RT;

    // Calcolo la nuova M, con l'aggiunta del vincolo y = 0
    Eigen::Matrix<float, 4, 4> M_new;
    M_new << M, 0, 1, 0, 0;

	// Siccome y = 0, posso eliminare la seconda colonna 
    Eigen::Matrix<float, 3, 3> M_new_cropped;
    M_new_cropped <<  M_new(0,0), M_new(0,2), M_new(0,3),
                      M_new(1,0), M_new(1,2), M_new(1,3),
                      M_new(2,0), M_new(2,2), M_new(2,3);

	// Calcolo la matrice inversa
    Eigen::Matrix<float, 3, 3> M_inv;
    M_inv = M_new_cropped.inverse();

	// Ciclo su ogni punto dell'immagine per calcolare i punti mondo
    for(int r = 0; r < image.rows; ++r) {
        for(int c = 0; c < image.cols; ++c) {
            Eigen::Vector3f uv_point(c, r, 1);
            Eigen::Vector3f point_world;

			// Calcolo il punto mondo
            point_world = M_inv*uv_point;

			// Normalizzo il punto mondo
			float w = point_world[2];
            float x = point_world[0]/w;
            float z = point_world[1]/w;

			// Fattore di scala
			float k = 50.0f;
			x*=k;
      		z*=k;

			// Offset delle origini dei due sistemi di riferimento
      		z = output.rows - z;
      		x = output.cols/2 + x;

            // Assegno il corrente pixel di input al pixel di output coordinate appena calcolate
            if (x > 0 && x < output.cols && z > 0 && z < output.rows) {
				output.at<cv::Vec3b>(z, x)[0] = image.at<cv::Vec3b>(r, c)[0];
				output.at<cv::Vec3b>(z, x)[1] = image.at<cv::Vec3b>(r, c)[1];
				output.at<cv::Vec3b>(z, x)[2] = image.at<cv::Vec3b>(r, c)[2];
        	}
        }
    }
}
/////////////////////////////////////////////////////////////////////////////


struct ArgumentList {
	std::string input_img;
	std::string params_dat;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
	int desired_args = 5;

	if(argc < desired_args || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help")) {
		std::cout<<"Usage: " << argv[0] << " -i <input_img> -p <params>" <<std::endl;
		return false;
	}

	int i = 1;
	while(i < argc) {
		if(std::string(argv[i]) == "-i") {
			++i;
			args.input_img = std::string(argv[i]);
		}

		else if(std::string(argv[i]) == "-p") {
			++i;
			args.params_dat = std::string(argv[i]);
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
		cv::Mat input_img = cv::imread(args.input_img);
		if(input_img.empty()) {
			std::cout << "Error loading input_img: " << argv[2] << std::endl;
    		return 1;
  		
		}
		
		std::string params_dat = args.params_dat;

		//////////////////////
		//processing code here

		// Devo effettuare una IPM dell'immagine di input sul piano y = 0
		// Prima di tutto calcolo i punti sul mondo, tramite la matrice M inversa
		// Dopodichè, calcolo i punti sull'immagine IPM tramite la normale trasformazione prospettica

		CameraParams params;
		LoadCameraParams(params_dat, params);

		cv::Mat IPM;
		BEV(input_img, params, IPM);


		/////////////////////

		//display images
		cv::namedWindow("input_img", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img", input_img);

		cv::namedWindow("IPM", cv::WINDOW_AUTOSIZE);
		cv::imshow("IPM", IPM);

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
