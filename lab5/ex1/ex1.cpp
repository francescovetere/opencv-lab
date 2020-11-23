//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath> // sinf, cosf

//std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList {
	std::string image_name;		    //!< input_img file name
	int wait_t;                     //!< waiting time
};

/**
 * Dichiaro una struct per il generico punto letto da scan.dat
 */
struct Point {
	float x;
	float y;
	float z;
};

struct Camera {
	float width;
	float height;
	float alpha;
	float beta;
	float u0;
	float v0;
	float orientation_x;
	float orientation_y;
	float orientation_z;
	float position_x;
	float position_y;
	float position_z;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<3 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -i <image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -i arg                   input_img name. Use %0xd format for multiple images."<<std::endl<<
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

/**
 * Funzione per la lettura di un file strutturato nel seguente modo
 * 	<N_punti>
 *  <point1.x> <point1.y> <point1.z>
 *  ...
 *  <pointN.x> <pointN.y> <pointN.z>
 *
 * Riceve in input la stringa il path del file
 * Restituisce in output l'array dei punti letti
 */
std::vector<Point> read_points(const std::string& path) {
	std::ifstream file;
	file.open(path.c_str());
	
	int num_points;
	file >> num_points;
	std::cout << num_points << std::endl;

	std::vector<Point> points;

	float point_coord;
	int i = 0;
	while(file >> point_coord) {
		float x, y, z;
		switch(i){
			case 0:
				x = point_coord;
				break;
			case 1:
				y = point_coord;
				break;
			case 2:
				z = point_coord;
				break;
		}
		
		Point p {x, y, z};
		points.push_back(p);

		i = (i+1) % 3;
	}

	file.close();
	
	return points;
}

/**
 * Funzione per la lettura di un file strutturato nel seguente modo
 * 640 480 //larghezza e altezza
 * 400 400 //lunghezza focale in pixel
 * 320 240 //centri ottici u0, vo
 * 0.0 0.0 0.0 //orientazione rispetto a x,y,z
 * 0.0 -5.0 -10.0 //posizione x,y,z
 *
 * Riceve in input la stringa il path del file
 * Restituisce in output l'array dei punti letti
 */
Camera read_camera(const std::string& path) {
	std::ifstream file;
	file.open(path.c_str());

	Camera camera;
	float x, y, z;

	file >> camera.width >> camera.height
		 >> camera.alpha >> camera.beta
	     >> camera.u0 >> camera.v0
		 >> camera.orientation_x >> camera.orientation_y >> camera.orientation_z
		//  >> x >> y >> z;
		>> camera.position_x >> camera.position_y >> camera.position_z;

	//////////////////////////////////////
	// I parametri letti sono purtroppo scritti in ordine sbagliato sul file, in questo particolare caso
	// camera.position_x = -y;
	// camera.position_y = -z;
	// camera.position_z = x;
	//////////////////////////////////////

	file.close();

	return camera;
}


int main(int argc, char **argv) {
	int frame_number = 0;
	char frame_name[256];
	bool exit_loop = false;

	//////////////////////
	//parse argument list:
	//////////////////////
	ArgumentList args;
	if(!ParseInputs(args, argc, argv)) {
		return 1;
	}

	// Lettura punti e parametri camera
	std::string scan_path("../../data/ex1/data/scan.dat");
	std::string params_front_path("../../data/ex1/data/params_front.dat");

	std::vector<Point> points = read_points(scan_path);

	// DEBUG
	// for(Point& point : points) {
	// 	std::cout << "x: " << point.x << ", y: " << point.y << ", z: " << point.z << "\n";
	// }

	Camera camera = read_camera(params_front_path);
	// DEBUG
	// std::cout << camera.alpha << std::endl;

	while(!exit_loop) {
		//generating file name
		//
		//multi frame case
		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //sinfgle frame case
			sprintf(frame_name,"%s",args.image_name.c_str());

		//opening file
		std::cout<<"Opening "<<frame_name<<std::endl;

		cv::Mat input_img = cv::imread(frame_name);
		if(input_img.empty()) {
			std::cout<<"Unable to open "<<frame_name<<std::endl;
			return 1;
		}


		//////////////////////
		//processing code here
		// Costruisco la matrice degli instrinseci K (3x3)
		float K_data[3][3] = {{camera.alpha, 0.0f, camera.u0},
						 	  {0.0f, camera.beta, camera.v0},
						      {0.0f, 0.0f, 1.0f}
					         };

		cv::Mat K(3, 3, CV_32FC1, K_data);
		
		// DEBUG
		// std::cout << K << std::endl;


		// Costruisco la matrice degli estrinseci [R T] (3x4)

		// R = Rx(alpha)*Ry(beta)*Rz(theta) (3x3)
		float Rx_data[3][3] = {{1.0f, 0.0f, 0.0f},
						 	   {0.0f, cosf(camera.orientation_x), -sinf(camera.orientation_x)},
						 	   {0.0f, sinf(camera.orientation_x), cosf(camera.orientation_x)}
					    	  };

		float Ry_data[3][3] = {{cosf(camera.orientation_y), 0.0f, sinf(camera.orientation_y)},
						 	   {0.0f, 1.0f, 0.0f},
						 	   {-sinf(camera.orientation_y), 0.0f, cosf(camera.orientation_y)}
					    	  };

		float Rz_data[3][3] = {{cosf(camera.orientation_z), -sinf(camera.orientation_z), 0.0f},
						 	   {sinf(camera.orientation_z), cosf(camera.orientation_z), 0.0f},
						 	   {0.0f, 0.0f, 1.0f}
					    	  };
		
		cv::Mat Rx(3, 3, CV_32FC1, Rx_data);
		cv::Mat Ry(3, 3, CV_32FC1, Ry_data);
		cv::Mat Rz(3, 3, CV_32FC1, Rz_data);

		cv::Mat R = Rx * Ry * Rz;

		// DEBUG
		// std::cout << R << std::endl;


		// Costruisco la matrice degli instrinseci K (3x3)
		float T_data[3][1] = {{camera.position_x}, {camera.position_y}, {camera.position_z}};

		cv::Mat T(3, 1, CV_32FC1, T_data);

		// DEBUG
		// std::cout << T << std::endl;


		// Calcolo M = K * [R T]
		cv::Mat RT;
		hconcat(R, T, RT);
		cv::Mat M = K * RT;

		//DEBUG
		std::cout << "M: " << M << std::endl;

		
		// Dichiaro la matrice in cui visualizzerò il risultato
		// Inizialmente tutta nera, metterò a bianco i pixel proiettati
		std::cout << camera.width << " " << camera.height << std::endl;
		cv::Mat out = cv::Mat::zeros(camera.height, camera.width, CV_8UC1);
		
		// Per ogni punto Pw nel mondo, calcolo la sua proiezione 2D tramite la formula P = M * Pw
		// Torno poi in coordinate euclidee, ottenendo quindi P = (x, y)
		for(Point& point : points) {
			// trasformo il punto Pw attuale in coordinate omogenee
			float Pw_data[4][1] = {{point.x}, {point.y}, {point.z}, 1.0f};
			cv::Mat Pw(4, 1, CV_32FC1, Pw_data);

			cv::Mat P = M * Pw;

			int x = P.at<float>(0,0) / P.at<float>(2,0);
			int y = P.at<float>(1,0) / P.at<float>(2,0);
			
			// DEBUG
			// std::cout << "(" << x << ", " << y << ")" << std::endl;

			std::cout << "(" << x << ", " << y << ")" << std::endl;
			// Visualizzo il punto solo se rientra nella finestra!
			if(x > 0 && x < camera.width && y > 0 && y < camera.height) {
				out.at<u_int8_t>(x, y) = 255;
				
			}
		}
		/////////////////////

		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_AUTOSIZE);
		cv::imshow("input_img", input_img);

		//display out
		cv::namedWindow("out", cv::WINDOW_AUTOSIZE);
		cv::imshow("out", out);

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

		// Ruoto attorno alla scena
		camera.orientation_x += 0.05f;
		camera.orientation_y += 0.05f;
		camera.orientation_z += 0.05f;
	}

	return 0;
}
