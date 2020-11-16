//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
 * Funzione per la lettura del file scan.dat
 * Riceve in input la stringa il path del file
 * Restituisce in output l'array dei punti letti
 */
std::vector<Point> read_scan_file(const std::string& scan_path) {
	std::ifstream scan_file;
	scan_file.open(scan_path.c_str());
	
	int num_points;
	scan_file >> num_points;
	std::cout << num_points << std::endl;

	std::vector<Point> points;

	float point_coord;
	int i = 0;
	while(scan_file >> point_coord) {
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

	scan_file.close();
	
	return points;
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

	while(!exit_loop) {
		//generating file name
		//
		//multi frame case
		if(args.image_name.find('%') != std::string::npos)
			sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
		else //single frame case
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

		std::string scan_path("../../data/ex1/data/scan.dat");
		std::string params_front_path("../../data/ex1/data/params_front.dat");

		std::vector<Point> points = read_scan_file(scan_path);

		for(Point& point : points) {
			std::cout << "x: " << point.x << ", y: " << point.y << ", z: " << point.z << "\n";
		}

		/////////////////////

		//display input_img
		cv::namedWindow("input_img", cv::WINDOW_NORMAL);
		cv::imshow("input_img", input_img);

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
