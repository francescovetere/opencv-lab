//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

// Non necessario per l'esame
#define USE_SGM

#ifdef USE_SGM
//richiede opencv_contrib/modules/stereo
#include <opencv2/calib3d.hpp>
#include <opencv2/stereo.hpp>

#include <opencv2/viz.hpp>

void testOpenCvSGM(const cv::Mat & left_image, const cv::Mat & right_image);
#endif

struct ArgumentList {
	std::string left_image_name;		    //!< image file name
	std::string right_image_name;		    //!< image file name
	int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv) {

	if(argc<4 || (argc==2 && std::string(argv[1]) == "--help") || (argc==2 && std::string(argv[1]) == "-h") || (argc==2 && std::string(argv[1]) == "-help"))
	{
		std::cout<<"usage: simple -il <left image_name> -ir <right image_name>"<<std::endl;
		std::cout<<"exit:  type q"<<std::endl<<std::endl;
		std::cout<<"Allowed options:"<<std::endl<<
				"   -h	                     produce help message"<<std::endl<<
				"   -il arg                  left image name. Use %0xd format for multiple images."<<std::endl<<
				"   -ir arg                  right image name. Use %0xd format for multiple images."<<std::endl<<
				"   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
		return false;
	}

	int i = 1;
	while(i<argc)
	{
		if(std::string(argv[i]) == "-il") {
			args.left_image_name = std::string(argv[++i]);
		}

		if(std::string(argv[i]) == "-ir") {
			args.right_image_name = std::string(argv[++i]);
		}

		if(std::string(argv[i]) == "-t") {
			args.wait_t = atoi(argv[++i]);
		}
		else
			args.wait_t = 0;

		++i;
	}

	if(args.left_image_name.empty() || args.right_image_name.empty())
	{
		std::cout<<"Please provide both left and right image names."<<std::endl;
		std::cout<<"usage: simple -il <left image_name> -ir <right image_name>"<<std::endl;
		return false;
	}

	return true;
}



///////////////////////////////////////////
// PROCESSING CODE HERE
//
//
// Per ogni pixel dell'immagine sinistra, cercare il suo corrispondente nell'immagine destra utilizzando la metrica SAD (Sum of Absolute Differences)
//
// Il corrispondente cercato e' il pixel sull'immagine destra tale per cui la somma delle differenze in valore assoluto di una finestra 5x5,
// centrata intorno ai relativi pixel destra/sinistra, e' minima
//
// In altre parole, per ogni pixel dell'immagine sx, confrontare il suo vicinato 5x5 con il corrispondente vicinato 5x5 di un pixel sulla destra.
//
// HINTS:
// - le immagini sono rettificate, quindi il corrispondente a destra su quale riga si trova?
// - dato un pixel (riga_l,colonna_l) sull'immagine sinistra, il corrispondente sulla destra (riga_r, colonna_r) si puo' trovare unicamente da uno specifico lato:
//   colonna_l < colonna_r? Oppure colonna_l > colonna_r? Quale dei due?
// - consideriamo spostamenti lungo la righa di massimo 128 colonne (cioe' disparita' massima 128)
//
//
// HINTS: si puo' fare con 5 cicli innestati
//
//
// REFERENCE: Lezioni 11_StereoMatching e 13_Feature2
//
void mySAD_Disparity(const cv::Mat & left_image, const cv::Mat & right_image, int radius, cv::Mat & out)
{
	int ww = radius;
	int wh = radius;

	out = cv::Mat(left_image.rows, left_image.cols, CV_32FC1, cv::Scalar(0));

	for(int i = wh; i <  left_image.rows - wh; ++i)
	{
		for(int j = ww; j <  left_image.cols - ww; ++j)
		{
			float disp=0;
			int min_sum = std::numeric_limits<int>::max();     //valore iniziale grande per la ricerca del minimo
			int min_sum_2nd = std::numeric_limits<int>::max();
			for(int d = 1;d < 128; ++d)
			{
				if(j-d >= ww) //verifico di non uscire dall'immagine destra
				{
					int sum = 0;
					for(int k = -wh; k<wh+1; ++k) //righe della finestra
						for(int v = -ww; v<ww+1; ++v) //colonne della finestra
							sum += std::abs( int(*(left_image.data + (i+k)*left_image.cols + j + v)) - int(*(right_image.data + (i+k)*left_image.cols + j + v - d)));

					if(sum < min_sum) //ho trovato un nuovo minimo?
					{
						min_sum_2nd = min_sum;   //imposto il secondo minimo uguale al minimo attuale

						min_sum = sum;           //aggiorno il minimo attuale
						disp = d;                //aggiorno la disparita' attuale
					}
					else
						if(sum < min_sum_2nd)        //questo mi serve se il minimo e' subito il primo elemento
							min_sum_2nd = sum;
				}
			}

			//ES1
			//
			out.at<float>(i,j) = disp;

			//ES2
			//
			//verifichiamo se ho trovato una disparita' poco significativa
			if(min_sum == std::numeric_limits<int>::max()          || //non ho trovato nessun minimo
					min_sum_2nd == std::numeric_limits<int>::max() || //non ho trovato nessun secondo minimo
					float(min_sum)/float(min_sum_2nd) > 0.8)          //il minimo e' troppo vicino al secondo minimo
				out.at<float>(i,j) = 0;                               //allora imposto la disparita' a 0 (o un altro valore convenzionale)

		}
	}
}
///////////////////////////////////////////


int main(int argc, char **argv)
{
	int frame_number = 0;
	char left_frame_name[256];
	char right_frame_name[256];
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
		if(args.left_image_name.find('%') != std::string::npos)
		{
			sprintf(left_frame_name,(const char*)(args.left_image_name.c_str()),frame_number);
			sprintf(right_frame_name,(const char*)(args.right_image_name.c_str()),frame_number);
		}
		else //single frame case
		{
			sprintf(left_frame_name,"%s",args.left_image_name.c_str());
			sprintf(right_frame_name,"%s",args.right_image_name.c_str());
		}

		//opening left file
		std::cout<<"Opening "<<left_frame_name<<std::endl;
		cv::Mat left_image = cv::imread(left_frame_name, CV_8UC1);
		if(left_image.empty())
		{
			std::cout<<"Unable to open "<<left_frame_name<<std::endl;
			return 1;
		}

		//opening right file
		std::cout<<"Opening "<<right_frame_name<<std::endl;
		cv::Mat right_image = cv::imread(right_frame_name, CV_8UC1);
		if(right_image.empty())
		{
			std::cout<<"Unable to open "<<right_frame_name<<std::endl;
			return 1;
		}


		////////////////////////////////////////////////////////////////////////////////////////////////////
		// PROCESSING

		//immagine di disparita' di output
		cv::Mat imgDisparity32FC1;

		//CHIAMATA ALLA VOSTRA FUNZIONE
		mySAD_Disparity(left_image, right_image, 2, imgDisparity32FC1);

		//visualizzazione
		cv::Mat adjMap;
		double minVal; double maxVal;

		cv::minMaxLoc(imgDisparity32FC1, &minVal, &maxVal);
		std::cout<<"max min mySAD "<<maxVal<<" "<<minVal<<std::endl;
		cv::convertScaleAbs(imgDisparity32FC1, adjMap, 255 / (maxVal-minVal));
		cv::namedWindow("mySAD", cv::WINDOW_NORMAL);
		cv::imshow("mySAD", adjMap);
		///////////////////////////////////////////////////////////////////////////////////////////////////////


#ifdef USE_SGM
		/// Esempio SGM di OpenCv
		//
		//  Non necessario per l'esame
		testOpenCvSGM(left_image, right_image);
#endif

		//display image
		cv::namedWindow("left image", cv::WINDOW_NORMAL);
		cv::imshow("left image", left_image);
		cv::namedWindow("right image", cv::WINDOW_NORMAL);
		cv::imshow("right image", right_image);

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

#ifdef USE_SGM
void testOpenCvSGM(const cv::Mat & left_image, const cv::Mat & right_image)
{
	////////////////////////////////////////////////////////////////////////
	///
	/// Esempio di disparita' ottenuta tramite metodo Semi-Global-Matching (SGM)
	///
	/// E' lo stato dell'arte per quanto riguarda il calcolo della disparita' in tempo reale
	///
	/// Esistono altri metodi piu' accurati, ma piu' lenti
	///

	int kernel_size = 5, number_of_disparities = 128, P1 = 100, P2 = 1000;
	int binary_descriptor_type = 0;
	cv::Mat imgDisparity16U(left_image.rows, left_image.cols, CV_16U, cv::Scalar(0));
	cv::Mat imgDisparity32FC1_SGM(left_image.rows, left_image.cols, CV_32FC1, cv::Scalar(0));

	// we set the corresponding parameters
	cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm = cv::stereo::StereoBinarySGBM::create(1, number_of_disparities, kernel_size);

	// setting the penalties for sgbm
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setMinDisparity(1);
	sgbm->setUniquenessRatio(5);
	sgbm->setSpeckleWindowSize(400);
	sgbm->setSpeckleRange(200);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setBinaryKernelType(binary_descriptor_type);
	sgbm->setSpekleRemovalTechnique(cv::stereo::CV_SPECKLE_REMOVAL_AVG_ALGORITHM);
	sgbm->setSubPixelInterpolationMethod(cv::stereo::CV_SIMETRICV_INTERPOLATION);
	sgbm->compute(left_image, right_image, imgDisparity16U);

	imgDisparity16U.convertTo(imgDisparity32FC1_SGM, CV_32FC1);

	cv::Mat adjMap, falseColorsMap;
	double minVal; double maxVal;

	//opencv restituisce la dispartia' in fixed point, con 4 bit per la parte frazionaria
	imgDisparity16U/=16;
	imgDisparity32FC1_SGM/=16.0;

	cv::minMaxLoc(imgDisparity32FC1_SGM, &minVal, &maxVal);
	std::cout<<"max min SGM "<<maxVal<<" "<<minVal<<std::endl;
	cv::convertScaleAbs(imgDisparity16U, adjMap, 255 / (maxVal-minVal));
	cv::namedWindow("SGM", cv::WINDOW_NORMAL);
	cv::imshow("SGM", adjMap);

	cv::waitKey(1000);

	// PUNTI 3D
	//
	// calcoliamo la nuvola di punti 3D a partire dalla disparita'
	//
	//
	// Abbiamo bisogno di alcune informazioni sul setup fisico delle telecamere:
	//
	// baseline     -> distanza tra le camere destra e sinistra, in metri
	// focal_lenght -> lunghezza focale in pixel
	// u0, v0       -> centro dell'immagine, principal point
	//
	// http://vision.middlebury.edu/stereo/data/scenes2005/
	//
	// Si veda la Lezione 11
	//
	float baseline = 0.16;          // distanza tra le due camere in metri
	float focal_lenght = 1131.6;    // lunghezza focale in pixe;
	float u0 = left_image.cols/2.0; // principal point
	float v0 = left_image.rows/2.0; // principal point
	std::vector<cv::Point3f> punti3d;
	for(int i = kernel_size;i <  left_image.rows-kernel_size; ++i)
	{
		for(int j = kernel_size;j <  left_image.cols-kernel_size; ++j)
		{
			cv::Point3f punto;
			float disp = imgDisparity32FC1_SGM.at<float>(i,j);

			// verifichiamo che la disparita' abbia senso
			if(disp>1)
			{
				//disp+=200.0/3; //http://vision.middlebury.edu/stereo/data/scenes2005/

				//Lezione 11
				punto.y = (i-v0)*baseline/disp;
				punto.x = (j-u0)*baseline/disp;
				punto.z = baseline*focal_lenght/disp;
				punti3d.push_back(punto);
			}
		}
	}

    // 3d visualization
	//
	// imposto la camera virtuale con gli stessi parametri intrinseci della sinistra (==destra)
    cv::viz::Camera camerap(focal_lenght, focal_lenght, u0, v0, cv::Size(left_image.cols, left_image.rows));
    cv::viz::Viz3d win("3D cloud");
    win.setCamera(camerap);

	// il punto di vista sara' quello della camera sinistra, spostato indietro di 5m ed in alto di 0.5m
    cv::Mat world_t_cam = cv::Mat(1,3,CV_32F);
    world_t_cam.at<float>(0,0) =  0.0;
    world_t_cam.at<float>(0,1) = -0.5;
    world_t_cam.at<float>(0,2) = -5.0;
    cv::Affine3f affine = cv::Affine3f(cv::Mat::eye(3,3, CV_32F), world_t_cam);
    win.setViewerPose(affine);

    // disegno la camera
    cv::viz::WCameraPosition cpw_frustum(cv::Vec2f(atan2(left_image.cols/2, focal_lenght), atan2(left_image.rows/2, focal_lenght))); // Camera frustum
    win.showWidget("cam_frustum", cpw_frustum);

    //disegno il sistema di riferimento, che conincide con la camera in questo caso
    win.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    //disegno i punti 3D
    win.showWidget("3D cloud", cv::viz::WCloud(punti3d));

    std::cout << "Press q to exit" << std::endl;
    win.spin();
	////////////////////////////////////////////////////////////////////////
}
#endif
