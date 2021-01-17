// std
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>

// opencv
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

// eigen
#include <eigen3/Eigen/Core>

using namespace std;
using namespace cv;

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


/**
 * Zero padding
 * 
 * Aggiunge alla matrice input una cornice di zeri, di padding_rows righe e padding_rows colonne,
 * ed inserisce il risultato in una matrice output
 */
void zero_padding(const cv::Mat& input, int padding_rows, int padding_cols, cv::Mat& output) {
	output = cv::Mat::zeros(input.rows + 2*padding_rows, input.cols + 2*padding_cols, input.type());

	for(int v = padding_rows; v < output.rows-padding_rows; ++v) {
		for(int u = padding_cols; u < output.cols-padding_cols; ++u) {
			for(int k = 0; k < output.channels(); ++k) {	
				output.data[((v*output.cols + u)*output.channels() + k)*output.elemSize1()]
				= input.data[(((v-padding_rows)*input.cols + (u-padding_cols))*input.channels() + k)*input.elemSize1()];
			}
		}
	}
}

/**
 * Contrast stretching
 * 
 * Riporta i valori della matrice input (CV_32FC1) nell'intervallo [0; MAX_RANGE], 
 * e mette il risultato in una matrice di output di tipo type, il quale puo' essere:
 * - CV_32FC1 => la matrice in output resta di tipo invariato, e dunque ne vengono semplicemente "schiacciati" i valori nell'intervallo richiesto (utile prima di una sogliatura)
 * - CV_8UC1 => la matrice in output, oltre a subire uno stretching dei valori, subisce anche una conversione di tipo (utile prima di una imshow)
 * 
 */
void contrast_stretching(const cv::Mat& input, cv::Mat& output, int output_type, float MAX_RANGE = 255.0f) {
	double min_pixel, max_pixel;

	// funzione di OpenCV per la ricerca di minimo e massimo
	cv::minMaxLoc(input, &min_pixel, &max_pixel);
	
	// DEBUG
	// std::cout << "min: " << min_pixel << ", max: " << max_pixel << std::endl;

	// In generale, contrast_and_gain(r, c) = a*f(r, c) + b
	// contrast_stretching ne è un caso particolare in cui:
	// a = 255 / max(f) - min(f)
	// b = (255 * min(f)) / max(f) - min(f)
	float a = (float) (MAX_RANGE / (max_pixel - min_pixel));
	float b = (float) (-1 * ((MAX_RANGE * min_pixel) / (max_pixel - min_pixel)));
	
	output.create(input.rows, input.cols, output_type);

	for(int r = 0; r < input.rows; ++r) {
		for(int c = 0; c < input.cols; ++c) {
			for(int k = 0; k < input.channels(); ++k) {
				float pixel_input;
				
				// distinguo il modo in cui accedo alla matrice di input in base al suo tipo
                int input_type = input.type();
				if(input_type == CV_8UC1)
					pixel_input = input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()];
				else if(input_type == CV_32FC1)
					// nel caso di matrice float, devo castare correttamente il puntatore
					// per farlo, prendo l'indirizzo di memoria e lo casto in modo opportuno, dopodichè lo dereferenzio
					pixel_input = *((float*) &(input.data[((r*input.cols + c)*input.channels() + k)*input.elemSize1()]));

				float stretched_pixel_input = a*pixel_input + b;
				
				// distinguo il modo in cui accedo alla matrice di output in base al tipo
				if(output_type == CV_8UC1)
					output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()] = (uchar) stretched_pixel_input;
				
				else if(output_type == CV_32FC1)
					// nel caso di matrice float, devo castare correttamente il puntatore
					// per farlo, prendo l'indirizzo di memoria e lo casto in modo opportuno, dopodichè lo dereferenzio
					*((float*)(&output.data[((r*output.cols + c)*output.channels() + k)*output.elemSize1()])) = stretched_pixel_input;
			}
		}
	}
}

/**
 * Convoluzione float
 */
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	// inizialmente calcolo le dimensioni senza 
	int padding_rows = 0, padding_cols = 0;

	// Calcolo le dimensioni di output dopo aver applicato il kernel (formula tratta dalle slide di teoria)
	int out_tmp_rows = floor((image.rows + 2*padding_rows - kernel.rows)/stride + 1);
	int out_tmp_cols = floor((image.cols + 2*padding_cols - kernel.cols)/stride + 1);

	// uso la create() per definire l'immagine di output (poichè non inizializzata dal main)
	out.create(out_tmp_rows, out_tmp_cols, CV_32FC1);

	// padding per riottenere le dimensioni di input
	padding_rows = floor((image.rows - out_tmp_rows)/2);
	padding_cols = floor((image.cols - out_tmp_cols)/2);
	zero_padding(out.clone(), padding_rows, padding_cols, out);
	
	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, pesata coi corrispondenti valori del kernel
	// grazie a questo, calcolerò poi il risultato della convoluzione come somma di questi valori
	std::vector<float> convolution_window;

	for(int r = 0; r < image.rows; ++r) {
		for(int c = 0; c < image.cols; ++c) {
			// Effettuo i calcoli solamente se, tenuto conto di size e stride, non fuoriesco dall'immagine
			if((r+kernel.rows)*stride <= image.rows && (c+kernel.cols)*stride <= image.cols) {
				for(int k = 0; k < out.channels(); ++k) {

					// 2 cicli per analizzare l'attuale kernel
					for(int r_kernel = 0; r_kernel < kernel.rows; ++r_kernel) {
						for(int c_kernel = 0; c_kernel < kernel.cols; ++c_kernel) {
							// estraggo il pixel corrente sull'immagine
							float image_pixel = image.data[((stride*(r+r_kernel)*image.cols + stride*(c+c_kernel))*image.channels() + k)*image.elemSize1()];
								
							// estraggo il pixel corrente sul kernel (ricondandomi di castare correttamente il puntatore restituito)
							float kernel_pixel = *((float*) &kernel.data[((r_kernel*kernel.cols + c_kernel)*kernel.channels() + k)*kernel.elemSize1()]);
								
							// calcolo il valore corrente della convoluzione, e lo inserisco nel vector
							float current_pixel = image_pixel*kernel_pixel;	

							convolution_window.push_back(current_pixel);
						}
					}

					// sommo i valori del vector, con la funzione accumulate
					float sum_val = std::accumulate(convolution_window.begin(), convolution_window.end(), 0.0f);

					// svuoto il vector per l'iterazione successiva
					convolution_window.clear();

					// accedo al pixel di output partendo dal pixel corrente nell'input, e sommando il padding necessario
					*((float*) &out.data[(((r+padding_rows)*out.cols + (c+padding_cols))*out.channels() + k)*out.elemSize1()]) = sum_val;
				}
			}
		}
	}
}

/**
 * Convoluzione intera
 */
void conv(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	// Richiamo la convoluzione float, e successivamente riporto i valori in un range [0; 255] con un contrast stretching
	// convertendo già a CV_8UC1 per ottenere un'immagine pronta da passare a imshow 
	cv::Mat convfloat_out;
	convFloat(image, kernel, convfloat_out, stride);
	contrast_stretching(convfloat_out, out, CV_8UC1);
}

/**
 * Derivata x con gradiente 1x5
 */
void grad_x(const cv::Mat& image, cv::Mat& derivative_x) {
	// applico una convoluzione di image con un filtro gradiente orizzontale
	float grad_x_data[] {-1, 0, 2, 0, -1};

	cv::Mat kernel_grad_x(1, 5, CV_32FC1, grad_x_data);
  
	// applico le convoluzioni
	conv(image, kernel_grad_x, derivative_x);
}

/**
 * Binarizzo un'immagine di input data una certa soglia
 */
void binarize(const cv::Mat& input_img, int threshold, cv::Mat& output_img) {
	int max_intensity = 255;
	output_img.create(input_img.rows, input_img.cols, input_img.type());

	for(int v = 0; v < output_img.rows; ++v) {	
		for(int u = 0; u < output_img.cols; ++u) {
				if((int)input_img.data[(v*input_img.cols + u)] >= threshold)
					output_img.data[(v*output_img.cols + u)] = max_intensity;
				else output_img.data[(v*output_img.cols + u)] = 0;
			}
		}
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
// Cioe', ogni pixel dell'immagine output corrisponde ad un preciso punto del mondo,
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
void BEV(const cv::Mat & image, const CameraParams& params, cv::Mat & output) {
    output.create(400, 400, image.type());

	/*** Calcolo RT ***/
    // attenzione: nei parametri di calibrazione c'e' orientazione e posizione della camera rispetto al mondo, 
    // quindi la RT che otteniamo è a partire da quei punti camera in punti mondo
    // voglio fare il contrario, voglio che i punti mondo vengano convertiti in punti camera: lo faccio con inv()

    cv::Affine3f RT_inv = params.RT.inv();
    Eigen::Matrix<float, 4, 4> RT;
    RT << RT_inv.matrix(0,0), RT_inv.matrix(0,1), RT_inv.matrix(0,2), RT_inv.matrix(0,3), 
        RT_inv.matrix(1,0), RT_inv.matrix(1,1), RT_inv.matrix(1,2), RT_inv.matrix(1,3), 
        RT_inv.matrix(2,0), RT_inv.matrix(2,1), RT_inv.matrix(2,2), RT_inv.matrix(2,3),
        0,                  0,                  0,                  1;

    /*** Calcolo K ***/
    Eigen::Matrix<float, 3, 4> K;
    K << params.ku,         0, params.u0, 0,
                0, params.kv, params.v0, 0,
                0,         0,         1, 0;

    /*** Calcolo M = K*RT ***/
    Eigen::Matrix<float, 3, 4> M;
    M = K*RT;

    // Calcolo i punti sul mondo Pw, e per ciascuno calcolo p = M*Pw ==> Pw = M^-1*p
    // Limiti dell'area di interesse nel mondo:
	float x_world_max =  10.0f;
    float x_world_min = -10.0f;
    float z_world_min =   0.0f;
    float z_world_max =  20.0f;

	float width_max = std::fabs(x_world_max - x_world_min);
	float height_max = std::fabs(z_world_max - z_world_min);
    
    // zw/20 = zbev/400
    // ==> zw = zbev*20/400
    for(int r = 0; r < output.rows; ++r) {
    	for(int c = 0; c < output.cols; ++c) {
            // Pw = [x 0 z 1]^T, dove x e z sono costruite nel seguente modo:
            float x = x_world_max - c*(width_max/output.cols);
            float z = z_world_max - r*(height_max/output.rows);

            Eigen::Vector4f point_world(x, 0, z, 1);

            // Calcolo p = M*Pw
            Eigen::Vector3f point_bev_homogeneus = M*point_world;

            // Trasformo in euclidee
            float u = point_bev_homogeneus[0]/point_bev_homogeneus[2];
            float v = point_bev_homogeneus[1]/point_bev_homogeneus[2];
            
            // Assegno al corrente pixel di ouput, il valore dell'immagine nelle coordinate appena calcolate
            if (u > 0 && u < image.cols && v > 0 && v < image.rows) {
				output.at<uint8_t>(r, c) = image.at<uint8_t>(v, u);
        	}
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
void FindLines(const cv::Mat & bev, cv::Mat & output) {
	output = cv::Mat(400, 400, CV_8UC1, cv::Scalar(0));

    /**
     * YOUR CODE HERE:
     *
     * 1) Convoluzione con kernel (1,5) opportuno
     * 2) Binarizzazione
     */
    // 1)
    grad_x(bev, output);
    
    // 2)
    int threshold = 200;
    binarize(output.clone(), threshold, output);
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





