// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

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
 * Funzione per la lettura di punti da un file
 * I punti vengono poi inseriti in un vector points, ritornato per riferimento
 */
void LoadPoints(const std::string& filename, std::vector< cv::Point3f >& points) {
    std::ifstream file;
    file.open(filename.c_str());
    
    int size;
    file >> size;
    
    for (unsigned int i = 0; i < size; ++i) 
    {
        cv::Point3f point, point_out;
        file >> point.x >> point.y >> point.z;

        //from "VisLab-body-like" to typical "camera-like" reference
        point_out.z = point.x;
        point_out.y = -point.z;
        point_out.x = -point.y;
        points.push_back(point_out);
    }
    
    file.close();
}

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
 * Funzione per la trasformazione proiettiva prospettica:
 * Dato un vector di punti 3D, ne calcola i rispettivi punti 2D secondo la formula del pin-hole:
 * p' = M*Pw
 */
void Project(const std::vector< Point3f >& points, const CameraParams& params, std::vector< Point2f >& uv_points) {
    /*** Calcolo RT ***/
    // attenzione: nei parametri di calibrazione c'e' orientazione e posizione della camera rispetto al mondo, 
    // quindi la RT che otteniamo è a partire da quei punti camera in punti mondo
    // voglio fare il contrario, voglio che i punti mondo vengano convertiti in punti camera: lo faccio con inv()

    Affine3f RT_inv = params.RT.inv(); 
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

    // std::cout << "M" << std::endl << M << std::endl;
    
    /*** Per ogni Pw, calcolo p' = M*Pw, e inserisco p' nel vector di output ***/
    uv_points.resize(points.size()); // assumo di proiettare tutti i punti del mondo sul piano immagine
    for (unsigned int i = 0; i < points.size(); ++i) {
        Eigen::Vector4f point;
        point.x() = points[i].x;
        point.y() = points[i].y;
        point.z() = points[i].z;
        point.w() = 1.0;

        Eigen::Vector3f uv_point;
        uv_point = M * point;

        uv_points[i].x = uv_point.x() / uv_point.z();
        uv_points[i].y = uv_point.y() / uv_point.z();
    }
}

/**
 * Funzione che data un'immagine float e un vector di punti, 
 * colora di bianco tali punti sull'immagine
 */
void DrawPixels(const std::vector< cv::Point2f >& uv_points, cv::Mat& image) {
    for (unsigned int i = 0; i < uv_points.size(); ++i) {
        float u = uv_points[i].x;
        float v = uv_points[i].y;

        if (u > 0 && u < image.cols && v > 0 && v < image.rows) {
            image.at<float>(v,u) = 1.0f;
        }
    }
}

/**
 * Funzione per la conversione di un vector di punti ad una matrice
 */
void PointsToMat(const std::vector< cv::Point3f >& points, cv::Mat& mat)
{
    mat = cv::Mat(1, 3*points.size(), CV_32FC3);
    for (unsigned int i,j = 0; i < points.size(); ++i, j+=3)
    {
        mat.at<float>(j) = points[i].x;
        mat.at<float>(j+1) = points[i].y;
        mat.at<float>(j+2) = points[i].z;
    }
}
