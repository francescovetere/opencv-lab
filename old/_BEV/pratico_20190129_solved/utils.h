// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

struct CameraParams
{
    // size
    int w, h;
    // intrinsics
    float ku, kv;
    float u0, v0;
    // estrinsics
    cv::Affine3f RT;
    
};

void LoadPoints(const std::string& filename, std::vector< cv::Point3f >& points)
{
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

void PoseToAffine(float rx, float ry, float rz, float tx, float ty, float tz, cv::Affine3f& affine)
{
    cv::Mat world_RvecX_cam = cv::Mat(1,3,CV_32F);
    world_RvecX_cam.at<float>(0,0) = rx;
    world_RvecX_cam.at<float>(0,1) = 0.0;
    world_RvecX_cam.at<float>(0,2) = 0.0;
    cv::Mat world_Rx_cam;
    cv::Rodrigues(world_RvecX_cam, world_Rx_cam);
    
    cv::Mat world_RvecY_cam = cv::Mat(1,3,CV_32F);
    world_RvecY_cam.at<float>(0,0) = 0.0;
    world_RvecY_cam.at<float>(0,1) = ry;
    world_RvecY_cam.at<float>(0,2) = 0.0;
    cv::Mat world_Ry_cam;
    cv::Rodrigues(world_RvecY_cam, world_Ry_cam);
    
    cv::Mat world_RvecZ_cam = cv::Mat(1,3,CV_32F);
    world_RvecZ_cam.at<float>(0,0) = 0.0;
    world_RvecZ_cam.at<float>(0,1) = 0.0;
    world_RvecZ_cam.at<float>(0,2) = rz;
    cv::Mat world_Rz_cam;
    cv::Rodrigues(world_RvecZ_cam, world_Rz_cam);
    
    cv::Mat world_R_cam = world_Rx_cam*world_Ry_cam*world_Rz_cam;
    
    cv::Mat world_t_cam = cv::Mat(1,3,CV_32F);
    world_t_cam.at<float>(0,0) = tx;
    world_t_cam.at<float>(0,1) = ty;
    world_t_cam.at<float>(0,2) = tz;
    
    affine = cv::Affine3f(world_R_cam, world_t_cam);
}

void LoadCameraParams(const std::string& filename, CameraParams& params)
{
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

