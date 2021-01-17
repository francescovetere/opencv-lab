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

using namespace cv;

std::string im_win_name = "Image";
std::string im_win_name_loop = "Image_loop";
std::string im_win_name_spiral = "Image_spiral";

int main(int argc, char **argv) {

  if (argc < 3) 
  {
    std::cerr << "Usage lab5_1 <points_filename> <camera_params_filename>" << std::endl;
    return 0;
  }

  // load point cloud from file
  std::vector<cv::Point3f> points;
  LoadPoints(argv[1], points);

  // load camera params from file
  CameraParams params;
  LoadCameraParams(argv[2], params);

#ifdef USE_OPENCVVIZ
  cv::Mat cloud;
  PointsToMat(points, cloud);

  // 3d visualization
  cv::viz::Viz3d win = Viz3D(params);

  win.showWidget("cloud", cv::viz::WCloud(cloud));

  std::cout << "Press q to exit" << std::endl;
  win.spin();
#endif

  // project 3d points on image
  std::vector<Point2f> uv_points;

  Project(points, params, uv_points);

  // draw image
  Mat image;
  image = Mat::zeros(params.h, params.w, CV_32FC1);
  DrawPixels(uv_points, image);

  namedWindow(im_win_name, WINDOW_AUTOSIZE );
  imshow(im_win_name, image);
  waitKey(0);

  // rotazione intorno all'edificio
  //
  // Provare ad implementare un loop di 16 posizioni sul piano XZ equidistanti dal baricentro dell'edificio, raggio 30m.
  // Per mantenere l'edificio al centro della visuale dobbiamo ruotare l'orientazione della camera di (2*M_PI/16) ad ogni step
  //
  //
  // centro del palazzo sul piano XZ, Y costante

  //baricentro dell'edificio ad altezza fissata
  float bx=0.0, by=-5.0, bz=0;

  for(unsigned int i=0;i<points.size();++i)
  {
    bx+=points[i].x;
    bz+=points[i].z;
  }

  bx/=points.size();
  bz/=points.size();

  std::cout<<"Building center "<<bx<<" "<<by<<" "<<bz<<std::endl;


  // L'idea e' di muoversi lungo una circonferenza di raggio radius e centrato nel baricentro dell'edificio
  //
  // Supponiamo di volerci spostare su 16 posizioni equidistanti, possiamo utilizzare una variable angle che da da
  // 0 a 2PI per step costanti (2*M_PI/16), e quindi calcolare la deltaX e deltaZ con seno e coseno.
  //
  float radius = 30.0;
  float angle=0.0;
  int steps = 16;
  namedWindow(im_win_name_loop, WINDOW_AUTOSIZE );
  int i = 0;
  while(1)
  {
    /**
     * YOUR CODE HERE:
     *
     * Calcolare i params opportuni per spostare il punto di vista lungo la circonferenza
     * mantenendo l'orientazione che punti verso l'edificio
     *
     * Utilizzare la funzione PoseToAffine fornita per calcolare i nuovi params
     */

    float rx, ry, rz, tx, ty, tz;

    tx = bx - sin(angle)*radius;
    ty = by; //non modifichiamo l'altezza
    tz = bz - cos(angle)*radius;

    rx = 0;      //nessuna rotazione intorno a X
    ry = angle;  //rotazione intorno ad Y per orientarci verso l'edificio
    rz = 0;      //nessuna rotazione intorno a Z

    PoseToAffine(rx, ry ,rz, tx, ty, tz, params.RT);

    // project 3d points on image
    uv_points.clear();
    Project(points, params, uv_points);

    // draw image
    Mat image_loop;
    image_loop = Mat::zeros(params.h, params.w, CV_32FC1);
    DrawPixels(uv_points, image_loop);

    imshow(im_win_name_loop, image_loop);
    waitKey(200);

    angle+=2*M_PI/float(steps);

    ++i;
  }

  return 0;
}
