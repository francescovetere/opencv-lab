//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


void myHarrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh)
{
  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   *
   *
   * E' ovviamente vietato utilizzare un detector di OpenCv....
   *
   */
  ////////////////////////////////////////////////////////
  /// HARRIS CORNER
  //

  

  // Disegnate tutti i risultati intermendi per capire se le cose funzionano
  //
  // Per la response di Harris:
  //    cv::Mat adjMap;
  //    cv::Mat falseColorsMap;
  //    double minr,maxr;
  //
  //    cv::minMaxLoc(response1(roi), &minr, &maxr);
  //    cv::convertScaleAbs(response1(roi), adjMap, 255 / (maxr-minr));
  //    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  //    cv::namedWindow("response1", cv::WINDOW_NORMAL);
  //    cv::imshow("response1", falseColorsMap);

  // HARRIS CORNER END
  ////////////////////////////////////////////////////////
}

void myFindHomographySVD(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, cv::Mat & H)
{
  cv::Mat A(points1.size()*2,9, CV_64FC1, cv::Scalar(0));

  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   *
   * Utilizzate la funzione:
   * cv::SVD::compute(A,D, U, Vt);
   *
   * In pratica dovete costruire la matrice A opportunamente e poi prendere l'ultima colonna di V
   *
   */

  // ricordatevi di normalizzare alla fine
  H/=H.at<double>(2,2);

  //std::cout<<"myH"<<std::endl<<H<<std::endl;
}

void myFindHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, const std::vector<cv::DMatch> & matches, int N, float epsilon, int sample_size, cv::Mat & H, std::vector<cv::DMatch> & matchesInlierBest)
{
  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   * Implementare il calcolo dell'omografia con un loop RANSAC
   *
   *
   * E' vietato utilizzare:
   * 		cv::findHomography(sample1, sample0, CV_RANSAC)
   *
   *
   *
   * Inizialmente utilizzare la findHomografy di OpenCV dentro al vostro loop RANSAC
   *
   *      cv::findHomography(sample1, sample0, 0)
   *
   *
   * Una volta verificato che il loop RANSAC funziona, sostituire la findHomography di OpenCV con la vostra
   *      cv::Mat HR;
   *      myFindHomographySVD( cv::Mat(sample[1]), cv::Mat(sample[0]), HR);
   *
   */
}


int main(int argc, char **argv) {

  if (argc < 4) 
  {
    std::cerr << "Usage prova <image_filename> <book_filename> <alternative_cover_filename>" << std::endl;
    return 0;
  }

  // images
  cv::Mat input, cover, newcover;

  // load image from file
  input = cv::imread(argv[1], CV_8UC1);
  if(input.empty())
  {
    std::cout<<"Error loading input image "<<argv[1]<<std::endl;
    return 1;
  }

  // load image from file
  cover = cv::imread(argv[2], CV_8UC1);
  if(cover.empty())
  {
    std::cout<<"Error loading book image "<<argv[2]<<std::endl;
    return 1;
  }

  ////////////////////////////////////////////////////////
  /// HARRIS CORNER
  //
  float alpha = 0.04;
  float harrisTh = 500000;    //da impostare in base alla propria implementazione!!!!!

  std::vector<cv::KeyPoint> keypoints0, keypoints1;

  // FASE 1
  //
  // Qui sotto trovate i corner di Harris di OpenCV
  //
  // Da commentare e sostituire con la propria implementazione
  //
  // {
  //   std::vector<cv::Point2f> corners;
  //   int maxCorners = 0;
  //   double qualityLevel = 0.01;
  //   double minDistance = 10;
  //   int blockSize = 3;
  //   bool useHarrisDetector = true;
  //   double k = 0.04;

  //   cv::goodFeaturesToTrack( input,corners,maxCorners,qualityLevel,minDistance,cv::noArray(),blockSize,useHarrisDetector,k ); // estrae strong feature (k -> alpha)
  //   std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints0), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} ); // applica funzione a range vector e memorizza in altro range 3->size del keypoint

  //   corners.clear();
  //   cv::goodFeaturesToTrack( cover, corners, maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, k );
  //   std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints1), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} );
  // }
  //
  //
  //
  // Abilitare il proprio detector una volta implementato
  //
  //
  myHarrisCornerDetector(input, keypoints0, alpha, harrisTh);
  myHarrisCornerDetector(cover, keypoints1, alpha, harrisTh);
  //
  //
  //


  std::cout<<"keypoints0 "<<keypoints0.size()<<std::endl;
  std::cout<<"keypoints1 "<<keypoints1.size()<<std::endl;
  //
  //
  ////////////////////////////////////////////////////////

  /* Questa parte non va toccata */
  ////////////////////////////////////////////////////////
  /// CALCOLO DESCRITTORI E MATCHES
  //
  int briThreshl=30;
  int briOctaves = 3;
  int briPatternScales = 1.0;
  cv::Mat descriptors0, descriptors1;

  //dichiariamo un estrattore di features di tipo BRISK
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
  //calcoliamo il descrittore di ogni keypoint
  extractor->compute(input, keypoints0, descriptors0);
  extractor->compute(cover, keypoints1, descriptors1);

  //associamo i descrittori tra me due immagini
  std::vector<std::vector<cv::DMatch> > matches;
  std::vector<cv::DMatch> matchesDraw;
  cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING); // brute force matcher (TODO aggiornare con .create()), usa hamming distance tra vettori
  //matcher.radiusMatch(descriptors0, descriptors1, matches, input.cols*2.0);
  matcher.match(descriptors0, descriptors1, matchesDraw);

  //copio i match dentro a dei semplici vettori oint2f
  std::vector<cv::Point2f> points[2];
  for(unsigned int i=0; i<matchesDraw.size(); ++i)
  {
    points[0].push_back(keypoints0.at(matchesDraw.at(i).queryIdx).pt);
    points[1].push_back(keypoints1.at(matchesDraw.at(i).trainIdx).pt);
  }
  ////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////
  // CALCOLO OMOGRAFIA
  //
  //
  // E' obbligatorio implementare RANSAC.
  //
  // Per testare i corner di Harris inizialmente potete utilizzare findHomography di opencv, che include gia' RANSAC
  //
  // Una volta che avete verificato che i corner funzionano, passate alla vostra implementazione di RANSAC
  //
  //
  cv::Mat H;                                  //omografia finale
  std::vector<cv::DMatch> matchesInliersBest; //match corrispondenti agli inliers trovati
  std::vector<cv::Point2f> corners_cover;     //coordinate dei vertici della cover sull'immagine di input
  bool have_match=false;                      //verra' messo a true in caso ti match

  //
  // Verifichiamo di avere almeno 4 inlier per costruire l'omografia
  //
  //
  if(points[0].size()>=4)
  {
    //
    // Soglie RANSAC
    //
    // Piuttosto critiche, da adattare in base alla propria implementazione
    //
    int N=50000;            //numero di iterazioni di RANSAC
    float epsilon = 3;      //distanza per il calcolo degli inliers


    // Dimensione del sample per RANSAC, quiesto e' fissato
    //
    int sample_size = 4;    //dimensione del sample di RANSAC

    //////////
    // FASE 2
    //
    //
    //
    // Inizialmente utilizzare questa chiamata OpenCV, che utilizza RANSAC, per verificare i vostri corner di Harris
    //
    //
    cv::Mat mask;
    H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), cv::RANSAC, 3, mask);
    for(std::size_t i=0;i<matchesDraw.size();++i)
      if(mask.at<uchar>(0,i) == 1) matchesInliersBest.push_back(matchesDraw[i]);
    //
    //
    //
    // Una volta che i vostri corner di Harris sono funzionanti, commentare il blocco sopra e abilitare la vostra myFindHomographyRansac
    //
    //myFindHomographyRansac(points[1], points[0], matchesDraw, N, epsilon, sample_size, H, matchesInliersBest);
    //
    //
    //

    std::cout<<std::endl<<"Risultati Ransac: "<<std::endl;
    std::cout<<"Num inliers / match totali  "<<matchesInliersBest.size()<<" / "<<matchesDraw.size()<<std::endl;

    std::cout<<"H"<<std::endl<<H<<std::endl;

    //
    // Facciamo un minimo di controllo sul numero di inlier trovati
    //
    //
    float match_kpoints_H_th = 0.1;
    if(matchesInliersBest.size() > matchesDraw.size()*match_kpoints_H_th)
    {
      std::cout<<"MATCH!"<<std::endl;
      have_match = true;


      // Calcoliamo i bordi della cover nell'immagine di input, partendo dai corrispondenti nell'immagine target
      //
      //
      cv::Mat p  = (cv::Mat_<double>(3, 1) << 0, 0, 1);
      cv::Mat pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << cover.cols-1, 0, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << cover.cols-1, cover.rows-1, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << 0,cover.rows-1, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }
    }
    else
    {
      std::cout<<"Pochi inliers! "<<matchesInliersBest.size()<<"/"<<matchesDraw.size()<<std::endl;
    }


  }
  else
  {
    std::cout<<"Pochi match! "<<points[0].size()<<"/"<<keypoints0.size()<<std::endl;
  }
  ////////////////////////////////////////////////////////

  ////////////////////////////////////////////
  /// WINDOWS
  cv::Mat inputKeypoints;
  cv::Mat coverKeypoints;
  cv::Mat outMatches;
  cv::Mat outInliers;

  cv::drawKeypoints(input, keypoints0, inputKeypoints);
  cv::drawKeypoints(cover, keypoints1, coverKeypoints);

  cv::drawMatches(input, keypoints0, cover, keypoints1, matchesDraw, outMatches);
  cv::drawMatches(input, keypoints0, cover, keypoints1, matchesInliersBest, outInliers);


  // se abbiamo un match, disegniamo sull'immagine di input i contorni della cover
  if(have_match)
  {
    for(unsigned int i = 0;i<corners_cover.size();++i)
    {
      cv::line(input, cv::Point(corners_cover[i].x , corners_cover[i].y ), cv::Point(corners_cover[(i+1)%corners_cover.size()].x , corners_cover[(i+1)%corners_cover.size()].y ), cv::Scalar(255), 2, 8, 0);
    }
  }

  cv::namedWindow("Input", cv::WINDOW_AUTOSIZE); // Noi dobbiamo fare passettino in più nel passo 5: prendere la cover e inserirla qui: doppio ciclo in cui spalmo la cover qui, usando H
  cv::imshow("Input", input);

  cv::namedWindow("BookCover", cv::WINDOW_AUTOSIZE);
  cv::imshow("BookCover", cover);

  cv::namedWindow("inputKeypoints", cv::WINDOW_AUTOSIZE);
  cv::imshow("inputKeypoints", inputKeypoints);

  cv::namedWindow("coverKeypoints", cv::WINDOW_AUTOSIZE);
  cv::imshow("coverKeypoints", coverKeypoints);

  cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE); // tutti i match, sia sensati che non
  cv::imshow("Matches", outMatches);

  cv::namedWindow("Matches Inliers", cv::WINDOW_AUTOSIZE); // solo i match sensati
  cv::imshow("Matches Inliers", outInliers);

  cv::waitKey();

  return 0;
}





