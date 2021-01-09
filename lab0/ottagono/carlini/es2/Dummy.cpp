/**
 * \file dummy.cpp
 * \author VisLab (vislab@ce.unipr.it)
 * \date 2016-09-21
 */

#include "Dummy.h"

#include <stdio.h>
#include <iostream>

void Dummy::On_Execute()
{
  cv::namedWindow("Lena",CV_WINDOW_NORMAL);
  cv::namedWindow("Output",CV_WINDOW_NORMAL);

  cv::Mat input = cv::imread(getFrameFromCamera(),CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat output(input.rows, input.cols, CV_8UC1);

  /***************************/
  //CODE HERE

  for(int u = 0; u<input.rows; ++u)
  {
	  for(int v = 0; v<input.cols; ++v)
	  {
		  int iu = (u + input.rows/2)%input.rows;
		  int iv = (v + input.cols/2)%input.cols;

		  output.data[u*input.cols + v] = input.data[iu*input.cols + iv];
	  }
  }


  /***************************/

  cv::imshow("Lena", input);
  cv::imshow("Output", output);
}
