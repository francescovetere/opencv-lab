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
  cv::namedWindow("App",cv::WINDOW_NORMAL);
  cv::imshow("App", cv::imread(getFrameFromCamera())); 
 
}
