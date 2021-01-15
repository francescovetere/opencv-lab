/**
 * \file dummy.cpp
 * \author VisLab (vislab@ce.unipr.it)
 * \date 2016-09-21
 */

#include "Dummy.h"

#include <stdio.h>
#include <iostream>

#include "utils.h"

//
// Nel caso proposto, cioe' un ottagono orientato con i lati paralleli agli assi righe/colonne,
// e' sufficiente controllare il segno dei gradienti
//
bool checkEdge(int lato, const cv::Mat & grad_x, const cv::Mat & grad_y, int u, int v)
{
	float eps = 0.001;

	float x = *((float *)(grad_x.data + (v*grad_x.cols + u)*grad_x.elemSize()));
	float y = *((float *)(grad_y.data + (v*grad_y.cols + u)*grad_y.elemSize()));

	switch (lato)
	{
	case 0:
		return (x < 0 && fabs(y) < eps);
	case 1:
		return (x < 0 && y > 0);
	case 2:
		return (fabs(x) < eps && y > 0);
	case 3:
		return (x > 0 && y > 0);
	case 4:
		return (x > 0 && fabs(y) < eps);
	case 5:
		return (x > 0 && y < 0);
	case 6:
		return (fabs(x) < eps && y < 0);
	case 7:
		return (x < 0 && y < 0);
	default:
		return false;
	}
}

void Dummy::On_Execute()
{
	cv::namedWindow("Ottagono",CV_WINDOW_NORMAL);
	cv::namedWindow("Lati",CV_WINDOW_NORMAL);

	cv::Mat input = cv::imread(getFrameFromCamera(),CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat lati(input.rows, input.cols, CV_8UC1);
	lati.setTo(255);

	/***************************/
	//CODE HERE
	cv::Mat in = input;

	cv::Mat sobel_x_kernel = (cv::Mat_<float>(3, 3) <<
			-1, -2, -1,
			0, 0, 0,
			1, 2, 1);
	cv::Mat sobel_y_kernel = (cv::Mat_<float>(3, 3) <<
			-1, 0, 1,
			-2, 0, 2,
			-1, 0, 1);

	//gradiente con segno
	cv::Mat grad_x, grad_y;
	conv_sign(sobel_x_kernel, in, grad_x);
	conv_sign(sobel_y_kernel, in, grad_y);

	//fase e modulo
	cv::Mat magnitude, phase;
	grad_magnitude(grad_x, grad_y, magnitude);
	grad_phase(grad_x, grad_y, phase);

	for(int u=0; u< phase.cols;++u)
	{
		for(int v=0; v< phase.rows;++v)
		{
			// es. matricola 235
			if(checkEdge(2, grad_x, grad_y, u, v) || checkEdge(3, grad_x, grad_y, u, v) || checkEdge(5, grad_x, grad_y, u, v))
			{
				for(int c = 0; c < lati.channels(); ++c)
					lati.data[(v*lati.cols + u)*lati.elemSize() + c] = 0;//image.data[(v*output.cols + u)*output.elemSize() + c];

			}

			///
			///
			/// soluzione piu' generale per forme geometriche CONVESSE
			///
			/// nel caso di un ottagono non orientato con i lati paralleli agli assi, il semplice controllo del segno
			/// del gradiente fallirebbe, in quanto resterebbero dei casi ambigui
			///
			/// la soluzione piu' generale sarebbe controllare il valore "esatto" della fase, per individuare lo specifico lato
			/// utilizzando la sua orientazione, che e' unica nel caso di una forma geometrica *convessa*
			///
			/// qui ad esempio controllo, con delle semplici soglie, se la fase e' 0 (lato 2), 45gradi (lato 3) o 135gradi (lato 5)
			///
			//			float * fase = (float *)(phase.data + (v*phase.cols + u)*phase.elemSize());
			//			float * magn_val = (float*) (magnitude.data + (v * magnitude.cols + u) * magnitude.elemSize());
			//			if(*magn_val>edge_threshold && ((*fase < 0.8 && *fase > 0.7) || (*fase < 0.1 && *fase > -0.1) || (*fase < 0.8+M_PI/2 && *fase > 0.7+M_PI/2)))
			//			{
			//				for(int c = 0; c < output.channels(); ++c)
			//					output.data[(v*output.cols + u)*output.elemSize() + c] = 0;//image.data[(v*output.cols + u)*output.elemSize() + c];
			//			}
		}
	}

	cv::Mat grad_x_norm, grad_x_norm_scaled;
	cv::normalize(grad_x, grad_x_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
	cv::convertScaleAbs(grad_x_norm, grad_x_norm_scaled );
	cv::imshow("sobel x", grad_x_norm_scaled);


	cv::Mat grad_y_norm, grad_y_norm_scaled;
	cv::normalize(grad_y, grad_y_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
	cv::convertScaleAbs(grad_y_norm, grad_y_norm_scaled );
	cv::imshow("sobel y", grad_y_norm_scaled);

	cv::imshow("sobel magn", magnitude);

	cv::Mat phase_norm, phase_norm_scaled;
	cv::normalize(phase, phase_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
	cv::convertScaleAbs(phase_norm, phase_norm_scaled);
	cv::imshow("sobel phase", phase_norm_scaled);


	/***************************/

	cv::imshow("Ottagono", input);
	cv::imshow("Lati", lati);
}
