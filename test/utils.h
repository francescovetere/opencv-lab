#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <string>

void zero_padding(const cv::Mat& input, int padding_size, cv::Mat& output) {
	for(int v = 0; v < output.rows; ++v) {
		for(int u = padding_size; u < output.cols; ++u) {
			for(int k = 0; k < output.channels(); ++k) {	
				if(v < padding_size || v >= output.rows - padding_size || u < padding_size || u >= output.cols - padding_size)
					output.data[(v*output.cols + u)*output.channels() + k] = 0;
				else
					output.data[(v*output.cols + u)*output.channels() + k]
					= input.data[(v*input.cols + u)*input.channels() + k];
			}
		}
	}
}
