#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void conv(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& output) {
	using namespace cv;
	
	assert(kernel.type() == CV_32FC1);
	assert(image.type() == CV_8UC1);
	
	output.create(image.rows-kernel.rows+1, image.cols-kernel.cols+1, image.type());
	
	for(int ii = 0; ii < output.rows; ++ii) {
		for(int jj = 0; jj < output.cols; ++jj) {
			//for every valid position
			float result = 0.0;
			for(int kk = 0; kk < kernel.rows; ++kk) {
				for(int ll = 0; ll < kernel.cols; ++ll) {
					float* kelement = (float*)(kernel.data+(kk*kernel.cols+ll)*kernel.elemSize());
					unsigned char* ielement = image.data+((kk+ii)*image.cols + ll+jj)*image.elemSize();
					
					result += (*kelement) * (*ielement);
				}
			}
			
			//clip result
			if(result > 255) result = 255;
			if(result < 0) result = 0;
			
			//write the result in the output
			output.data[(ii*output.cols+jj)*output.elemSize()] = (unsigned char)result;
		}
	}
}

void conv_sign(const cv::Mat& kernel, const cv::Mat& in, cv::Mat& out) {
    int rows = in.rows;
    int cols = in.cols;
    int kr = std::ceil((kernel.rows - 1) * 0.5);
    int kc = std::ceil((kernel.cols - 1) * 0.5);

    out.create(in.rows, in.cols, CV_32FC1);

    for (int c = 0; c < in.cols; ++c) {
        for (int r = 0; r < in.rows; ++r) {
            float conv_val = 0;
            for (int cc = -kc; cc <= kc; ++cc) {
                for (int rr = -kr; rr <= kr; ++rr) {
                    if (c + cc >= 0 && c + cc < cols && r + rr >= 0 && r + rr < rows &&
                            kc + cc >= 0 && kc + cc < kernel.cols && kr + rr >= 0 && kr + rr < kernel.rows) {
                        float* kval = (float*) (kernel.data + ((rr + kr) * kernel.cols + (cc + kc)) * kernel.elemSize());
                        uint8_t* inval = (uint8_t*) (in.data + ((rr + r) * in.cols + (cc + c)) * in.elemSize());
                        conv_val += (*kval) * (*inval);
                    }
                }
            }
            *(float *)(out.data + (r * out.cols + c) * out.elemSize()) = conv_val;
        }
    }
}

void gaussian_kernel(cv::Mat& kernel, unsigned int size, float sigma) {
    kernel.create(size, size, CV_32FC1);
    unsigned int hsize = (size - 1) * 0.5;
    float k = 1 / (2 * M_PI * std::pow(sigma, 2));
    float sum = 0.0;
    for (unsigned int r = 0; r < size; ++r) {
        for (unsigned int c = 0; c < size; ++c) {
			float rr = r - hsize;
			float cc = c - hsize;
            float k_val = k * std::exp(-(rr * rr + cc * cc) / (2 * std::pow(sigma, 2)));            
	    	*((float*)(kernel.data + (r * kernel.cols + c) * kernel.elemSize())) = k_val;
            sum += k_val;
        }
    }

    for (unsigned int r = 0; r < size; ++r) {
        for (unsigned int c = 0; c < size; ++c) {
            float* kval = (float*) (kernel.data + (r * kernel.cols + c) * kernel.elemSize());
            float new_val = (*kval) / sum;            
	    	*((float*)(kernel.data + (r * kernel.cols + c) * kernel.elemSize())) = new_val;
        }
    }

}



void grad_magnitude(const cv::Mat& grad_x, const cv::Mat& grad_y, cv::Mat& magnitude) {
    magnitude.create(grad_x.rows, grad_x.cols, CV_32FC1);

    for (int r = 0; r < magnitude.rows; ++r) {
        for (int c = 0; c < magnitude.cols; ++c) {
            float* xval = (float*)(grad_x.data + (r * grad_x.cols + c) * grad_x.elemSize());
            float* yval = (float*)(grad_y.data + (r * grad_y.cols + c) * grad_y.elemSize());
            float magn_val =
            	std::sqrt(std::pow(*xval, 2) + std::pow(*yval, 2));

            *(float *)(magnitude.data + (r * magnitude.cols + c) * magnitude.elemSize()) = magn_val;
        }
    }
}
void grad_phase(const cv::Mat& grad_x, const cv::Mat& grad_y, cv::Mat& phase) {
    phase.create(grad_x.rows, grad_x.cols, CV_32FC1);

    for (int r = 0; r < phase.rows; ++r) {
        for (int c = 0; c < phase.cols; ++c) {
        	//uint8_t* xval = (uint8_t*)(grad_x.data + (r * grad_x.cols + c) * grad_x.elemSize());
        	//uint8_t* yval = (uint8_t*)(grad_y.data + (r * grad_y.cols + c) * grad_y.elemSize());

        	float* xval = (float*)(grad_x.data + (r * grad_x.cols + c) * grad_x.elemSize());
        	float* yval = (float*)(grad_y.data + (r * grad_y.cols + c) * grad_y.elemSize());


            float ph_val = std::atan2((float) (*xval) , (float) (*yval));
			*((float*)(phase.data + (r * phase.cols + c) * phase.elemSize())) = ph_val;
        }
    }
}



void display_phase(const cv::Mat& phase, const cv::Mat& magnitude, uint8_t threshold, cv::Mat& image) {
    image.create(phase.rows, phase.cols, CV_8UC3);
    image.zeros(phase.rows, phase.cols, CV_8UC3);
    for (int r = 0; r < phase.rows; ++r) {
        for (int c = 0; c < phase.cols; ++c) {
            uint8_t* magn_val = (uint8_t*) (magnitude.data + (r * magnitude.cols + c) * magnitude.elemSize());

            if (*magn_val > threshold) {
				float* ph_val = (float*) (phase.data + (r * phase.cols + c) * phase.elemSize());
                if (*ph_val > 0.0) {
                    image.data[(r * image.cols + c) * image.elemSize() + 0] = 0;
                    image.data[(r * image.cols + c) * image.elemSize() + 1] = 255 - uint8_t(255 * ((*ph_val) / M_PI_2));
                    image.data[(r * image.cols + c) * image.elemSize() + 2] = 255;
                } else {
                    image.data[(r * image.cols + c) * image.elemSize() + 0] = 0;
                    image.data[(r * image.cols + c) * image.elemSize() + 1] = 255;
                    image.data[(r * image.cols + c) * image.elemSize() + 2] = 255 - uint8_t(255 * (-(*ph_val) / M_PI_2));
                }
            }
        }
    }
}


void binarize(const cv::Mat& in, cv::Mat& out, uint8_t threshold) {
    out.create(in.rows, in.cols, in.type());

    for (int r = 0; r < in.rows; ++r) {
        for (int c = 0; c < in.cols; ++c) {
            uint8_t* val = (uint8_t*) (in.data + (r * in.cols + c) * in.elemSize());
            if (*val > threshold)
                out.data[(r * out.cols + c) * out.elemSize()] = 255;
			else
            	out.data[(r * out.cols + c) * out.elemSize()] = 0;
        }
    }
}
