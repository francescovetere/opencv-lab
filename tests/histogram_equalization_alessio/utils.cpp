/* ## Display an image in a new window with a given name ## */
void display(const Mat& image, const string name)
{
    namedWindow(name);
    imshow(name, image);
}

/* ## Apply a zero padding ## */
Mat zero_padding(const Mat& image, const int pad){

    Mat padded_row = Mat(image.rows + 2*pad, image.cols, image.type(), Scalar(0));
    image.copyTo(padded_row.rowRange(pad, image.rows + pad));

    Mat padded_both = Mat(padded_row.rows, padded_row.cols + 2*pad, image.type(), Scalar(0));
    padded_row.copyTo(padded_both.colRange(pad, padded_row.cols + pad));

    return padded_both;
}

/* ## Generate an intensity histogram ## */
void histogram(const Mat& image, int h[], const int levels)
{

    for (int i = 0; i < levels; i++)
        h[i] = 0;

    for (size_t i = 0; i < image.rows * image.cols * image.elemSize(); i += image.elemSize())
    {
        h[image.data[i]]++;
    }
}

/* ## Given yaw (vertical), pitch (forward) and roll (side) rotation generate R ## */
Eigen::Matrix3d generate_R(double yaw, double pitch, double roll){
    
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    return q.matrix();
}

/* ## Generate a translaction vector in x,y,z ## */
Eigen::Vector3d generate_t(double x, double y, double z){
   
    Eigen::MatrixXd translation(3, 1);
    translation << x, y, z;
    return translation;
}

/* ## Extract K,R,t and image size from a '.dat' file ## */
void load_params(string filename, Eigen::MatrixXd &K, Eigen::MatrixXd &R, Eigen::Vector3d &t, Size &image_size)
{
    std::ifstream file(filename);
    std::string line;

    float vals_k[3][2] = {0};

    for (int i = 0; i < 3; i++)
    {
        std::getline(file, line);
        std::stringstream linestream(line);
        linestream >> vals_k[i][0] >> vals_k[i][1];
    }

    image_size.width = vals_k[0][0];
    image_size.height = vals_k[0][1];

    K.resize(3, 3);
    K << vals_k[1][0], 0, vals_k[2][0], 0, vals_k[1][1], vals_k[2][1], 0, 0, 1;

    float r_t[2][3] = {0};

    for (int i = 0; i < 2; i++)
    {
        std::getline(file, line);
        std::stringstream linestream(line);
        linestream >> r_t[i][0] >> r_t[i][1] >> r_t[i][2];
    }

    R = generate_R(r_t[0][0], r_t[0][1], r_t[0][2]);
    t = generate_t(r_t[1][0], r_t[1][1], r_t[1][2]);
}

/* ## Convolution, [CV_8UC1] input, [CV_32FC1] output ## */
void convFloat(const Mat &image, const Mat &kernel, Mat &out, int stride)
{

    if (kernel.rows % 2 == 0 or kernel.cols % 2 == 0)
    {
        cout << "Pool size must be odd!" << endl;
        exit(1);
    }

    Mat padded;

    if(kernel.rows >= kernel.cols) padded = zero_padding(image, (kernel.rows - 1) / 2);
    else padded = zero_padding(image, (kernel.cols - 1) / 2);

     
    /* compute new size and initialize 'out' */
    int new_rows = floor(abs((image.rows - kernel.rows)) / stride + 1);
    int new_cols = floor(abs((image.cols - kernel.cols)) / stride + 1);
    out = Mat(new_rows, new_cols, CV_32F, Scalar(BLACK));
    
    /* index for identification of pixels in 'out' */
    int out_row = 0, 
    out_col = 0, 

    /* index for identification of pixels in 'kernel' */
    kernel_row,
    kernel_col,

    /* compute radius of the kernel */
    kernel_row_radius = (kernel.rows - 1) / 2, 
    kernel_col_radius = (kernel.cols - 1) / 2;

    /* this will store the sum for convolution */
    float weighted_sum = 0.0;

    /* with the first 2 'for' go through each pixel, considering the stride and pool size */
    for (int i = kernel_row_radius; i < image.rows - kernel_row_radius; i += stride)
    {

        for (int j = kernel_col_radius; j < image.cols - kernel_col_radius; j += stride )
        {

            weighted_sum = 0, kernel_row = 0, kernel_col = 0;

            /* compute image * kernel (convolution) */
            for (int k = i - kernel_row_radius; k <= i + kernel_row_radius; k++)
            {

                for (int l = j - kernel_col_radius; l <= j + kernel_col_radius; l ++)
                {
                    weighted_sum += image.ptr(k)[l] * kernel.ptr<float>(kernel_row)[kernel_col];
                    kernel_col ++;
                }

                /* update the row and column to identify the pixel of 'kernel' */
                kernel_row ++;
                kernel_col = 0;
            }

            /* save the convolution in a pixel of 'out', update 'out' column */
            out.ptr<float>(out_row)[out_col] = weighted_sum;
            out_col ++;
        }

        /* update the row and column to identify the pixel of 'out' */
        out_col = 0;
        out_row++;
    }

}

/* ## Convolution, [CV_8UC1] input, [CV_8UC1] output ## */
void conv(const Mat &image, const Mat &kernel, Mat &out, int stride)
{

    Mat conv_float;
    convFloat(image, kernel, conv_float, stride);
    out = Mat(conv_float.size(), CV_8UC1, Scalar(BLACK));

    float min = 0, max = 0;

    /* compute the max/min values of the float image */
    for (int i = 0; i < conv_float.rows; i++)
    {
        for (int j = 0; j < conv_float.cols; j++)
        {
            if(max < conv_float.ptr<float>(i)[j]) max = conv_float.ptr<float>(i)[j];
            if(min > conv_float.ptr<float>(i)[j]) min = conv_float.ptr<float>(i)[j];
        }

    }

    int index = 0;

    /* use the max/min computed to apply constant stretching */
    for (int i = 0; i < conv_float.rows; i++)
    {
        for (int j = 0; j < conv_float.cols; j++)
        {
            /* stretch and save the value into the new matrix */
            out.data[index] = floor(255*(conv_float.ptr<float>(i)[j] - min)/(max - min));
            index ++;
        }

    }

}

/* ## Generate a 1D gaussian kernel ## */
void GaussianKernel1D(float sigma, int radius, Mat &kernel)
{
    int size = radius * 2 + 1, index = 0;
    float sum = 0;
    kernel = Mat(1, size, CV_32F, Scalar(0.5));

    /* calculate the gaussian */
    for (float j = -radius; j <= radius; j++)
    {
        kernel.ptr<float>(0)[index] = 1 / (2 * M_PI * sigma * sigma) * exp(-1 * (j * j) / (2 * sigma * sigma));
        sum += kernel.ptr<float>(0)[index];
        index ++;
    }

    /* normalize the kernel */
    for (int i = 0; i < kernel.cols; i ++){
        kernel.ptr<float>(0)[i] /= sum;
    }
}

/* ## Apply Sobel in a particular direction ##
    dimension == -1 --->  both horizontal and vertical
    dimension == 0 --> horizontal
    dimension == 1 --> vertical  
    
    if reversed_direction is enabled then the gradient direction is inverted
*/
void Sobel_gradient(const Mat& image, Mat& gradient, const int dimension, const bool reversed_direction){

    float data_h[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    Mat kernel_h = Mat(3,3, CV_32F, data_h);
    float data_v[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    Mat kernel_v = Mat(3,3, CV_32F, data_v);

    if(reversed_direction){
        kernel_h *= -1;
        kernel_v *= -1;
    }

    Mat temp_step;

    switch(dimension)
    {
    case -1:
        conv(image, kernel_h, temp_step);
        conv(temp_step, kernel_v, gradient);
        break;
    
    case 0:
        conv(image, kernel_h, gradient);
        break;

    case 1:
        conv(image, kernel_v, gradient);
        break;

    default:
        image.copyTo(gradient);
        break;
    }
    

}

/* ## Apply Sobel, both horizontal and vertical ## */
void Sobel_full(const Mat& image, Mat& magnitude, Mat& orientation){

    float data[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    cv::Mat kernel_v = cv::Mat(3,3, CV_32F, data);

    /* get the vertical magnitude */
    cv::Mat magnitude_v;
    convFloat(image, kernel_v, magnitude_v, 1);

    /* get the horizontal magnitude */
    cv::Mat magnitude_h;
    convFloat(image, kernel_v.t(), magnitude_h, 1);

    /* initialize images */
    magnitude = cv::Mat(magnitude_v.size(), CV_32F, cv::Scalar(BLACK));
    orientation = cv::Mat(magnitude_v.size(), CV_32F, cv::Scalar(BLACK));

    float max_orientation = 0, min_orientation = 0;
    float max_magnitude = 0, min_magnitude = 0;

    /* compute the orientation and the total magnitude, saves also the min/max for later stretching */
    for(int i = 0; i < magnitude_v.rows; i++){
        for(int j = 0; j < magnitude_v.cols; j++){
            magnitude.ptr<float>(i)[j] = sqrtf(powf(magnitude_v.ptr<float>(i)[j],2) + powf(magnitude_h.ptr<float>(i)[j],2));
            orientation.ptr<float>(i)[j] = atan2f(magnitude_v.ptr<float>(i)[j], magnitude_h.ptr<float>(i)[j]);
            if(orientation.ptr<float>(i)[j] > max_orientation) max_orientation = orientation.ptr<float>(i)[j];
            if(orientation.ptr<float>(i)[j] < min_orientation) min_orientation = orientation.ptr<float>(i)[j];
            if(magnitude.ptr<float>(i)[j] > max_magnitude) max_magnitude = magnitude.ptr<float>(i)[j];
            if(magnitude.ptr<float>(i)[j] < min_magnitude) min_magnitude = magnitude.ptr<float>(i)[j];
        }
    }

    /* apply stretching for both images, we need a magnitude in [0,1] because imshow wants this range (or convert to CV_8U) */
    for(int i = 0; i < magnitude.rows; i++){
        for(int j = 0; j < magnitude.cols; j++){
            orientation.ptr<float>(i)[j] = 2*M_PI*(orientation.ptr<float>(i)[j] - min_orientation)/(max_orientation - min_orientation);
            magnitude.ptr<float>(i)[j] = (magnitude.ptr<float>(i)[j] - min_magnitude)/(max_magnitude - min_magnitude);
        }
    }

}