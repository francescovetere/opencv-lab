// giuseppe 
void addPadding(const cv::Mat& input, int dim_paddingRows, int dim_paddingCols, cv::Mat& output){
        
    int sizes[] = {input.rows + dim_paddingRows ,input.cols + dim_paddingCols};
    output.create(2,sizes,CV_8U);
        
    for(int v = 0; v < output.rows; v++){
        for(int u=0; u<output.cols; u++){
            if(v < dim_paddingRows || u < dim_paddingCols || v > (output.rows - dim_paddingRows) || u > (output.cols - dim_paddingCols)){
                output.data[(u+v*output.cols)] = 0;
            }
            else{
                output.data[(u+v*output.cols)] = input.data[(u+v*input.cols)];
            }    
        }
    }

 

    //cv::namedWindow("padded", cv::WINDOW_NORMAL);
    //cv::imshow("padded", output);

 

    std::cout<<"Padded size: "<<output.rows<<"x"<<output.cols<<std::endl;
}

void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1){

 

    std::cout<<"\n##### convFloat() function ######"<<std::endl;
    std::cout<<"-> Applying kernel "<<kernel.rows<<"x"<<kernel.cols<<std::endl;
    std::cout <<std::endl << kernel << std::endl;
    std::cout<<"\nActual size: "<<image.rows<<"x"<<image.cols<<std::endl;

 

    /* Aggiunta padding necessario per la convoluzione */ 
    cv::Mat tmp_padded;
    addPadding(image,kernel.rows-1,kernel.cols-1,tmp_padded);

 

    /* Creazione dell'immagine di output */
    int size1_new = floor(((image.rows + 2*(kernel.rows-1) - kernel.rows)/ stride) + 1);
    int size2_new = floor(((image.cols + 2*(kernel.cols-1) - kernel.cols)/ stride) + 1);
    int sizes[] = {size1_new-kernel.rows+1, size2_new-kernel.cols+1};
    out.create(2,sizes,CV_32F);

 

    //cv::namedWindow("interm", cv::WINDOW_NORMAL);
    //cv::imshow("interm", tmp_padded);

 

    std::cout<<"Output size: "<<out.rows<<"x"<<out.cols<<std::endl;

 

    for(int v = 0; v < out.rows; v++){
        float* Oi = out.ptr<float>(v);
            for(int u = 0; u < out.cols; u++){

 

                float tmp_val = 0;
                // Cicla con il kernel spostandosi di stride

 

                for(int i = 0; i < kernel.rows; i++)
                {
                    const float* Ki = kernel.ptr<float>(i);
                    for(int j = 0; j < kernel.cols; j++){

 

                        float val = tmp_padded.data[(stride*(v+i)*tmp_padded.cols + stride*(u+j))];
                        tmp_val += (Ki[j]*val);            
                    }
                }

 

                Oi[u] = tmp_val; // Assegna il valore ottenuto    
            }
        }        

 

}


// francesco
void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out, int stride = 1) {
	int padding_size = 0; // inizialmente lavoro senza padding, lo aggiungo al termine

	// Calcolo le dimensioni di output (formula tratta dalle slide di teoria)
	int out_rows = floor((image.rows + 2*padding_size - kernel.rows)/stride + 1);
	int out_cols = floor((image.cols + 2*padding_size - kernel.cols)/stride + 1);

	out.create(out_rows, out_cols, CV_32FC1);

	// definisco un vettore che conterrà ad ogni iterazione la maschera di pixel attuale, pesata coi corrispondenti valori del kernel
	// grazie a questo, calcolerò poi il risultato della convoluzione ad ogni iterazione
	// i valori saranno chiaramente float
	std::vector<float> convolution_window;
	
	// utilizzo due indici per capire dove mi trovo sull'immagine di output
	int current_out_r = -1;
	int current_out_c = -1;

	// 3 cicli per posizionarmi su ogni pixel dell'immagine di input, muovendomi di un passo pari alla stride
	for(int r = 0; r <= image.rows-std::max(kernel.rows, stride); r+=stride) {
		
		// ad ogni riga dell'immagine di input, incremento la riga corrente sull'output, riazzerando la colonna corrente sull'output!
		++current_out_r;
		current_out_c = -1;
		
		for(int c = 0; c <= image.cols-std::max(kernel.cols, stride); c+=stride) {

			// ad ogni colonna dell'immagine di input, incremento la colonna corrente sull'output
			++current_out_c;

			for(int k = 0; k < out.channels(); ++k) {

				// 2 cicli per analizzare ogni pixel dell'attuale kernel
				for(int r_kernel = 0; r_kernel < kernel.rows; ++r_kernel) {
					for(int c_kernel = 0; c_kernel < kernel.cols; ++c_kernel) {
				
						// eseguo la somma di prodotti tra pixel sull'immagine e pixel sul kernel
						float image_pixel = image.data[(((r+r_kernel)*image.cols + (c+c_kernel))*image.channels() + k)*image.elemSize1()];
						// il kernel è CV_32FC1: devo ricordarmi di castare il puntatore verso un float*, e solo successivamente prenderne l'elemento puntato
						float kernel_pixel = *((float*) &kernel.data[((r_kernel*kernel.cols + c_kernel)*kernel.channels() + k)*kernel.elemSize1()]);
						
						float current_pixel = image_pixel*kernel_pixel;	

						convolution_window.push_back(current_pixel);
					}
				}

				float sum_val = std::accumulate(convolution_window.begin(), convolution_window.end(), 0.0f);

				// svuoto il vector per la window successiva
				convolution_window.clear();

				// accedo all'immagine di output usando gli appositi indici, dichiarati prima dei for
				// l'output è CV_32FC1: devo ricordarmi di castare il puntatore verso un float*, e solo successivamente prenderne l'elemento puntato
				*((float*) &out.data[((current_out_r*out.cols + current_out_c)*out.channels() + k)*out.elemSize1()]) = sum_val;
			}
		}
	}
}