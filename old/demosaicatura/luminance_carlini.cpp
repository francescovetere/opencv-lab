cv::Mat mono(m.rows, m.cols, CV_8UC1);
  
  
  float blue = 0.0;
  float green = 0.0;
  float red = 0.0;
  float result = 0.0;
  
  for(int u=0; u<width-1; ++u)
    for (int v=0; v<height-1; ++v)
       {
	 if(u%2==0 && v%2==0) //colonna pari riga pari
	   {
	    blue = (float) m.data[(v*width+(u+1))*m.elemSize()];
	    green = ((float) m.data[(v*width+u)*m.elemSize()] + (float) m.data[((v+1)*width+(u+1))*m.elemSize()])/2;
	    red = (float) m.data[((v+1)*width+u)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result;
	   }
	    	    
	 if(u%2!=0 && v%2==0) // colonna dispari riga pari
	   {
	    blue = (float) m.data[(v*width+u)*m.elemSize()];
	    green = ((float) m.data[(v*width+u+1)*m.elemSize()] + (float) m.data[((v+1)*width+u)*m.elemSize()])/2;
	    red = (float) m.data[((v+1)*width+u+1)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result; 
	   }
	   
	  if(u%2==0 && v%2!=0) // colonna pari riga dispari
	   {
	    blue = (float) m.data[((v+1)*width+(u+1))*m.elemSize()];
	    green = ((float) m.data[(v*width+u+1)*m.elemSize()] + (float) m.data[((v+1)*width+u)*m.elemSize()])/2;
	    red = (float) m.data[(v*width+u)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result; 
	   } 
	   
	 if(u%2!=0 && v%2!=0) // colonna dispari riga dispari
	   {
	    blue = (float) m.data[((v+1)*width+u)*m.elemSize()];
	    green = ((float) m.data[(v*width+u)*m.elemSize()] + (float) m.data[((v+1)*width+(u+1))*m.elemSize()])/2;
	    red = (float) m.data[(v*width+u+1)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result;
	   } 
       }
     
   
   //ultima riga
   int v=height-1;
   for(int u=0; u<width-1; ++u)
      {
       if(u%2==0 && v%2==0) //colonna pari riga pari
	   {
	    blue = (float) m.data[(v*width+(u+1))*m.elemSize()];
	    green = ((float) m.data[(v*width+u)*m.elemSize()] + (float) m.data[((v-1)*width+(u+1))*m.elemSize()])/2;
	    red = (float) m.data[((v-1)*width+u)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result;
	   }
	    	    
	 if(u%2!=0 && v%2==0) // colonna dispari riga pari
	   {
	    blue = (float) m.data[(v*width+u)*m.elemSize()];
	    green = ((float) m.data[(v*width+u-1)*m.elemSize()] + (float) m.data[((v-1)*width+u)*m.elemSize()])/2;
	    red = (float) m.data[((v-1)*width+u-1)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result; 
	   }
	   
	  if(u%2==0 && v%2!=0) // colonna pari riga dispari
	   {
	    blue = (float) m.data[((v+1)*width+(u+1))*m.elemSize()];
	    green = ((float) m.data[(v*width+u+1)*m.elemSize()] + (float) m.data[((v+1)*width+u)*m.elemSize()])/2;
	    red = (float) m.data[(v*width+u)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result; 
	   } 
	   
	 if(u%2!=0 && v%2!=0) // colonna dispari riga dispari
	   {
	    blue = (float) m.data[((v+1)*width+u)*m.elemSize()];
	    green = ((float) m.data[(v*width+u)*m.elemSize()] + (float) m.data[((v+1)*width+(u+1))*m.elemSize()])/2;
	    red = (float) m.data[(v*width+u+1)*m.elemSize()];
	    result = 0.11*blue + 0.59*green + 0.3*red;
	    mono[(v*width+u)*mono.elemSize()] = (unsigned char) result;
	   } 
	 
   	
	
	//
        //
	


	cv::imshow("DEBAYER", mono);