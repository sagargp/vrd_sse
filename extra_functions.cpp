vector<Image<PixGray<byte>>> calculateGradient(Image<PixGray<byte>> gray, int const rad)
{
	int w = gray.width();
	int h = gray.height();

	vector<Image<PixGray<byte>>> gradImg(2);
	gradImg[0] = Image<PixGray<byte>>(w, h);
	gradImg[1] = Image<PixGray<byte>>(w, h);

	// bbGray <-- boxBlur(gray);
	
	Eigen::VectorXf dx = Eigen::VectorXf::LinSpaced(NUM_GRADIENT_DIRECTIONS, 0, (NUM_GRADIENT_DIRECTIONS-1)*2*M_PI/NUM_GRADIENT_DIRECTIONS);
	Eigen::VectorXf dy = Eigen::VectorXf::LinSpaced(NUM_GRADIENT_DIRECTIONS, 0, (NUM_GRADIENT_DIRECTIONS-1)*2*M_PI/NUM_GRADIENT_DIRECTIONS);

	dx = dx.array().cos();
	dy = dy.array().sin();

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j < h; j++)
		{
			float sumX = 0.0;
			float sumY = 0.0;
			for (uint k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
			{
				int i1 = abs(i + rad*dx[k]);
				int j1 = abs(j + rad*dy[k]);

				int i2 = abs(i - rad*dx[k]);		
				int j2 = abs(j - rad*dy[k]);
				
				if(i1 >= w) i1 = 2*w - 2 - i1;
				if(j1 >= h) j1 = 2*h - 2 - j1;
				  
				if(i2 >= w) i2 = 2*w - 2 - i2;
				if(j2 >= h) j2 = 2*h - 2 - j2;
				
				//float val = varBbImg.at(i1,j1).val() - varBbImg.at(i2,j2).val();
				float val = gray.at(i1,j1).val() - gray.at(i2,j2).val();
				
				sumX +=  val * dx[k];
				sumY +=  val * dy[k]; 
			}
			gradImg[0](i, j) = sumX;
			gradImg[1](i, j) = sumY;
		}
	}
	return gradImg;
}

Image<PixGray<byte>> calculateRidge(vector<Image<PixGray<byte>>> const &gradImg, int const rad)
{
	int w = gradImg[0].width();
	int h = gradImg[0].height();
	
	Image<PixGray<byte>> ridgeImg(w, h);

	Eigen::VectorXf dx = Eigen::VectorXf::LinSpaced(NUM_GRADIENT_DIRECTIONS,0,(NUM_GRADIENT_DIRECTIONS-1)*2*M_PI/NUM_GRADIENT_DIRECTIONS);
	Eigen::VectorXf dy = Eigen::VectorXf::LinSpaced(NUM_GRADIENT_DIRECTIONS,0,(NUM_GRADIENT_DIRECTIONS-1)*2*M_PI/NUM_GRADIENT_DIRECTIONS);

	dx = dx.array().cos();
	dy = dy.array().sin();

	std::vector<std::vector<Eigen::MatrixXf> > dVin(NUM_GRADIENT_DIRECTIONS);

	// Look at neighboring pixels in a border defined by radius (rad) in the gradient image for evidence that supports the gradient orientation (k) at this pixel (i,j)
	// Only set the pixel (dVin) if there is positive evidence (threshold at 0)
	for (uint k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
	{
		dVin[k].resize(2);
		dVin[k][0] = Eigen::MatrixXf::Zero(w, h);
		dVin[k][1] = Eigen::MatrixXf::Zero(w, h);

		for (int i = 0; i < w; i++)
		{
			for (int j = 0; j < h; j++)
			{
				int ii = abs(i + rad*dx[k]);
				int jj = abs(j + rad*dy[k]); 

				if(ii >= w) ii = 2*w - 2 - ii;
				if(jj >= h) jj = 2*h - 2 - jj;
				  
				float vX = gradImg[0].at(ii,jj).val(); 
				float vY = gradImg[1].at(ii,jj).val();
				if((vX*dx[k] + vY*dy[k]) < 0.0)
				{
				  dVin[k][0](i,j) = vX;
				  dVin[k][1](i,j) = vY;
				}
			}
		}
	}

	vector<Eigen::MatrixXf> rDir(NUM_RIDGE_DIRECTIONS);
	for(uint k = 0; k < NUM_RIDGE_DIRECTIONS; k++)
	{
		rDir[k].setZero(w,h); 

		uint k2 = k + NUM_RIDGE_DIRECTIONS;

		// Calculate the dot product between the gradient on the positive side (k) and the negative side (k2) 
		Eigen::MatrixXf gVal = -(dVin[k][0].array()*dVin[k2][0].array() + dVin[k][1].array()*dVin[k2][1].array());
		// rDir is set to zero, so this operation with rectify gVal at zero
		rDir[k] = rDir[k].cwiseMax(gVal);
		// Take square root of direction
		rDir[k] = rDir[k].array().sqrt();    
	}

	// Next step is to find the maximum ridge response across all ridge directions
	// To do this, we will max pairs of ridge direction matrices and merge iteratively
	int endRes = NUM_RIDGE_DIRECTIONS;
	while(endRes>1)
	{
		int leftOver = 0;
		for(int i=0;i<endRes;i+=2)
		{
			if(i+1<endRes)
				rDir[i/2] = rDir[i].cwiseMax(rDir[i+1]);
			else
			{
				rDir[i/2] = rDir[i];
				leftOver = 1;
			}
		}
		endRes = (endRes >> 1) + leftOver;
	}
	return eigenMatrixToImage<float>(rDir[0]);
}

Image<PixGray<byte>> subtractGradImg(Image<PixGray<byte>> const &ridgeImg, vector<Image<PixGray<byte>>> const &grayImg)
{
	Image<PixGray<float>> quadEnergy = channel_transform(gradImg[0], gradImg[1], [](float const& c1, float const& c2)
		{
			return (sqrt(c1*c1 + c2*c2));
		});
	
	return ridgeImg - quadEnergy;
}

Image<PixGray<byte> > calculateNonMaxSuppression(Image<PixGray<byte> > bImg, shared_ptr<vector<Image<PixGray<byte> > > > ridgeDirectionNMS)
{
  int32 w = bImg.width();
  int32 h = bImg.height();

  if(ridgeDirectionNMS)
  {
    // Non-max suppressed values for each direction
    ridgeDirectionNMS->clear();
    for(uint k = 0; k < NUM_RIDGE_DIRECTIONS; k++)
      ridgeDirectionNMS->push_back( Image<PixGray<byte>>(w,h,ImageInitPolicy::Zeros));   
  }

  // Only create these coordinate sets once, every time after just grab the references
  vector<vector<Point2D<int32>>> sCoordsL;
  vector<vector<Point2D<int32>>> sCoordsR;
  vector<vector<Point2D<int32>>> cCoords;
  createNMSCoordList(sCoordsL,sCoordsR,cCoords);

  Image<PixGray<byte> > bImgNMS(w,h,ImageInitPolicy::Zeros);

  // go through each point in the interior first, with no branching
  int const wSize = BOUNDARY_STEP_SIZE+1;
  for(int i = wSize; i < w-wSize; i++)
  {
    for(int j = wSize; j < h-wSize; j++)
    {
      // get the value
      float val = bImg.at(i,j).val();
      Point2D<int32> cpt(i,j);
	  
      for(uint k = 0; k < NUM_RIDGE_DIRECTIONS; k++)
      {
        float totalC = 0.0; uint ctC = cCoords[k].size();
        for(uint cc = 0; cc < ctC; cc++)
    	{
    	  Point2D<int32> pt(cCoords[k][cc] + cpt);  
          totalC += bImg.at(pt).val();
    	}
	      
        float totalL = 0.0; uint ctL = sCoordsL[k].size();
        for(uint cl = 0; cl < ctL; cl++)
    	{
    	  Point2D<int32> pt(sCoordsL[k][cl] + cpt);  
          totalL += bImg.at(pt).val();
    	}
	      
        float totalR = 0.0; uint ctR = sCoordsR[k].size();
        for(uint cr = 0; cr < ctR; cr++)
    	{
    	  Point2D<int32> pt(sCoordsR[k][cr] + cpt);  
          totalR += bImg.at(pt).val();
    	}
	      
        if(totalC/ctC > totalR/ctR && totalC/ctC > totalL/ctL 
           && val > 0.0)  
    	{
    	  bImgNMS(i,j) = val;
          // break; // Can't break if we are calculating ridge directions
          if(ridgeDirectionNMS)            
            (*ridgeDirectionNMS)[k](i,j) =  totalC/ctC*2 - totalR/ctR - totalL/ctL;
          else
            break;
    	}
      }
      
    }
  }
  for(int i = 0; i < w; i++)
  {
    for(int j = 0; j < h; j++)
	{
      // Skip interior
      if(j>wSize && j<h-wSize && i>wSize && i<w-wSize)
      {
        j=h-wSize;
      }
	  // get the value
	  float val = bImg.at(i,j).val();
	  Point2D<int32> cpt(i,j);
	  
	  for(uint k = 0; k < NUM_RIDGE_DIRECTIONS; k++)
      {
        float totalC = 0.0; uint ctC = 0;
        for(uint cc = 0; cc < cCoords[k].size(); cc++)
		{
		  Point2D<int32> pt(cCoords[k][cc] + cpt);  
		  if(bImg.coordsOk(pt)) 
          {
            totalC += bImg.at(pt).val();
            ctC++;
          }
		}
	      
        float totalL = 0.0; uint ctL = 0;
        for(uint cl = 0; cl < sCoordsL[k].size(); cl++)
		{
		  Point2D<int32> pt(sCoordsL[k][cl] + cpt);  
		  if(bImg.coordsOk(pt)) 
          {
            totalL += bImg.at(pt).val(); ctL++;
          }
		}
	      
        float totalR = 0.0; uint ctR = 0;
        for(uint cr = 0; cr < sCoordsR[k].size(); cr++)
		{
		  Point2D<int32> pt(sCoordsR[k][cr] + cpt);  
		  if(bImg.coordsOk(pt)) 
          {
            totalR += bImg.at(pt).val(); ctR++;
          }
		}
	      
        if(totalC/ctC > totalR/ctR && totalC/ctC > totalL/ctL 
           && val > 0.0)  
		{
		  bImgNMS(i,j) = val;
          // break; // Can't break if we are calculating ridge directions
          if(ridgeDirectionNMS)
            (*ridgeDirectionNMS)[k](i,j) =  totalC/ctC*2 - totalR/ctR - totalL/ctL;
          else
            break;
		}
      }
	}
  }

  return bImgNMS;
}
