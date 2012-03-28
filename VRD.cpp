/*=================================================================
 * VRD.cpp - Compute Variance Ridge Detector on LAB images
 *
 * Input:   LAB-uint8 image, radius of edge detection
 * Output:  2D array of edge magnitudes
 *
 * Copyright 2012 Randolph Voorhies
 *	
 *=================================================================*/
#include "mex.h"
#include "vrd_sse.h"
#include <stdint.h>

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  if(nrhs != 2)
    mexErrMsgTxt("Inputs must be [imagearray, radius]");
  if(nlhs != 1)
    mexErrMsgTxt("Only one output provided");

  if(!(mxIsClass(prhs[0],"uint8") && mxGetNumberOfDimensions(prhs[0]) == 3))
  {
    char buffer[512];
    sprintf(buffer, "First argument must be a LAB-uint8 image. Type is [%s], nDims are [%d]", 
        mxGetClassName(prhs[0]), mxGetNumberOfDimensions(prhs[0]));
    mexErrMsgTxt(buffer);
  }

  // Get the input dimensions
  mwSize const * dims = mxGetDimensions(prhs[0]);

  // Get the desired radius
  int radius  = mxGetScalar(prhs[1]);

  // Get the input array
  uint8_t * mxInput = static_cast<uint8_t*>(mxGetData(prhs[0]));

  // Pack the input pixels into a LABx configuration
  float * const img = static_cast<float*>(calloc(dims[0]*dims[1]*4, sizeof(float)));
  int const npixels = dims[0] * dims[1];
  for(int i=0; i<3; ++i)
  {
    float *imgit = img + i;
    for(int j=0; j<npixels; ++j)
    {
      *imgit = *mxInput;
      mxInput += 1;
      imgit   += 4;
    }
  }

  // Create the output array
  mxArray *mxOutput = mxCreateNumericArray(2, &dims[0], mxSINGLE_CLASS, mxComplexity(0));

  // Do the VRDing
  vrd_sse(img, dims[0], dims[1], radius, static_cast<float*>(mxGetData(mxOutput)));


  //int numlabels;
  //SLIC SLICengine;
  //SLICengine.DoSuperpixelSegmentation_ForGivenK(reinterpret_cast<uint32_t*>(img),
  //    dims[0], dims[1], static_cast<int*>(mxGetData(mxOutput)), numlabels, numsuperpixels, 30);

  plhs[0] = mxOutput;

  free(img);
}
