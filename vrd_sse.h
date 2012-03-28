#include <stdint.h>

//! Run the Variance Ridge Detector on an input image
/*! This method simply chains together blurredVarianceSSE(), calculateGradientSSE(), and calculateRidgeSSE(), and is really the only
 *  method that users should need.
 *
 *  \param[in] inputImage a w*h*4 float array containing the LABX image
 *  \param[in] w The width of the input image
 *  \param[in] h The height of the input image
 *  \param[in] r The desired radius of the ridge detector (a smaller radius will detect finer edges). A good default is 3.
 *  \param[out] outputImage a pointer to an allocated w*h chunk of floats where the output edge map will be written */
void vrd_sse(float const * const inputImage, int const w, int const h, int const r, float * outputImage);

//! Calculate the blurred variance on an input image (Step 1 of VRD)
/*! \param[in] inputImage A w*h*4 float array containing the LABX image
 *  \param[in] w The width of the input image
 *  \param[in] h The height of the input image
 *  \param[in] r The desired blur radius
 *  \param[out] outputImage A pointer to an allocated w*h chunk of floats to be used as the output image */
void blurredVarianceSSE(float const * const inputImage, int const w, int const h, int const r, float * outputImage);

//! Calculate the gradient on an input image (Step 2 of VRD)
/*! \param[in] inputImage a w*h float array containing a grayscale image
 *  \param[in] w The width of the input image
 *  \param[in] h The height of the input image,
 *  \param[in] r The radius in which to calculate the gradient
 *  \param[out] gradX A pointer to an allocated w*h chunk of floats to be used as the horizontal gradient output
 *  \param[out] gradY A pointer to an allocated w*h chunk of floats to be used as the vertical gradient output */
void calculateGradientSSE(float const * const inputImage, int const w, int const h, int const r, float * gradX, float * gradY);

//! Calculate the ridge on a horizontal and vertical gradient (Step 3 of VRD)
/*! \param[in] gradX A w*h float array containing the horizontal gradient
 *  \param[in] gradY A w*h float array containing the vertical gradient
 *  \param[in] w The width of the images
 *  \param[in] h The height of the images
 *  \param[in] r The radius in which to calculate the ridge
 *  \param[out] ridgeImage A pointer to an allocated w*h chunk of floats to be used as the ridge output */
void calculateRidgeSSE(float const * const gradX, float const * const gradY, int const w, int const h, int const r, float * ridgeImage);


