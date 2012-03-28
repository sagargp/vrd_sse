#include <stdint.h>

//! Calculate the blurred variance on an input image
/*! /param inputImage A w*h*4 float array containing the LABX image
 *  /param w The width of the input image
 *  /param h The height of the input image
 *  /param r The desired blur radius
 *  /outputImage A pointer to an allocated w*h chunk of floats to be used as the output image */
void blurredVarianceSSE(float const * const inputImage, int32_t const w, int32_t const h, int32_t const r, float * outputImage);

//! Calculate the gradient on an input image
/*! /param inputImage a w*h float array containing a grayscale image
 *  /param w The width of the input image
 *  /param h The height of the input image,
 *  /param r The radius in which to calculate the gradient
 *  /param gradX A pointer to an allocated w*h chunk of floats to be used as the horizontal gradient output
 *  /param gradY A pointer to an allocated w*h chunk of floats to be used as the vertical gradient output */
void calculateGradientSSE(float const * const inputImage, int const w, int const h, int const r, float * gradx, float * grady);

//! Calculate the ridge on a horizontal and vertical gradient
/*! /param gradX A w*h float array containing the horizontal gradient
 *  /param gradY A w*h float array containing the vertical gradient
 *  /param w The width of the images
 *  /param h The height of the images
 *  /param r The radius in which to calculate the ridge
 *  /param ridgeImage A pointer to an allocated w*h chunk of floats to be used as the ridge output */
void calculateRidgeSSE(float const * const gradX, float const * const gradY, int const w, int const h, int const r, float * ridgeImage);

