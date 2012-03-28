#ifdef MAKE_SLOW_VERSION
#include <nrt/Eigen/Eigen.H>
#include <nrt/Eigen/EigenConversions.H>
#endif
#include <nrt/ImageProc/Math/RangeOps.H>
#include <nrt/ImageProc/IO/ImageSink/ImageSinks.H>
#include <nrt/ImageProc/IO/ImageSource/ImageSources.H>
#include <nrt/ImageProc/IO/ImageSource/ImageReaders/ImageReader.H>
#include <nrt/Core/Debugging/TimeProfiler.H>
#include <nrt/Core/Model/Manager.H>
#include <emmintrin.h> // sse3
#include <xmmintrin.h> // sse
#include <valgrind/callgrind.h>
#include "PixelTypes.H"
#include "vrd_sse.h"

using namespace nrt;
using namespace std;

#define NUM_GRADIENT_DIRECTIONS 8
#define NUM_RIDGE_DIRECTIONS    NUM_GRADIENT_DIRECTIONS/2
#define BOUNDARY_STEP_SIZE      NUM_GRADIENT_DIRECTIONS

#define TIMER_BLUR_SLOW         nrt::TimeProfiler<0>::instance()
#define TIMER_MAG_SLOW          nrt::TimeProfiler<1>::instance()
#define TIMER_GRAD_SLOW         nrt::TimeProfiler<2>::instance()
#define TIMER_RIDGE_SLOW        nrt::TimeProfiler<3>::instance()
#define TIMER_BLUR_SSE          nrt::TimeProfiler<4>::instance()
#define TIMER_MAG_SSE           nrt::TimeProfiler<5>::instance()
#define TIMER_GRAD_SSE          nrt::TimeProfiler<6>::instance()
#define TIMER_RIDGE_SSE         nrt::TimeProfiler<7>::instance()

void _print_reg(__m128 *r)
{
  float result[4];
  _mm_store_ps(result, *r);

  for (int i = 0; i < 4; i++)
    cout << result[i] << " "; 
}

inline float hadd_ps(__m128 *a)
{ 
  float data[4];
  _mm_store_ps(data, *a);

  return data[0] + data[1] + data[2] + data[3];
}

/********************
 * SSE code
 ********************/
namespace sse 
{

  Image<PixGray<float>> blurredVariance(Image<PixLABX<float>> const input, int const r)
  {
    Image<PixGray<float>, UniqueAccess> output(input.dims(), ImageInitPolicy::None);
    blurredVarianceSSE(input.pod_begin(), input.width(), input.height(), r, output.pod_begin());
    return Image<PixGray<float>>(output);
  }

  vector<Image<PixGray<float>>> calculateGradient(Image<PixGray<float>> input, int const r)
  {
    TIMER_GRAD_SSE.begin();

    int w = input.width();
    int h = input.height();

    vector<Image<PixGray<float>>> gradImg(2);
    gradImg[0] = Image<PixGray<float>>(w, h);
    gradImg[1] = Image<PixGray<float>>(w, h);

    calculateGradientSSE(input.pod_begin(), input.width(), input.height(), r, gradImg[0].pod_begin(), gradImg[1].pod_begin());

    TIMER_GRAD_SSE.end();
    return gradImg;

    //float const * const input_ptr = input.pod_begin();

    //float dx[NUM_GRADIENT_DIRECTIONS];
    //float dy[NUM_GRADIENT_DIRECTIONS];
    //float rdxdy[NUM_GRADIENT_DIRECTIONS*4];

    //float const pi2 = 2.0f*M_PI;
    //float const norm = 1/float(NUM_GRADIENT_DIRECTIONS);

    ///* pre-load r*dx and r*dy */
    //for (int k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
    //{
    //  float const idx = pi2*float(k)*norm;
    //  dx[k] = cos(idx);
    //  dy[k] = sin(idx);

    //  rdxdy[4*k + 0] = int(+r*dx[k]);
    //  rdxdy[4*k + 1] = int(+r*dy[k]);
    //  rdxdy[4*k + 2] = int(-r*dx[k]);
    //  rdxdy[4*k + 3] = int(-r*dy[k]);
    //}
    ///* now data starting at rdxdy[k] contains: (rdx[k], rdy[k], -rdx[k], -rdy[k]) */

    //__m128 _clamp = _mm_set_ps(h-2, w-2, h-2, w-2);
    //__m128 _1w1w = _mm_set_ps(w, 1, w, 1);
    //__m128 _signmask = _mm_set1_ps(-0.f);

    //float ij[4];

    //for (int j = 0; j < h; j++)
    //{
    //  for (int i = 0; i < w; i++)
    //  {
    //    float sumX = 0.0;
    //    float sumY = 0.0;

    //    for (int k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
    //    {
    //      /* load (r*dx, r*dy, -r*dx, -r*dy) into an SSE reg */
    //      __m128 _rdxdy = _mm_load_ps(&rdxdy[4*k]); 

    //      /* set _ij = (i, j, i, j) */
    //      __m128 _ij = _mm_set_ps(j, i, j, i);

    //      /* _ij = (i+rdx, j+rdy, i-rdx, j-rdx) */
    //      _ij = _mm_add_ps(_ij, _rdxdy);

    //      /* _ij = abs( (i+rdx, j+rdy, i-rdx, j-rdx) ) */
    //      _ij = _mm_andnot_ps(_signmask, _ij);

    //      /* clamp values inside _ij */
    //      _ij = _mm_min_ps(_clamp, _ij);

    //      /* reshape the coords in _ij into 1d */
    //      _ij = _mm_mul_ps(_ij, _1w1w);

    //      /* store _ij back into a regular float array: ij = j2 i2 j1 i1 */ 
    //      _mm_store_ps(ij, _ij);

    //      float val = input_ptr[int(ij[0] + ij[1])] - input_ptr[int(ij[2]+ij[3])];

    //      sumX += val * dx[k];
    //      sumY += val * dy[k];
    //    }
    //    gradImg[0](i, j) = sumX;
    //    gradImg[1](i, j) = sumY;
    //  }
    //}
    //TIMER_GRAD_SSE.end();
    //return gradImg;
  }

  Image<PixGray<float>> calculateRidge(vector<Image<PixGray<float>>> const &gradImg, int const r)
  {
    TIMER_RIDGE_SSE.begin();

    int w = gradImg[0].width();
    int h = gradImg[0].height();

    Image<PixGray<float>> ridgeImage(w,h);
    calculateRidgeSSE(gradImg[0].pod_begin(), gradImg[1].pod_begin(), w, h, r, ridgeImage.pod_begin());

    TIMER_RIDGE_SSE.end();
    return ridgeImage;
  }
}

#ifdef MAKE_SLOW_VERSION
/********************
 * Non-SSE code
 ********************/
namespace slow {
  Image<PixLAB<float>> blurredVarianceIntegralImage(Image<PixLAB<float>> const input, int const r)
  {
    TIMER_BLUR_SLOW.begin();
    int const w = input.width();
    int const h = input.height();

    Image<PixLAB<float>, UniqueAccess> integral(w, h, ImageInitPolicy::None);
    Image<PixLAB<float>, UniqueAccess> output(w, h, ImageInitPolicy::None);

    // set the first row and first column
    integral(0, 0) = input(0, 0);
    for (int x = 1; x < w; x++) integral(x, 0) = integral.at(x-1, 0) + input.at(x, 0);
    for (int y = 1; y < h; y++) integral(0, y) = integral.at(0, y-1) + input.at(0, y);

    // set every remaining pixel
    for (int y = 1; y < h; y++)
    {   
      for (int x = 1; x < w; x++)
      {
        PixLAB<float> const & left_i    = integral.at(x-1, y);
        PixLAB<float> const & top_i     = integral.at(x, y-1);
        PixLAB<float> const & topleft_i = integral.at(x-1, y-1);
        PixLAB<float> const & current   = input.at(x, y);

        PixLAB<float> & result = integral(x, y);
        result.channels[0] = left_i.channels[0] + top_i.channels[0] - topleft_i.channels[0] + current.channels[0];
        result.channels[1] = left_i.channels[1] + top_i.channels[1] - topleft_i.channels[1] + current.channels[1];
        result.channels[2] = left_i.channels[2] + top_i.channels[2] - topleft_i.channels[2] + current.channels[2];
      }
    }

    // compute the blur
    for (int y = 0; y < h; y++)
    {
      int const ytop = max(y-r, 0);
      int const ybot = min(y+r, h-1);

      for (int x = 0; x < w; x++)
      {
        int const xlef = max(x-r, 0);
        int const xrig = min(x+r, w-1);

        PixLAB<float> const & toplef = integral.at(xlef, ytop); 
        PixLAB<float> const & toprig = integral.at(xrig, ytop); 
        PixLAB<float> const & botrig = integral.at(xrig, ybot); 
        PixLAB<float> const & botlef = integral.at(xlef, ybot); 

        float const norm = (xrig-xlef)*(ybot-ytop);
        PixLAB<float> & result = output(x, y);
        result.channels[0] = (botrig.channels[0] - botlef.channels[0] - toprig.channels[0] + toplef.channels[0]) / norm; 
        result.channels[1] = (botrig.channels[1] - botlef.channels[1] - toprig.channels[1] + toplef.channels[1]) / norm; 
        result.channels[2] = (botrig.channels[2] - botlef.channels[2] - toprig.channels[2] + toplef.channels[2]) / norm; 
      }
    }
    TIMER_BLUR_SLOW.end();
    return Image<PixLAB<float>>(output);
  }

  Image<PixGray<float>> blurredVariance(Image<PixLAB<float>> const lab, int const r)
  {
    TIMER_BLUR_SLOW.begin();

    int boxSize = pow(2*r + 1, 2);
    int w = lab.width();
    int h = lab.height();

    Image<PixLAB<float>> output(w, h, ImageInitPolicy::None);

    float const * const labbegin = lab.pod_begin();

    for (int y = 0; y < h; y++)
    {
      for (int x = 0; x < w; x++)
      {
        float sqsum_r, sqsum_g, sqsum_b;
        float sum_r, sum_g, sum_b;

        sqsum_r = sum_r = 0;
        sqsum_g = sum_g = 0;
        sqsum_b = sum_b = 0;

        int ytop = max(y-r, 0);
        int ybot = min(y+r, h-1);
        int xlef = max(x-r, 0);
        int xrig = min(x+r, w-1);

        for (int j = ytop; j <= ybot; j++)
        {
          for (int i = xlef; i <= xrig; i++)
          {
            float const * const pixbegin = &labbegin[(j*w + i)*3];
            float const r = pixbegin[0];
            float const g = pixbegin[1];
            float const b = pixbegin[2];

            sum_r += r;
            sum_g += g;
            sum_b += b;
          }
        }
        float r = sum_r/boxSize;
        float g = sum_g/boxSize;
        float b = sum_b/boxSize;

        output(x,y) = PixLAB<float>(r,g,b);
      }
    }
   
    // this used to be in magnitudeLAB()
    Image<PixGray<float>> grayout(w, h);
    for (int x = 0; x < w; x++)
      for (int y = 0; y < h; y++)
        grayout(x, y) = sqrt(pow(output(x, y).l(), 2)
            + pow(output(x, y).a(), 2)
            + pow(output(x, y).b(), 2));

    TIMER_BLUR_SLOW.end();
    return grayout;
  }

  vector<Image<PixGray<float>>> calculateGradient(Image<PixGray<float>> gray, int const rad)
  {
    TIMER_GRAD_SLOW.begin();

    int w = gray.width();
    int h = gray.height();

    vector<Image<PixGray<float>>> gradImg(2);
    gradImg[0] = Image<PixGray<float>>(w, h);
    gradImg[1] = Image<PixGray<float>>(w, h);

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

          float val = gray.at(i1,j1).val() - gray.at(i2,j2).val();

          sumX +=  val * dx[k];
          sumY +=  val * dy[k]; 
        }
        gradImg[0](i, j) = sumX;
        gradImg[1](i, j) = sumY;
      }
    }

    TIMER_GRAD_SLOW.end();
    return gradImg;
  }

  Image<PixGray<float>> calculateRidge(vector<Image<PixGray<float>>> const &gradImg, int const rad)
  {
    TIMER_RIDGE_SLOW.begin();

    int w = gradImg[0].width();
    int h = gradImg[0].height();

    Image<PixGray<float>> ridgeImg(w, h);

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

    TIMER_RIDGE_SLOW.end();
    return eigenMatrixToImage<float>(rDir[0]);
  }
}
#endif

int main(int argc, const char** argv)
{
  // begin vrd transform
  //Image<PixRGB<float>> input = readImage(...);
  //Image<PixLAB<float>> lab(input);

  //varImg = standardDeviationLAB(lab, radius);
  //gradImgs = calculateGradient(varImg, radius);
  //ridgeImg = calculateRidge(gradImgs, radius);
  //boundaryImg = subtractGradImg(ridgeImg, gradImgs);
  //boundaryNMSImg = calculateNonMaxSuppression(boundaryImg);
  //return boundaryNMSImage;

  Manager mgr(argc, argv);
  Parameter<string> imageName(ParameterDef<string>("image", "The image filename", ""), &mgr);
  Parameter<int> nruns(ParameterDef<int>("runs", "The number of times to run the algorithm", 1), &mgr);
  Parameter<int> r(ParameterDef<int>("radius", "The radius", 5), &mgr);
  Parameter<bool> sseonly(ParameterDef<bool>("sse_only", "If true, only run the SSE code, otherwise run the slow code too", true), &mgr);
  shared_ptr<ImageSink> mySink(new ImageSink("MySink"));

  shared_ptr<ImageSource> mySource(new ImageSource);
  mgr.addSubComponent(mySource);

  mgr.addSubComponent(mySink);
  mgr.launch();

  int radius = r.getVal();
  while(mySource->ok())
  {
    Image<PixRGB<float>> input(mySource->in().convertTo<PixRGB<float>>());
    //Image<PixRGB<float>> input = readImage(imageName.getVal()).convertTo<PixRGB<float>>();
    Image<PixLAB<float>> lab(input);
    Image<PixLABX<float>> labx(input);

    mySink->out(GenericImage(input), "Original RGB image");

    for (int i = 0; i < nruns.getVal(); i++)
    {
#ifdef MAKE_SLOW_VERSION
      ///* Non-SSE */
      if (!sseonly.getVal())
      {
        NRT_INFO("Starting slow transform");

        Image<PixGray<float>>         blurred(slow::blurredVarianceIntegralImage(lab, radius));
        vector<Image<PixGray<float>>> gradImgs = slow::calculateGradient(blurred, radius);
        Image<PixGray<float>>         ridgeImg(slow::calculateRidge(gradImgs, radius));

        mySink->out(GenericImage(ridgeImg), "Slow (Non-SSE)");
        NRT_INFO("Done with slow transform");
      }
#endif 

      /* SSE */
      {
        NRT_INFO("Starting SSE transform");

        Image<PixGray<float>>         blurred(sse::blurredVariance(labx, radius));
        vector<Image<PixGray<float>>> gradImgs = sse::calculateGradient(blurred, radius);
        Image<PixGray<float>>         ridgeImg(sse::calculateRidge(gradImgs, radius));

        Image<PixRGB<byte>> displayImage(normalize<float>(ridgeImg, PixGray<float>(0.0), PixGray<float>(255.0)));
        mySink->out(GenericImage(displayImage), "SSE (normalized)");
        NRT_INFO("Done with SSE transform");
      }
    }

    NRT_INFO("Timing info for " << nruns.getVal() << " runs:");
    NRT_INFO("Image Dims:\t" << input.dims());

    NRT_INFO("Slow Box Blur:\t" << TIMER_BLUR_SLOW.report());
    NRT_INFO("SSE Box Blur:\t" << TIMER_BLUR_SSE.report());

    NRT_INFO("Slow Gradient:\t" << TIMER_GRAD_SLOW.report());
    NRT_INFO("SSE Gradient:\t" << TIMER_GRAD_SSE.report());

    NRT_INFO("Slow Var Ridge:\t" << TIMER_RIDGE_SLOW.report());
    NRT_INFO("SSE Var Ridge:\t" << TIMER_RIDGE_SSE.report());
  }

  while (true);

  return 0;
}
