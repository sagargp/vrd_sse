#include "vrd_sse.h"
#include <emmintrin.h> // sse3
#include <xmmintrin.h> // sse
#include <math.h>

#define NUM_GRADIENT_DIRECTIONS 8
#define NUM_RIDGE_DIRECTIONS    NUM_GRADIENT_DIRECTIONS/2
#define BOUNDARY_STEP_SIZE      NUM_GRADIENT_DIRECTIONS

inline float hadd_ps(__m128 *a)
{ 
  float data[4];
  _mm_store_ps(data, *a);

  return data[0] + data[1] + data[2] + data[3];
}

void blurredVarianceSSE(float const * const inputImage, int32_t const w, int32_t const h, int32_t const r, float * outputImage)
{
    float * const integral  = new float[w*h*4];
    float * const integral2 = new float[w*h*4];

    // pre-compute some constants used below
    int const norm_2r = 2*r;
    int const norm_2r2 = 2*r*r;
    int const norm_r2 = r*r;
    int const norm_w1r = w+r-1;
    int const norm_h1r = h+r-1;
    int const w4 = 4*w;
    int const xrigw = 4*(2*w-2-r);
    int const yboth = w4*(2*h-2-r);
    __m128 _norm = _mm_setzero_ps();

    // set the first row
    for(int i=0; i<4; ++i)
    {
      integral[i]  = inputImage[i];
      integral2[i] = inputImage[i];
    }

    for (int x = 1; x < w; x++)
    {
      __m128 _prev = _mm_load_ps( &integral[ (x-1)*4] );
      __m128 _prev2 = _mm_load_ps( &integral2[ (x-1)*4] );
      __m128 _curr = _mm_load_ps( &inputImage[ x*4] );

      __m128 _resl = _mm_add_ps(_prev, _curr);
      __m128 _resl2 = _mm_add_ps(_prev2, _mm_mul_ps(_curr, _curr));

      _mm_store_ps((float*)&integral[x*4], _resl);
      _mm_store_ps((float*)&integral2[x*4], _resl2);
    }

    // set the first column
    for (int y = 1; y < h; y++)
    {
      __m128 _prev = _mm_load_ps( &integral[ (y-1)*w*4] );
      __m128 _prev2 = _mm_load_ps( &integral2[ (y-1)*w*4] );
      __m128 _curr = _mm_load_ps( &inputImage[ y*w*4] );

      __m128 _resl = _mm_add_ps(_prev, _curr);
      __m128 _resl2 = _mm_add_ps(_prev2, _mm_mul_ps(_curr, _curr));

      _mm_store_ps((float*)&integral[y*w4], _resl);
      _mm_store_ps((float*)&integral2[y*w4], _resl2);
    }

    // compute the integral image
    for (int y = 1; y < h; y++)
    {
      int const y1 = w4*(y-1); 
      int const yw = y*w;

      for (int x = 1; x < w; x++)
      {
        int const x1 = 4*(x-1);

        __m128 _currn = _mm_load_ps(&inputImage[ x*4 + w4*y ]); //(x + y*w)*4 ]);
        __m128 _currn2 = _mm_mul_ps(_currn, _currn); 

        __m128 _lfint = _mm_load_ps(&integral[ x1 + w4*y ]); // ((x-1) + (y-0)*w)*4 ]);
        __m128 _tpint = _mm_load_ps(&integral[ x*4 + y1 ]); // ((x-0) + (y-1)*w)*4 ]);
        __m128 _tlint = _mm_load_ps(&integral[ x1 + y1 ]); //((x-1) + (y-1)*w)*4 ]);

        __m128 _reslt = _mm_add_ps(_lfint, _tpint); 
        _reslt = _mm_sub_ps(_reslt, _tlint); 
        _reslt = _mm_add_ps(_reslt, _currn); 

        _mm_store_ps((float*)&integral[(yw + x)*4], _reslt);

        /* squared */
        __m128 _lfint2 = _mm_load_ps(&integral2[ x1 + w4*y ]); // ((x-1) + (y-0)*w)*4 ]);
        __m128 _tpint2 = _mm_load_ps(&integral2[ x*4 + y1 ]); // ((x-0) + (y-1)*w)*4 ]);
        __m128 _tlint2 = _mm_load_ps(&integral2[ x1 + y1 ]); //((x-1) + (y-1)*w)*4 ]);

        __m128 _reslt2 = _mm_add_ps(_lfint2, _tpint2); 
        _reslt2 = _mm_sub_ps(_reslt2, _tlint2); 
        _reslt2 = _mm_add_ps(_reslt2, _currn2); 

        _mm_store_ps((float*)&integral2[(yw + x)*4], _reslt2);
      }
    }

    // compute the blur when y<r and x<r (top left corner) 
    for (int y = 0; y < r;  y++)
    {
      int const ybot = w4*(y+r);
      int const ytop = w4*abs(y-r);
      float * outputrowptr = outputImage + y*w;

      for (int x = 0; x < r; x++)
      {
        int const xrig = 4*(x+r);
        int const xlef = 4*abs(x-r);

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( (x+r)*(y+r) );

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        // squared
        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur when y<r but x>2r (top edge)
    for (int y = 0; y < r; y++)
    {
      int const ybot = w4*(y+r);
      int const ytop = w4*abs(y-r);
      float * outputrowptr = outputImage + y*w;

      for (int x = r; x < w-r; x++)
      {
        int const xlef = 4*(x-r);
        int const xrig = 4*(x+r);

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( y * norm_2r + norm_2r2 );

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur when y<r and x>(w-r) (top right corner) 
    for (int y = 0; y < r; y++)
    {
      int const ybot = w4*(y+r);
      int const ytop = w4*abs(y-r);
      float * outputrowptr = outputImage + y*w;

      for (int x = w-r; x < w; x++)
      {
        int const xlef = 4*(x-r);
        int const xrig = xrigw - 4*w; 

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( (y+r)*(norm_w1r-x) );

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur when y>r and x<r (left edge)
    for (int y = r; y < h-r; y++)
    {
      int const ytop = w4*(y-r);
      int const ybot = w4*(y+r);
      float * outputrowptr = outputImage + y*w;

      for (int x = 0; x < r; x++)
      {
        int const xrig = 4*(x+r);
        int const xlef = 4*abs(x-r);

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( x*norm_2r + norm_2r2 );

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur when y>(h-r) and x<r (bottom left corner)
    for (int y = h-r; y < h; y++)
    {
      int const ytop = w4*(y-r);
      int const ybot = yboth - y*w4;
      float * outputrowptr = outputImage + y*w;

      for (int x = 0; x < r; x++)
      {
        int const xrig = 4*(x+r);
        int const xlef = 4*abs(x-r);

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( (norm_h1r-y)*(x+r) );

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur when y>(h-r) and x>r (bottom edge)
    for (int y = h-r; y < h; y++)
    {
      int const ytop = w4*(y-r);
      int const ybot = yboth - y*w4; 
      float * outputrowptr = outputImage + y*w;

      for (int x = r; x < w-r; x++)
      {
        int const xlef = 4*(x-r);
        int const xrig = 4*(x+r);

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( (norm_h1r-y)*norm_2r );

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + yboth ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + yboth ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur when y>(h-r) and x>(w-r) (bottom right corner)
    for (int y = h-r; y < h; y++)
    {
      int const ytop = w4*(y-r);
      int const ybot = yboth - y*w4; 
      float * outputrowptr = outputImage + y*w;

      for (int x = w-r; x < w; x++)
      {
        int const xlef = 4*(x-r);
        int const xrig = xrigw - 4*x;

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( (norm_h1r-y)*(norm_w1r-x) ); 

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur when y>r and x>(w-r) (right edge)    
    for (int y = r; y < h-r; y++)
    {
      int const ytop = w4*(y-r);
      int const ybot = w4*(y+r);
      float * outputrowptr = outputImage + y*w;

      for (int x = w-r; x < w; x++)
      {
        int const xlef = 4*(x-r);
        int const xrig = xrigw - 4*x;

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        _norm = _mm_set1_ps( norm_2r*(norm_w1r-x) ); 

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2
        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(sqrt(hadd_ps(&_l2norm)));
        outputrowptr++;
      }
    }

    // compute the blur on the rest of the image
    _norm = _mm_set1_ps( 4*r*r ); 
    for (int y = r; y < h-r; y++)
    {
      int const ytop = w4*(y-r);
      int const ybot = w4*(y+r);

      float * outputrowptr = outputImage + y*w + r; 

      for (int x = r; x < w-r; x++)
      {
        int const xlef = 4*(x-r);
        int const xrig = 4*(x+r);

        __m128 _toplef = _mm_load_ps( &integral[ xlef + ytop ] );
        __m128 _toprig = _mm_load_ps( &integral[ xrig + ytop ] );
        __m128 _botrig = _mm_load_ps( &integral[ xrig + ybot ] );
        __m128 _botlef = _mm_load_ps( &integral[ xlef + ybot ] );

        __m128 _reslt = _mm_sub_ps(_botrig, _botlef);
        _reslt = _mm_sub_ps(_reslt, _toprig);
        _reslt = _mm_add_ps(_reslt, _toplef);
        _reslt = _mm_div_ps(_reslt, _norm);

        //_mm_store_ps((float*)&output_ptr[4*(yw + x)], _reslt);

        /* squared */
        __m128 _toplef2 = _mm_load_ps( &integral2[ xlef + ytop ] );
        __m128 _toprig2 = _mm_load_ps( &integral2[ xrig + ytop ] );
        __m128 _botrig2 = _mm_load_ps( &integral2[ xrig + ybot ] );
        __m128 _botlef2 = _mm_load_ps( &integral2[ xlef + ybot ] );

        __m128 _reslt2 = _mm_sub_ps(_botrig2, _botlef2);
        _reslt2 = _mm_sub_ps(_reslt2, _toprig2);
        _reslt2 = _mm_add_ps(_reslt2, _toplef2);
        _reslt2 = _mm_div_ps(_reslt2, _norm);

        // output = integral2 - integral^2

        __m128 _l2norm = _mm_sub_ps(_reslt2, _mm_mul_ps(_reslt, _reslt));
        _l2norm = _mm_mul_ps(_l2norm, _l2norm);

        *outputrowptr = sqrt(hadd_ps(&_l2norm));
        outputrowptr++;
        //_mm_store_ps((float*)&output_ptr[4*(yw + x)], _reslt);
      }
    }


    delete[] integral;
    delete[] integral2;
}

void calculateGradientSSE(float const * const inputImage, int const w, int const h, int const r, float * gradX, float * gradY)
{
  float dx[NUM_GRADIENT_DIRECTIONS];
  float dy[NUM_GRADIENT_DIRECTIONS];
  float rdxdy[NUM_GRADIENT_DIRECTIONS*4];

  float const pi2 = 2.0f*M_PI;
  float const norm = 1/float(NUM_GRADIENT_DIRECTIONS);

  /* pre-load r*dx and r*dy */
  for (int k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
  {
    float const idx = pi2*float(k)*norm;
    dx[k] = cos(idx);
    dy[k] = sin(idx);

    rdxdy[4*k + 0] = int(+r*dx[k]);
    rdxdy[4*k + 1] = int(+r*dy[k]);
    rdxdy[4*k + 2] = int(-r*dx[k]);
    rdxdy[4*k + 3] = int(-r*dy[k]);
  }
  /* now data starting at rdxdy[k] contains: (rdx[k], rdy[k], -rdx[k], -rdy[k]) */

  __m128 _clamp = _mm_set_ps(h-2, w-2, h-2, w-2);
  __m128 _1w1w = _mm_set_ps(w, 1, w, 1);
  __m128 _signmask = _mm_set1_ps(-0.f);

  float ij[4];

  for (int j = 0; j < h; j++)
  {
    for (int i = 0; i < w; i++)
    {
      float sumX = 0.0;
      float sumY = 0.0;

      for (int k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
      {
        /* load (r*dx, r*dy, -r*dx, -r*dy) into an SSE reg */
        __m128 _rdxdy = _mm_load_ps(&rdxdy[4*k]); 

        /* set _ij = (i, j, i, j) */
        __m128 _ij = _mm_set_ps(j, i, j, i);

        /* _ij = (i+rdx, j+rdy, i-rdx, j-rdx) */
        _ij = _mm_add_ps(_ij, _rdxdy);

        /* _ij = abs( (i+rdx, j+rdy, i-rdx, j-rdx) ) */
        _ij = _mm_andnot_ps(_signmask, _ij);

        /* clamp values inside _ij */
        _ij = _mm_min_ps(_clamp, _ij);

        /* reshape the coords in _ij into 1d */
        _ij = _mm_mul_ps(_ij, _1w1w);

        /* store _ij back into a regular float array: ij = j2 i2 j1 i1 */ 
        _mm_store_ps(ij, _ij);

        float val = inputImage[int(ij[0] + ij[1])] - inputImage[int(ij[2]+ij[3])];

        sumX += val * dx[k];
        sumY += val * dy[k];
      }
      gradX[i + j*w] = sumX;
      gradY[i + j*w] = sumY;
    }
  }
}

void calculateRidgeSSE(float const * const gradX, float const * const gradY, int const w, int const h, int const r, float * ridgeImage)
{
  float dx[NUM_GRADIENT_DIRECTIONS];
  float dy[NUM_GRADIENT_DIRECTIONS];
  float rdx[NUM_GRADIENT_DIRECTIONS];
  float rdy[NUM_GRADIENT_DIRECTIONS];

  float const pi2 = 2.0f*M_PI;
  float const norm = 1.0/float(NUM_GRADIENT_DIRECTIONS);

  /* pre-load dx and dy */
  for (int k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
  {
    float const idx = pi2*float(k)*norm;
    dx[k] = cos(idx);
    dy[k] = sin(idx);

    rdx[k] = int(r*dx[k]);
    rdy[k] = int(r*dy[k]);
  }

  for (int j = 0; j < h; j++)
  {
    for (int i = 0; i < w; i++)
    {
      float max = -INFINITY;

      for (int k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
      {
        int i_p = fmin(float(w-2), abs(i + rdx[k]));
        int j_p = fmin(float(h-2), abs(j + rdy[k]));
        int i_m = fmin(float(w-2), abs(i - rdx[k]));
        int j_m = fmin(float(h-2), abs(j - rdy[k]));

        float rgeo = sqrt(fmax(0.0F, 
              -(gradX[i_m + j_m*w] * dx[k] +
                gradY[i_m + j_m*w] * dy[k]) * 

              (gradX[i_p + j_p*w] * dx[k] +
               gradY[i_p + j_p*w] * dy[k])));

        float rarith = fmax(0.0F,
            (gradX[i_m + j_m*w] * dx[k] + gradY[i_m + j_m*w] * dy[k]) - 
            (gradX[i_p + j_p*w] * dx[k] + gradY[i_p + j_p*w] * dy[k])
            );

        max = fmax(max, rgeo+rarith);
      }
      ridgeImage[i+j*w] = max - sqrt(pow(gradX[i + j*w], 2) + pow(gradY[i + j*w], 2));
    }
  }
}
