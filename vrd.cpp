#include <nrt/Core/Model/Manager.H>
#include <nrt/ImageProc/IO/ImageSink/ImageSinks.H>
#include <nrt/ImageProc/IO/ImageSource/ImageReaders/ImageReader.H>
#include <nrt/Core/Debugging/TimeProfiler.H>
#include <nrt/Eigen/Eigen.H>
#include <nrt/Eigen/EigenConversions.H>
#include <xmmintrin.h> // sse
#include <emmintrin.h> // sse3
#include <valgrind/callgrind.h>

using namespace nrt;
using namespace std;

// vrd(Image<PixRGB<float>> input) {
//	I <- rgb2lab(input)
//	V <- variance_transform(I)
//	Vg <- grad(V)
//	R <- ridge(Vg)
//	B <- grad_magnitude_subtraction(R)
//	return B
// }

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))
#define FROUND(x) ( (int((x) * 1.0) / 1.0) )

#define NUM_GRADIENT_DIRECTIONS	8
#define NUM_RIDGE_DIRECTIONS	NUM_GRADIENT_DIRECTIONS/2
#define BOUNDARY_STEP_SIZE		NUM_GRADIENT_DIRECTIONS

#define TIMER_BLUR_SLOW	nrt::TimeProfiler<0>::instance()
#define TIMER_MAG_SLOW	nrt::TimeProfiler<1>::instance()
#define TIMER_GRAD_SLOW	nrt::TimeProfiler<2>::instance()
#define TIMER_BLUR_SSE	nrt::TimeProfiler<3>::instance()
#define TIMER_MAG_SSE	nrt::TimeProfiler<4>::instance()
#define TIMER_GRAD_SSE	nrt::TimeProfiler<5>::instance()
#define TIMER_BLUR_INT	nrt::TimeProfiler<6>::instance()

void _print_reg(__m128 *r)
{
	float result[4];
	_mm_store_ps(result, *r);

	for (int i = 0; i < 4; i++)
		cout << result[i] << " "; 
}

/********************
 * SSE code
 ********************/
namespace sse {
	Image<PixLABX<float>> blurredVarianceIntegralImage(Image<PixLABX<float>> const input, int const r)
	{
		TIMER_BLUR_SSE.begin();
		int const w = input.width();
		int const h = input.height();

		Image<PixLABX<float>, UniqueAccess> integral(w, h, ImageInitPolicy::None);
		Image<PixLABX<float>, UniqueAccess> output(w, h, ImageInitPolicy::None);
        
        float const * const integral_ptr = integral.pod_begin();
        float const * const input_ptr = input.pod_begin();
        float const * const output_ptr = output.pod_begin();

		// set the first row and first column
		integral(0, 0) = input(0, 0);
		for (int x = 1; x < w; x++)
        {
            __m128 _prev = _mm_load_ps( &integral_ptr[ (x-1)*4] );
            __m128 _curr = _mm_load_ps( &input_ptr[ x*4] );

            __m128 _resl = _mm_setzero_ps();
            _resl = _mm_add_ps(_prev, _curr);

            _mm_store_ps((float*)&integral_ptr[x*4], _resl);
        }
        for (int y = 1; y < h; y++)
        {
            integral(0, y) = integral.at(0, y-1) + input.at(0, y);
            
            __m128 _prev = _mm_load_ps( &integral_ptr[ (y-1)*w*4] );
            __m128 _curr = _mm_load_ps( &input_ptr[ y*w*4] );

            __m128 _resl = _mm_setzero_ps();
            _resl = _mm_add_ps(_prev, _curr);

            _mm_store_ps((float*)&integral_ptr[y*w*4], _resl);
        }

        // compute the integral image
        for (int y = 1; y < h; y++)
        {
            for (int x = 1; x < w; x++)
            {
                __m128 _lfint = _mm_load_ps(&integral_ptr[ ((x-1) + (y-0)*w)*4 ]);
                __m128 _tpint = _mm_load_ps(&integral_ptr[ ((x-0) + (y-1)*w)*4 ]);
                __m128 _tlint = _mm_load_ps(&integral_ptr[ ((x-1) + (y-1)*w)*4 ]);
                __m128 _currn = _mm_load_ps(&input_ptr[ (x + y*w)*4 ]);
               
                __m128 _reslt = _mm_setzero_ps();
                _reslt = _mm_add_ps(_lfint, _tpint); 
                _reslt = _mm_sub_ps(_reslt, _tlint); 
                _reslt = _mm_add_ps(_reslt, _currn); 
                
                _mm_store_ps((float*)&integral_ptr[(y*w + x)*4], _reslt);
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
                
                __m128 _toplef = _mm_load_ps( &integral_ptr[ (xlef + ytop*w)*4 ] );
                __m128 _toprig = _mm_load_ps( &integral_ptr[ (xrig + ytop*w)*4 ] );
                __m128 _botrig = _mm_load_ps( &integral_ptr[ (xrig + ybot*w)*4 ] );
                __m128 _botlef = _mm_load_ps( &integral_ptr[ (xlef + ybot*w)*4 ] );
                
                __m128 _norm = _mm_set1_ps( (xrig-xlef) * (ybot-ytop) );

                __m128 _reslt = _mm_setzero_ps();
                _reslt = _mm_sub_ps(_botrig, _botlef);
                _reslt = _mm_sub_ps(_reslt, _toprig);
                _reslt = _mm_add_ps(_reslt, _toplef);
                _reslt = _mm_div_ps(_reslt, _norm);

                _mm_store_ps((float*)&output_ptr[(y*w + x)*4], _reslt);
            }
        }
		TIMER_BLUR_SSE.end();
		return Image<PixLABX<float>>(output);
	}

	Image<PixGray<float>> magnitudeLAB(Image<PixLABX<float>> lab)
	{
		TIMER_MAG_SSE.begin();

		int w = lab.width();
		int h = lab.height();
		Image<PixGray<float>> output(w, h);

		float const * labbegin = lab.pod_begin();
		float result[4];

		for (int y = 0; y < lab.height(); y++)
		{
			for (int x = 0; x < lab.width(); x++)
			{
				//												 LSB		 MSB
				// load the first 4 floats from labbegin: _sum = {f1, f2, f3, f4}
				// (note f4 is the first channel of the next pixel; so we don't care about it for this iteration
				__m128 _sum = _mm_load_ps(labbegin);

				// square each of them: {f1=f1^2, f2=f2^2, ...}
				_sum = _mm_mul_ps(_sum, _sum);

				// shift _sum right (toward the LSB):
				// _sum1 = {f2, f3, f4, 00}
				// _sum2 = {f3, f4, 00, 00}
				__m128 _sum1 = (__m128)_mm_srli_si128(_sum, 4);
				__m128 _sum2 = (__m128)_mm_srli_si128(_sum, 8);

				// now add everything up:
				// _sum3 = {f1, f2, f3, f4} +
				//		   {f2, f3, f4, 00} =
				//		   {f1+f2, f2+f3, f3+f4, f4}
				//
				// _sum4 = {f1+f2, f2+f3, f3+f4, f4} +
				//		   {f3,    f4,    00,    00} =
				//		   {f1+f2+f3, f2+f3+f4, f3+f4, f4}
				//
				// remember we don't care about f4 so the sum we want is now in the least significant 32 bits of _sum4
				// if we wanted f4 too, we could just shift right one more time
				__m128 _sum3 = _mm_add_ps(_sum, _sum1);
				__m128 _sum4 = _mm_add_ps(_sum3, _sum2);

				// now take the square root of the whole thing:
				__m128 _sq = _mm_sqrt_ps(_sum4);

				// and save the result
				_mm_store_ps(result, _sq);
				output(x, y) = result[0];

				labbegin += 4;
			}
		}

		TIMER_MAG_SSE.end();
		return output;
	}

	vector<Image<PixGray<float>>> calculateGradient(Image<PixGray<float>> input, int const r)
	{
		TIMER_GRAD_SSE.begin();

		int w = input.width();
		int h = input.height();

		vector<Image<PixGray<float>>> gradImg(2);
		gradImg[0] = Image<PixGray<float>>(w, h);
		gradImg[1] = Image<PixGray<float>>(w, h);

		float const * const input_ptr = input.pod_begin();

		float dx[NUM_GRADIENT_DIRECTIONS];
		float dy[NUM_GRADIENT_DIRECTIONS];
		float rdxdy[NUM_GRADIENT_DIRECTIONS*4];

		/* pre-load r*dx and r*dy */
		for (int k = 0; k < NUM_GRADIENT_DIRECTIONS; k++)
		{
			float const idx = 2.0f*M_PI*float(k)/float(NUM_GRADIENT_DIRECTIONS);
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

					float val = input_ptr[int(ij[0] + ij[1])] - input_ptr[int(ij[2]+ij[3])];

					sumX += val * dx[k];
					sumY += val * dy[k];
				}
				gradImg[0](i, j) = sumX;
				gradImg[1](i, j) = sumY;
			}
		}
		TIMER_GRAD_SSE.end();
		return gradImg;
	}
}

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

	Image<PixLAB<float>> blurredVariance(Image<PixLAB<float>> const lab, int const r)
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

		TIMER_BLUR_SLOW.end();
		return output;
	}

	Image<PixGray<float>> magnitudeLAB(Image<PixLAB<float>> lab)
	{
		TIMER_MAG_SLOW.begin();

		int w = lab.width();
		int h = lab.height();
		Image<PixGray<float>> output(w, h);

		for (int x = 0; x < lab.width(); x++)
			for (int y = 0; y < lab.height(); y++)
				output(x, y) = sqrt(pow(lab(x, y).l(), 2)
						+ pow(lab(x, y).a(), 2)
						+ pow(lab(x, y).b(), 2));

		TIMER_MAG_SLOW.end();
		return output;
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
		return eigenMatrixToImage<float>(rDir[0]);
	}
}

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
	shared_ptr<ImageSink> mySink(new ImageSink("MySink"));
	mgr.addSubComponent(mySink);
	mgr.launch();

	int radius = 5;
	Image<PixRGB<float>> input = readImage(imageName.getVal()).convertTo<PixRGB<float>>();
	Image<PixLAB<float>> lab(input);
	Image<PixLABX<float>> labx(input);

	//mySink->out(GenericImage(input), "Original RGB image");
	
	/* Non-SSE */
	{
		NRT_INFO("Starting slow transform");

		Image<PixLAB<float>>			blurred(slow::blurredVarianceIntegralImage(lab, radius));
		mySink->out(GenericImage(Image<PixGray<float>>(blurred)), "Slow blur");	
        
        //Image<PixGray<float>>			varImg(slow::magnitudeLAB(blurred));
		//vector<Image<PixGray<float>>>	gradImgs = slow::calculateGradient(varImg, radius);
		//Image<PixGray<float>>			ridgeImg(slow::calculateRidge(gradImgs, radius));
		//
		//mySink->out(GenericImage(ridgeImg), "Slow (Non-SSE)");
		//NRT_INFO("Done with slow transform");
	}

	/* SSE */
	{
		NRT_INFO("Starting SSE transform");

		Image<PixLABX<float>>			blurred(sse::blurredVarianceIntegralImage(labx, radius));
		mySink->out(GenericImage(Image<PixGray<float>>(blurred)),  	   "SSE blur");	
		
		//vector<Image<PixGray<float>>>	gradImgs = sse::calculateGradient(varImg, radius);
		//Image<PixGray<float>>			ridgeImg(slow::calculateRidge(gradImgs, radius));
		//
		//mySink->out(GenericImage(ridgeImg), "SSE");
		//NRT_INFO("Done with SSE transform");
	}

	NRT_INFO("Image Dims: " << input.dims());
	NRT_INFO("Slow Blurred Variance: " << TIMER_BLUR_SLOW.report());
	NRT_INFO("SSE Blurred Variance: " << TIMER_BLUR_SSE.report());
	
	NRT_INFO("Slow Magnitude LAB:    " << TIMER_MAG_SLOW.report());
	NRT_INFO("SSE Magnitude LAB:    " << TIMER_MAG_SSE.report());
	
	NRT_INFO("Slow Gradient:         " << TIMER_GRAD_SLOW.report());
	NRT_INFO("SSE Gradient:         " << TIMER_GRAD_SSE.report());
	
	while(true) { } 

	return 0;
}
