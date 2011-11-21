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

void _print_reg(__m128 *r)
{
	float result[4];
	_mm_store_ps(result, *r);

	for (int i = 0; i < 4; i++)
		cout << result[i] << " "; 
	cout << endl;
}

/********************
 * SSE code
 ********************/
namespace sse {
	Image<PixLABX<float>> blurredVarianceIntegralImage(Image<PixLABX<float>> const input, int const r)
	{
		int w = input.width();
		int h = input.height();

		Image<PixLABX<double>> integral(input);
		Image<PixLABX<float>> output(w, h);
		
		// set the first row
		for (int x = 1; x < w; x++)
		{
			PixLABX<double> current = integral(x, 0);
			PixLABX<double> prev = integral(x-1, 0);
			
			double l = current.l() + prev.l();
			double a = current.a() + prev.a();
			double b = current.b() + prev.b();

			integral(x, 0) = PixLABX<double>(l, a, b, 0.0);
		}

		// set the first col 
		for (int y = 1; y < h; y++)
		{
			PixLABX<double> current = integral(0, y);
			PixLABX<double> prev = integral(0, y-1);
			
			double l = current.l() + prev.l();
			double a = current.a() + prev.a();
			double b = current.b() + prev.b();

			integral(0, y) = PixLABX<double>(l, a, b, 0.0);
		}

		// set every remaining pixel
		for (int x = 1; x < w; x++)
		{
			for (int y = 1; y < h; y++)
			{
				PixLABX<double> left_i		= integral(x-1, y);
				PixLABX<double> top_i		= integral(x, y-1);
				PixLABX<double> toplef_i	= integral(x-1, y-1);
				PixLABX<double> toplef		= input(x-1, y-1);
				PixLABX<double> current		= input(x, y);
				
				double l = left_i.l() + top_i.l() - toplef_i.l() - toplef.l() + current.l(); 
				double a = left_i.a() + top_i.a() - toplef_i.a() - toplef.a() + current.a(); 
				double b = left_i.b() + top_i.b() - toplef_i.b() - toplef.b() + current.b(); 

				integral(x, y) = PixLABX<double>(l, a, b, 0.0);
			}
		}

		// naive integral image
		//for (int x = 0; x < w; x++)
		//{
		//	for (int y = 0; y < h; y++)
		//	{
		//		double l = 0.0;
		//		double a = 0.0;
		//		double b = 0.0;
		//		
		//		for (int row = 0; row <= x; row++) {
		//			l += input(row, y).l();
		//			a += input(row, y).a();
		//			b += input(row, y).b();
		//		}
		//		
		//		for (int col = 0; col < y; col++) {
		//			l += input(x, col).l();
		//			a += input(x, col).a();
		//			b += input(x, col).b();
		//		}
		//		integral(x, y) = PixLABX<double>(l, a, b, 0.0);
		//	}
		//}

		for (int x = 0; x < w; x++)
		{
			for (int y = 0; y < h; y++)
			{
				cout << input(x, y).l() << "\t";
			}
			cout << endl;
		}
		cout << endl;
		for (int x = 0; x < w; x++)
		{
			for (int y = 0; y < h; y++)
			{
				cout << integral(x, y).l() << "\t";
			}
			cout << endl;
		}

		// compute the blur
		for (int x = 0; x < w; x++)
		{
			for (int y = 0; y < h; y++)
			{
				int ytop = max(y-r, 0);
				int ybot = min(y+r, h-1);
				int xlef = max(x-r, 0);
				int xrig = min(x+r, w-1);

				PixLABX<double> toplef = integral(xlef, ytop); 
				PixLABX<double> toprig = integral(xrig, ytop); 
				PixLABX<double> botrig = integral(xrig, ybot); 
				PixLABX<double> botlef = integral(xlef, ybot); 

				float l = (botrig.l() - botlef.l() - toprig.l() + toplef.l()) / (r*r);// ((xrig-xlef) * (ybot-ytop));
				float a = (botrig.a() - botlef.a() - toprig.a() + toplef.a()) / (r*r);// ((xrig-xlef) * (ybot-ytop));
				float b = (botrig.b() - botlef.b() - toprig.b() + toplef.b()) / (r*r);// ((xrig-xlef) * (ybot-ytop));

				output(x, y) = PixLABX<float>(l, a, b, 0.0);
			}
		}
		return output;
	}

	Image<PixLABX<float>> blurredVariance(Image<PixLABX<float>> const lab, int const r)
	{
		TIMER_BLUR_SSE.begin();

		int boxSize = pow(2*r+1, 2);
		int w = lab.width();
		int h = lab.height();

		Image<PixLABX<float>> output(w, h);
		float* output_ptr = output.pod_begin();

		float const * const labbegin = lab.pod_begin();

		__m128 _boxsize = _mm_set_ps1(boxSize);
		__m128 _result = _mm_setzero_ps();

		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				__m128 _sum = _mm_setzero_ps();

				int const ytop = max(y-r, 0);
				int const ybot = min(y+r, h-1);
				int const xlef = max(x-r, 0);
				int const xrig = min(x+r, w-1);

				for (int j = ytop; j <= ybot; j++)
				{
					for (int i = xlef; i <= xrig; i++)
					{
						float const * const pixbegin = &labbegin[(j*w+i)*4];

						__m128 _curpix = _mm_load_ps(pixbegin);
						_sum = _mm_add_ps(_curpix, _sum);
					}
				}
				_result = _mm_div_ps(_sum, _boxsize);

				float local_result[4];
				_mm_store_ps(local_result, _result);

				float const l = local_result[0];
				float const a = local_result[1];
				float const b = local_result[2];

				size_t const pos = (y*w + x)*4;
				output_ptr[pos + 0] = l;
				output_ptr[pos + 1] = a;
				output_ptr[pos + 2] = b;
			}
		}

		TIMER_BLUR_SSE.end();
		return output;
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
	//{
	//	NRT_INFO("Starting slow transform");

	//	Image<PixLAB<float>>			blurred(slow::blurredVariance(lab, radius));
	//	Image<PixGray<float>>			varImg(slow::magnitudeLAB(blurred));
	//	vector<Image<PixGray<float>>>	gradImgs = slow::calculateGradient(varImg, radius);
	//	Image<PixGray<float>>			ridgeImg(slow::calculateRidge(gradImgs, radius));
	//	
	//	mySink->out(GenericImage(ridgeImg), "Slow (Non-SSE)");
	//	NRT_INFO("Done with slow transform");
	//}

	/* SSE */
	{
		NRT_INFO("Starting SSE transform");

		Image<PixLABX<float>>			blurred(sse::blurredVariance(labx, radius));
		Image<PixLABX<float>>			blurIntegral(sse::blurredVarianceIntegralImage(labx, radius));
		
		Image<PixGray<float>>			varImg(sse::magnitudeLAB(blurred));
		Image<PixGray<float>>			varImgIntegral(sse::magnitudeLAB(blurIntegral));
		
		mySink->out(GenericImage(varImg), "Standard blur");	
		mySink->out(GenericImage(varImgIntegral), "Integral blur");	
		
		
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
