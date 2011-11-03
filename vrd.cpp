#include <nrt/Core/Model/Manager.H>
#include <nrt/ImageProc/IO/ImageSink/ImageSinks.H>
#include <nrt/ImageProc/IO/ImageSource/ImageReaders/ImageReader.H>
#include <nrt/Core/Debugging/TimeProfiler.H>
#include <eigen3/Eigen/Eigen>
#include <xmmintrin.h> // sse
#include <emmintrin.h> // sse3

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

#define NUM_GRADIENT_DIRECTIONS	8
#define NUM_RIDGE_DIRECTIONS	NUM_GRADIENT_DIRECTIONS/2
#define BOUNDARY_STEP_SIZE		NUM_GRADIENT_DIRECTIONS

/********************
 * SSE code
 ********************/
{
	Image<PixLAB<float>> blurredVariance_SSE(Image<PixLAB<float>> const lab, int const r)
	{
		NRT_INFO("Starting SSE blur r=" << r);
		int boxSize = pow(2*r+1, 2);
		int w = lab.width();
		int h = lab.height();

		Image<PixLAB<float>> output(w, h, ImageInitPolicy::None);
		
		float const * const labbegin = lab.pod_begin();
		
		__m128 _boxsize = _mm_set_ps1(boxSize);
		__m128 _result = _mm_setzero_ps();

		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				__m128 _sum = _mm_setzero_ps();

				int ytop = max(y-r, 0);
				int ybot = min(y+r, h-1);
				int xlef = max(x-r, 0);
				int xrig = min(x+r, w-1);
			
				for (int j = ytop; j <= ybot; j++)
				{
					for (int i = xlef; i <= xrig; i++)
					{
						// warning:
						// at the last pixel, this could try to load 4 bytes of memory we don't own
						float const * const pixbegin = &labbegin[(j*w+i)*3];

						__m128 _curpix = _mm_loadu_ps(pixbegin);
						_sum = _mm_add_ps(_curpix, _sum);
					}
				}
				_result = _mm_div_ps(_sum, _boxsize);
				
				float local_result[4];
				_mm_storeu_ps(local_result, _result);

				float l = local_result[0];
				float a = local_result[1];
				float b = local_result[2];

				output(x, y) = PixLAB<float>(l, a, b);
			}
		}
		NRT_INFO("Finished SSE blur");
		return output;
	}

	Image<PixGray<float>> magnitudeLAB_SSE(Image<PixLAB<float>> lab)
	{
		Image<PixGray<float>> output(lab.dims());

		for ( float const * labbegin = lab.pod_begin(); labbegin != lab.pod_end(); labbegin += 3)
		{
			__m128 _channels = _mm_loadu_ps(labbegin); // _channels = [l, a, b]
			_channels = _mm_mul_ps(_channels, _channels); // _channels = [l^2, a^2, b^2]

			__m128i _sum = (__m128i) _channels;	
			_sum = _mm_add_epi8(_sum, _mm_srli_si128(_sum, 1));
			_sum = _mm_add_epi8(_sum, _mm_srli_si128(_sum, 2));
			_sum = _mm_add_epi8(_sum, _mm_srli_si128(_sum, 4));
			_sum = _mm_add_epi8(_sum, _mm_srli_si128(_sum, 8));
			
			__m128 _sq = _mm_sqrt_ps( (__m128)_sum );
			output(x, y) = _mm_cvtsi128_si32( (__m128i)_sq );
		}
		return output;
	}

	Image<PixGray<float>> standardDeviationLAB_SSE(Image<PixLAB<float>> lab, int const r)
	{
		Image<PixLAB<float>> blurred(blurredVariance_SSE(lab, r));
		return magnitudeLAB_SSE(blurred);
	}
}

/********************
 * Non-SSE code
 ********************/
{
	Image<PixLAB<float>> blurredVariance(Image<PixLAB<float>> const lab, int const r)
	{
		NRT_INFO("Starting Blur r=" << r);
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

		NRT_INFO("Finished blur");
		return output;
	}

	Image<PixGray<float>> magnitudeLAB(Image<PixLAB<float>> lab)
	{
		int w = lab.width();
		int h = lab.height();
		Image<PixGray<float>> output(w, h);
		
		for (int x = 0; x < lab.width(); x++)
			for (int y = 0; y < lab.height(); y++)
				output(x, y) = sqrt(pow(lab(x, y).l(), 2)
								  + pow(lab(x, y).a(), 2)
								  + pow(lab(x, y).b(), 2));
		return output;
	}

	Image<PixGray<float>> standardDeviationLAB(Image<PixLAB<float>> lab, int const r)
	{
		Image<PixLAB<float>> blurred(blurredVariance(lab, r));
		return magnitudeLAB(blurred);
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

	TimeProfiler<0> timer;
	timer.reset();

	Image<PixRGB<float>> input = readImage(imageName.getVal()).convertTo<PixRGB<float>>();
	Image<PixLAB<float>> lab(input);
	
	timer.begin();
	Image<PixGray<float>> std(standardDeviationLAB(lab, 5));
	timer.end();
	NRT_INFO("normal blur: " << timer.report());
	
	timer.reset();

	timer.begin();
	Image<PixGray<float>> std_sse(standardDeviationLAB_SSE(lab, 5));
	timer.end();
	NRT_INFO("SSE blur: " << timer.report());
	
	Image<PixRGB<float>> output(std);

	mySink->out(GenericImage(input), "Original RGB image");
	mySink->out(GenericImage(output), "Output image");

	//Image<PixLAB<float>> blurredLAB(blurredVariance(convertedLAB, 8));
	//Image<PixRGB<float>> convertedRGB(convertedLAB);
	//Image<PixRGB<float>> convertedBlurredRGB(blurredLAB);

	//mySink->out(GenericImage(originalRGB), "Original RGB image");
	//mySink->out(GenericImage(convertedRGB), "Converted RGB image");
	//mySink->out(GenericImage(convertedBlurredRGB), "Blurred RGB image");
	
	while(true)
	{
	}

	return 0;
}
