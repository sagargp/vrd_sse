#include <nrt/Core/Model/Manager.H>
#include <nrt/ImageProc/IO/ImageSink/ImageSinks.H>
#include <nrt/ImageProc/IO/ImageSource/ImageReaders/ImageReader.H>
#include <nrt/Core/Debugging/TimeProfiler.H>
#include <nrt/Eigen/Eigen.H>
#include <nrt/Eigen/EigenConversions.H>
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
namespace {
	Image<PixLAB<float>> blurredVariance_SSE(Image<PixLAB<float>> const lab, int const r)
	{
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
		return output;
	}

	Image<PixGray<float>> magnitudeLAB_SSE(Image<PixLAB<float>> lab)
	{
		int w = lab.width();
		int h = lab.height();
		Image<PixGray<float>> output(w, h);

		float const * labbegin = lab.pod_begin();
		float result[4];
		
		for (int y = 0; y < lab.height(); y++)
		{
			for (int x = 0; x < lab.width(); x++)
			{
				__m128 _sum = _mm_loadu_ps(labbegin);
			
				_sum = _mm_mul_ps(_sum, _sum);

				__m128 _sum1 = (__m128)_mm_srli_si128(_sum, 4);
				__m128 _sum2 = (__m128)_mm_srli_si128(_sum, 8);

				__m128 _sum3 = _mm_add_ps(_sum, _sum1);
				__m128 _sum4 = _mm_add_ps(_sum3, _sum2);

				__m128 _sq = _mm_sqrt_ps(_sum4);
				
				_mm_storeu_ps(result, _sq);
				output(x, y) = result[0];
				
				labbegin += 3;
			}
		}
		return output;
	}

	Image<PixGray<float>> standardDeviationLAB_SSE(Image<PixLAB<float>> lab, int const r)
	{
		Image<PixLAB<float>> blurred(blurredVariance_SSE(lab, r));
		return magnitudeLAB_SSE(blurred);
	}

	vector<Image<PixGray<float>>> calculateGradient_SSE(Image<PixGray<float>> input, int const rad)
	{
		int w = input.width();
		int h = input.height();

		vector<Image<PixGray<float>>> gradImg(2);
		gradImg[0] = Image<PixGray<float>>(w, h);
		gradImg[1] = Image<PixGray<float>>(w, h);

		// bbGray <-- boxBlur(input);
		
		Eigen::VectorXf dx = Eigen::VectorXf::LinSpaced(NUM_GRADIENT_DIRECTIONS, 0, (NUM_GRADIENT_DIRECTIONS-1)*2*M_PI/NUM_GRADIENT_DIRECTIONS);
		Eigen::VectorXf dy = Eigen::VectorXf::LinSpaced(NUM_GRADIENT_DIRECTIONS, 0, (NUM_GRADIENT_DIRECTIONS-1)*2*M_PI/NUM_GRADIENT_DIRECTIONS);

		dx = dx.array().cos();
		dy = dy.array().sin();

		float const * input_itr = input.pod_begin();

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
					
					//float val = input.at(i1,j1).val() - input.at(i2,j2).val();
					float val = &input_itr[w*i1+j1] - &input_itr[w*i2+j2];

					sumX +=  val * dx[k];
					sumY +=  val * dy[k]; 
				}
				gradImg[0](i, j) = sumX;
				gradImg[1](i, j) = sumY;
			}
		}
		return gradImg;
	}
}

/********************
 * Non-SSE code
 ********************/
namespace {
	Image<PixLAB<float>> blurredVariance(Image<PixLAB<float>> const lab, int const r)
	{
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

	vector<Image<PixGray<float>>> calculateGradient(Image<PixGray<float>> gray, int const rad)
	{
		int w = gray.width();
		int h = gray.height();

		vector<Image<PixGray<float>>> gradImg(2);
		gradImg[0] = Image<PixGray<float>>(w, h);
		gradImg[1] = Image<PixGray<float>>(w, h);

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

	TimeProfiler<0> timer;
	timer.reset();

	Image<PixRGB<float>> input = readImage(imageName.getVal()).convertTo<PixRGB<float>>();
	Image<PixLAB<float>> lab(input);
	
	mySink->out(GenericImage(input), "Original RGB image");
	
	/* Non-SSE */
	{
		NRT_INFO("Starting non-SSE transform");
		
		timer.begin();
		Image<PixGray<float>> varImg(standardDeviationLAB(lab, 5));
		vector<Image<PixGray<float>>> gradImgs = calculateGradient(varImg, 5);
		Image<PixGray<float>> ridgeImg(calculateRidge(gradImgs, 5));
		timer.end();
	
		NRT_INFO("Normal: " << timer.report());
		
		mySink->out(GenericImage(ridgeImg), "Non-SSE");

		NRT_INFO("Done with non-SSE transform");
	}

	timer.reset();
	
	/* SSE */
	{
		NRT_INFO("Starting SSE transform");

		timer.begin();
		Image<PixGray<float>> varImg_sse(standardDeviationLAB_SSE(lab, 5));
		vector<Image<PixGray<float>>> gradImgs_sse = calculateGradient_SSE(varImg_sse, 5);
		Image<PixGray<float>> ridgeImg_sse(calculateRidge(gradImgs_sse, 5));
		timer.end();
		
		NRT_INFO("SSE: " << timer.report());
		
		mySink->out(GenericImage(ridgeImg_sse), "SSE");

		NRT_INFO("Done with SSE transform");
	}


	while(true)
	{
	}

	return 0;
}
