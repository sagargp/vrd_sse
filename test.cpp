#include <emmintrin.h> // sse3
#include <iostream>
#include <nrt/Core/Model/Manager.H>
#include <nrt/Eigen/Eigen.H>
#include <nrt/Eigen/EigenConversions.H>
#include <nrt/ImageProc/IO/ImageSink/ImageSinks.H>
#include <nrt/ImageProc/IO/ImageSource/ImageReaders/ImageReader.H>
#include <xmmintrin.h> // sse

using namespace std;

void _print_reg(__m128 *r)
{
	float result[4];
	_mm_store_ps(result, *r);

	for (int i = 0; i < 4; i++)
		cout << result[i] << " "; 
	cout << endl;
}

int main()
{
  int a = 100;
  int b = a<<1;
  int c = a<<2;

  cout << b << endl;
  cout << c << endl;
}
