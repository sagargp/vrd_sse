//NRT_HEADER_BEGIN
/*! @file details/PixHSVImpl.H */

// ////////////////////////////////////////////////////////////////////////
//              The iLab Neuromorphic Robotics Toolkit (NRT)             //
// Copyright 2010 by the University of Southern California (USC) and the //
//                              iLab at USC.                             //
//                                                                       //
//                iLab - University of Southern California               //
//                Hedco Neurociences Building, Room HNB-10               //
//                    Los Angeles, Ca 90089-2520 - USA                   //
//                                                                       //
//      See http://ilab.usc.edu for information about this project.      //
// ////////////////////////////////////////////////////////////////////////
// This file is part of The iLab Neuromorphic Robotics Toolkit.          //
//                                                                       //
// The iLab Neuromorphic Robotics Toolkit is free software: you can      //
// redistribute it and/or modify it under the terms of the GNU General   //
// Public License as published by the Free Software Foundation, either   //
// version 3 of the License, or (at your option) any later version.      //
//                                                                       //
// The iLab Neuromorphic Robotics Toolkit is distributed in the hope     //
// that it will be useful, but WITHOUT ANY WARRANTY; without even the    //
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR       //
// PURPOSE.  See the GNU General Public License for more details.        //
//                                                                       //
// You should have received a copy of the GNU General Public License     //
// along with The iLab Neuromorphic Robotics Toolkit.  If not, see       //
// <http://www.gnu.org/licenses/>.                                       //
// ////////////////////////////////////////////////////////////////////////
//
// Primary maintainer for this file:
//
//NRT_HEADER_END


//// ######################################################################
template <class T>
PixLABX<T>::PixLABX() : PixelBase<T, 4, PixLABX>()
{ }

// ######################################################################
template <class T>
template <typename T1>
PixLABX<T>::PixLABX(T1 value) :
  PixelBase<T, 4, PixLABX>(value)
{ }

// ######################################################################
template <class T>
PixLABX<T>::PixLABX(T l, T a, T b, T x)
{
  this->channels[0] = l;
  this->channels[1] = a;
  this->channels[2] = b;
  this->channels[3] = x;
}

// ######################################################################
template<class T>
template<class T2>
PixLABX<T>::PixLABX(PixLABX<T2> const& other) :
  PixelBase<T, 4, PixLABX>(other)
{ }

// ######################################################################
template <class T>
template <class lType, class aType, class bType, class xType>
PixLABX<T>::PixLABX(lType l, aType a, bType b, xType x)
{
  this->channels[0] = clamped_convert<T>(l);
  this->channels[1] = clamped_convert<T>(a);
  this->channels[2] = clamped_convert<T>(b);
  this->channels[3] = clamped_convert<T>(x);
}

// ######################################################################
template <class T>
T PixLABX<T>::l() const
{ return this->channels[0]; }

// ######################################################################
template <class T>
T PixLABX<T>::a() const
{ return this->channels[1]; }

// ######################################################################
template <class T>
T PixLABX<T>::b() const
{ return this->channels[2]; }

// ######################################################################
template <class T>
T PixLABX<T>::x() const
{ return this->channels[3]; }

// ######################################################################
template <class T>
void PixLABX<T>::setL(T value)
{ this->channels[0] = value; }

// ######################################################################
template <class T>
void PixLABX<T>::setA(T value)
{ this->channels[1] = value; }

// ######################################################################
template <class T>
void PixLABX<T>::setB(T value)
{ this->channels[2] = value; }

// ######################################################################
template <class T>
void PixLABX<T>::setX(T value)
{ this->channels[3] = value; }

// ######################################################################
template <class T>
template<class T2>
void PixLABX<T>::setL(T2 value)
{ this->channels[0] = clamped_convert<T>(value); }

// ######################################################################
template <class T>
template<class T2>
void PixLABX<T>::setA(T2 value)
{ this->channels[1] = clamped_convert<T>(value); }

// ######################################################################
template <class T>
template<class T2>
void PixLABX<T>::setB(T2 value)
{ this->channels[2] = clamped_convert<T>(value); }

// ######################################################################
template <class T>
template<class T2>
void PixLABX<T>::setX(T2 value)
{ this->channels[3] = clamped_convert<T>(value); }

// ######################################################################
template <class T>
PixRGB<T> PixLABX<T>::toRGB() const
{
  double Xn = 095.0456;
  double Yn = 100.0000;
  double Zn = 108.8754;
  
  double var_Y = (l()+16.0)/116.0;
  double var_X = a()/500.0 + var_Y;
  double var_Z = var_Y - b()/200.0;

  if (pow(var_Y, 3.0) > 0.008856)
    var_Y = pow(var_Y, 3.0);
  else
    var_Y = (var_Y-16.0/116.0) / 7.787;

  if (pow(var_X, 3.0) > 0.008856)
    var_X = pow(var_X, 3.0);
  else
    var_X = (var_X - 16.0/116.0)/7.787;

  if (pow(var_Z, 3.0) > 0.008856)
    var_Z = pow(var_Z, 3.0);
  else
    var_Z = (var_Z-16.0/116.0)/7.787;
  
  double X = Xn * var_X;
  double Y = Yn * var_Y;
  double Z = Zn * var_Z;
  
  var_X = X / 100.0;
  var_Y = Y / 100.0;
  var_Z = Z / 100.0;
  
  double var_R =  3.240479*var_X - 1.537150*var_Y - 0.498535*var_Z;
  double var_G = -0.969256*var_X + 1.875991*var_Y + 0.041556*var_Z;
  double var_B =  0.055648*var_X - 0.204043*var_Y + 1.057311*var_Z;
  
  if ( var_R > 0.0031308 ) 
    var_R = 1.055 * pow(var_R,1.0/2.4) - 0.055;
  else
    var_R = 12.92 * var_R;

  if ( var_G > 0.0031308 )
    var_G = 1.055 * pow(var_G,1.0/2.4) - 0.055;
  else
    var_G = 12.92 * var_G;

  if ( var_B > 0.0031308 )
    var_B = 1.055 * pow(var_B,1.0/2.4) - 0.055;
  else
    var_B = 12.92 * var_B;
  
  double R = var_R * 255.0;
  double G = var_G * 255.0;
  double B = var_B * 255.0;
  
  return PixRGB<T>(R, G, B);
}

// ######################################################################
template <class T>
PixLABX<T> PixLABX<T>::fromRGB(PixRGB<T> const & other)
{
  // from opencv: http://bit.ly/pp5ZJm
  
  double fX, fY, fZ, L, a, b;
  
  double Xn = 0.950456;
  double Yn = 1.000000;
  double Zn = 1.088754;
  
  double X = 0.412453*other.r() + 0.357580*other.g() + 0.180423*other.b();
  double Y = 0.212671*other.r() + 0.715160*other.g() + 0.072169*other.b();
  double Z = 0.019334*other.r() + 0.119193*other.g() + 0.950227*other.b();
              
  X = X/(255.0*Xn);
  Y = Y/(255.0*Yn);
  Z = Z/(255.0*Zn);
  
  if (Y > 0.008856) {
    fY = pow(Y, 1.0/3.0);
    L = 116.0*fY - 16.0 + 0.5;
  } else {
    fY = 7.787*Y + 16.0/116.0;
    L = 903.3*Y + 0.5;
  }
  
  if (X > 0.008856)
    fX = pow(X, 1.0/3.0);
  else
    fX = 7.787*X + 16.0/116.0;
  
  if (Z > 0.008856)
    fZ = pow(Z, 1.0/3.0);
  else
    fZ = 7.787*Z + 16.0/116.0;
  
  a = 500.0 * (fX - fY) + 0.5;
  b = 200.0 * (fY - fZ) + 0.5;
 
  return PixLABX<T>(L, a, b, 0.0);
}

// ######################################################################
template <class T>
PixRGBD<T> PixLABX<T>::toRGBD() const
{
	double Xn = 095.0456;
	double Yn = 100.0000;
	double Zn = 108.8754;

	double var_Y = (l()+16.0)/116.0;
	double var_X = a()/500.0 + var_Y;
	double var_Z = var_Y - b()/200.0;

	if (pow(var_Y, 3.0) > 0.008856)
		var_Y = pow(var_Y, 3.0);
	else
		var_Y = (var_Y-16.0/116.0) / 7.787;

	if (pow(var_X, 3.0) > 0.008856)
		var_X = pow(var_X, 3.0);
	else
		var_X = (var_X - 16.0/116.0)/7.787;

	if (pow(var_Z, 3.0) > 0.008856)
		var_Z = pow(var_Z, 3.0);
	else
		var_Z = (var_Z-16.0/116.0)/7.787;

	double X = Xn * var_X;
	double Y = Yn * var_Y;
	double Z = Zn * var_Z;

	var_X = X / 100.0;
	var_Y = Y / 100.0;
	var_Z = Z / 100.0;

	double var_R =  3.240479*var_X - 1.537150*var_Y - 0.498535*var_Z;
	double var_G = -0.969256*var_X + 1.875991*var_Y + 0.041556*var_Z;
	double var_B =  0.055648*var_X - 0.204043*var_Y + 1.057311*var_Z;

	if ( var_R > 0.0031308 ) 
		var_R = 1.055 * pow(var_R,1.0/2.4) - 0.055;
	else
		var_R = 12.92 * var_R;

	if ( var_G > 0.0031308 )
		var_G = 1.055 * pow(var_G,1.0/2.4) - 0.055;
	else
		var_G = 12.92 * var_G;

	if ( var_B > 0.0031308 )
		var_B = 1.055 * pow(var_B,1.0/2.4) - 0.055;
	else
		var_B = 12.92 * var_B;

	double R = var_R * 255.0;
	double G = var_G * 255.0;
	double B = var_B * 255.0;

	return PixRGBD<T>(R, G, B, 0.0);
}

// ######################################################################
template <class T>
PixLABX<T> PixLABX<T>::fromRGBD(PixRGBD<T> const & other)
{
	// from opencv: http://bit.ly/pp5ZJm

	double fX, fY, fZ, L, a, b;

	double Xn = 0.950456;
	double Yn = 1.000000;
	double Zn = 1.088754;

	double X = 0.412453*other.r() + 0.357580*other.g() + 0.180423*other.b();
	double Y = 0.212671*other.r() + 0.715160*other.g() + 0.072169*other.b();
	double Z = 0.019334*other.r() + 0.119193*other.g() + 0.950227*other.b();

	X = X/(255.0*Xn);
	Y = Y/(255.0*Yn);
	Z = Z/(255.0*Zn);

	if (Y > 0.008856) {
		fY = pow(Y, 1.0/3.0);
		L = 116.0*fY - 16.0 + 0.5;
	} else {
		fY = 7.787*Y + 16.0/116.0;
		L = 903.3*Y + 0.5;
	}

	if (X > 0.008856)
		fX = pow(X, 1.0/3.0);
	else
		fX = 7.787*X + 16.0/116.0;

	if (Z > 0.008856)
		fZ = pow(Z, 1.0/3.0);
	else
		fZ = 7.787*Z + 16.0/116.0;

	a = 500.0 * (fX - fY) + 0.5;
	b = 200.0 * (fY - fZ) + 0.5;

	return PixLABX<T>(L, a, b, 0.0);
}

