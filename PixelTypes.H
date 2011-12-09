//NRT_HEADER_BEGIN
/*! @file Image/PixelTypes.H Definitions of some common color spaces */

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
// Primary maintainer for this file: Randolph Voorhies
//
//NRT_HEADER_END

#ifndef SAGAR_NRT_CORE_IMAGE_PIXELTYPES_H
#define SAGAR_NRT_CORE_IMAGE_PIXELTYPES_H

#include <nrt/Core/Image/PixelBase.H>
#include <nrt/Core/Debugging/Log.H>

namespace nrt
{
  //! All pixel values range from 0-255
  template <class T> class PixLAB;
  template <class T> class PixLABX;

  // ######################################################################
  //! A pixel class to hold lightness-alpha-beta color components
  template <class T>
  struct PixLAB : public PixelBase<T, 3, PixLAB>
  {
    //! Default Constructor
    /*! All components will be uninitialized */
    PixLAB();
  
    //! Copy Constructor
    template<class T2>
    PixLAB(PixLAB<T2> const& other);
  
    //! initialize with a POD type or pixel
    template <typename T1>
    explicit PixLAB(T1 val);
  
    //! Value Constructor
    /*! All components will be initialized to the given values */
    PixLAB(T l, T a, T b);
  
    //! Clamping Value Constructor
    /*! All components will be initialized to the given values and clamped/rounded if necessary */
    template<class lType, class aType, class bType>
    PixLAB(lType l, aType a, bType b);
  
    //! Read the lightness channel
    T l() const;
    //! Read the alpha channel
    T a() const;
    //! Read the beta channel
    T b() const;
  
    //! Set the lightness channel
    void setL(T value);
    //! Set the alpha channel
    void setA(T value);
    //! Set the beta channel
    void setB(T value);
    //! Set the lightness channel and clamp/round if necessary
    template<class T2> void setL(T2 value);
    //! Set the alpha channel and clamp/round if necessary
    template<class T2> void setA(T2 value);
    //! Set the beta channel and clamp/round if necessary
    template<class T2> void setB(T2 value);

    //! Convert this color space to RGB
    PixRGB<T> toRGB() const;
    //! Convert to this color space from RGB
    static PixLAB<T> fromRGB(PixRGB<T> const & other);
  };

  #include "PixLABImpl.H"

  //! Stream out as PixLAB(l,a,b)
  /*! \relates PixLAB */
  template<class T>
  inline std::ostream& operator<< (std::ostream &out, PixLAB<T> pix)
  {
    out << "PixLAB" << pix.toString();
    return out;
  }

  // ######################################################################
  //! A pixel class to hold lightness-alpha-beta color components (plus one extra channel for SSE alignment)
  template <class T>
  struct PixLABX : public PixelBase<T, 4, PixLABX>
  {
    //! Default Constructor
    /*! All components will be uninitialized */
    PixLABX();
  
    //! Copy Constructor
    template<class T2>
    PixLABX(PixLABX<T2> const& other);
  
    //! initialize with a POD type or pixel
    template <typename T1>
    explicit PixLABX(T1 val);
  
    //! Value Constructor
    /*! All components will be initialized to the given values */
    PixLABX(T l, T a, T b, T x);
  
    //! Clamping Value Constructor
    /*! All components will be initialized to the given values and clamped/rounded if necessary */
    template<class lType, class aType, class bType, class xType>
    PixLABX(lType l, aType a, bType b, xType x);
  
    //! Read the lightness channel
    T l() const;
    //! Read the alpha channel
    T a() const;
    //! Read the beta channel
    T b() const;
    //! Read the junk channel
    T x() const;
  
    //! Set the lightness channel
    void setL(T value);
    //! Set the alpha channel
    void setA(T value);
    //! Set the beta channel
    void setB(T value);
    //! Set the extra channel
    void setX(T value);
    //! Set the lightness channel and clamp/round if necessary
    template<class T2> void setL(T2 value);
    //! Set the alpha channel and clamp/round if necessary
    template<class T2> void setA(T2 value);
    //! Set the beta channel and clamp/round if necessary
    template<class T2> void setB(T2 value);
    //! Set the extra channel and clamp/round if necessary
    template<class T2> void setX(T2 value);

    //! Convert this color space to RGB
    PixRGB<T> toRGB() const;
    //! Convert to this color space from RGB
    static PixLABX<T> fromRGB(PixRGB<T> const & other);
    
	//! Convert this color space to RGBD
    PixRGBD<T> toRGBD() const;
    //! Convert to this color space from RGBD
    static PixLABX<T> fromRGBD(PixRGBD<T> const & other);
  };

  #include "PixLABXImpl.H"

  //! Stream out as PixLABX(l,a,b)
  /*! \relates PixLABX */
  template<class T>
  inline std::ostream& operator<< (std::ostream &out, PixLABX<T> pix)
  {
    out << "PixLABX" << pix.toString();
    return out;
  }
}

#endif // NRT_CORE_IMAGE_PIXELTYPES_H

