// ========================================================================== //
// The MIT License (MIT)                                                      //
//                                                                            //
// Copyright (c) 2017 Jefferson Amstutz                                       //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// in all copies or substantial portions of the Software.                     //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
// ========================================================================== //

#pragma once

#include <cmath>
#include <type_traits>

namespace psimd {

  // FUTURE CONFIG ITEMS //////////////////////////////////////////////////////

#ifndef DEFAULT_WIDTH
#  define DEFAULT_WIDTH 8
#endif

#ifdef _WIN32
#  define PSIMD_ALIGN(...) __declspec(align(__VA_ARGS__))
#else
#  define PSIMD_ALIGN(...) __attribute__((aligned(__VA_ARGS__)))
#endif

  // pack<> type //////////////////////////////////////////////////////////////

  template <typename T, int W = DEFAULT_WIDTH>
  struct PSIMD_ALIGN(16) pack
  {
    pack() = default;
    pack(T value);

    const T& operator[](int i) const;
          T& operator[](int i);

    // Data //

    T data[W];
    enum {static_size = W};
    using type = T;
  };

  template <int W = DEFAULT_WIDTH>
  using mask = pack<int, W>;

  // pack<> inlined members ///////////////////////////////////////////////////

  template <typename T, int W>
  inline pack<T, W>::pack(T value)
  {
    #pragma omp simd
    for(int i = 0; i < W; ++i)
      data[i] = value;
  }

  template <typename T, int W>
  inline const T& pack<T, W>::operator[](int i) const
  {
    return data[i];
  }

  template <typename T, int W>
  inline T& pack<T, W>::operator[](int i)
  {
    return data[i];
  }

  // pack<> arithmetic operators //////////////////////////////////////////////

  // binary operator+() //

  template <typename T, int W>
  inline pack<T, W> operator+(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] + p2[i]);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator+(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] + v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator+(const OTHER_T &v, const pack<T, W> &p1)
  {
    return p1 + v;
  }

  // binary operator+=() //

  template <typename T, int W>
  inline pack<T, W>& operator+=(pack<T, W> &p1, const pack<T, W> &p2)
  {
    return p1 = (p1 + p2);
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>&>::type
  operator+=(pack<T, W> &p1, const OTHER_T &v)
  {
    return p1 = (p1 + pack<T, W>(v));
  }

  // binary operator-() //

  template <typename T, int W>
  inline pack<T, W> operator-(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] - p2[i]);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator-(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] - v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator-(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) - p1;
  }

  // binary operator-=() //

  template <typename T, int W>
  inline pack<T, W>& operator-=(pack<T, W> &p1, const pack<T, W> &p2)
  {
    return p1 = (p1 - p2);
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>&>::type
  operator-=(pack<T, W> &p1, const OTHER_T &v)
  {
    return p1 = (p1 - pack<T, W>(v));
  }

  // binary operator*() //

  template <typename T, int W>
  inline pack<T, W> operator*(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] * p2[i]);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator*(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] * v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator*(const OTHER_T &v, const pack<T, W> &p1)
  {
    return p1 * v;
  }

  // binary operator*=() //

  template <typename T, int W>
  inline pack<T, W>& operator*=(pack<T, W> &p1, const pack<T, W> &p2)
  {
    return p1 = (p1 * p2);
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>&>::type
  operator*=(pack<T, W> &p1, const OTHER_T &v)
  {
    return p1 = (p1 * pack<T, W>(v));
  }

  // binary operator/() //

  template <typename T, int W>
  inline pack<T, W> operator/(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = p1[i] / p2[i];

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator/(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] / v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator/(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) / p1;
  }

  // binary operator/=() //

  template <typename T, int W>
  inline pack<T, W>& operator/=(pack<T, W> &p1, const pack<T, W> &p2)
  {
    return p1 = (p1 / p2);
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>&>::type
  operator/=(pack<T, W> &p1, const OTHER_T &v)
  {
    return p1 = (p1 / pack<T, W>(v));
  }

  // binary operator%() //

  template <typename T, int W>
  inline pack<T, W> operator%(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] % p2[i]);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator%(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] % v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator%(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) % p1;
  }

  // binary operator%=() //

  template <typename T, int W>
  inline pack<T, W>& operator%=(pack<T, W> &p1, const pack<T, W> &p2)
  {
    return p1 = (p1 % p2);
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>&>::type
  operator%=(pack<T, W> &p1, const OTHER_T &v)
  {
    return p1 = (p1 % pack<T, W>(v));
  }

  // pack<> bitwise operators /////////////////////////////////////////////////

  // binary operator<<() //

  template <typename T, int W>
  inline pack<T, W> operator<<(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] << p2[i]);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator<<(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] << v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator<<(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) << p1;
  }

  // binary operator>>() //

  template <typename T, int W>
  inline pack<T, W> operator>>(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] >> p2[i]);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator>>(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] >> v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator>>(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) >> p1;
  }

  // binary operator^() //

  template <typename T, int W>
  inline pack<T, W> operator^(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] ^ p2[i]);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator^(const pack<T, W> &p1, const OTHER_T &v)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] ^ v);

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, pack<T, W>>::type
  operator^(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) ^ p1;
  }

  // pack<> logic operators ///////////////////////////////////////////////////

  // binary operator==() //

  template <typename T, int W>
  inline mask<W> operator==(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] == p2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator==(const pack<T, W> &p1, const OTHER_T &v)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] == v) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator==(const OTHER_T &v, const pack<T, W> &p1)
  {
    return p1 == v;
  }

  // binary operator!=() //

  template <typename T, int W>
  inline mask<W> operator!=(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] != p2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator!=(const pack<T, W> &p1, const OTHER_T &v)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] != v) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator!=(const OTHER_T &v, const pack<T, W> &p1)
  {
    return p1 != v;
  }

  // binary operator<() //

  template <typename T, int W>
  inline mask<W> operator<(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] < p2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator<(const pack<T, W> &p1, const OTHER_T &v)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] < v) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator<(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) < p1;
  }

  // binary operator<=() //

  template <typename T, int W>
  inline mask<W> operator<=(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] <= p2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator<=(const pack<T, W> &p1, const OTHER_T &v)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] <= v) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator<=(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) <= p1;
  }

  // binary operator>() //

  template <typename T, int W>
  inline mask<W> operator>(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] > p2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator>(const pack<T, W> &p1, const OTHER_T &v)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] > v) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator>(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) > p1;
  }

  // binary operator>=() //

  template <typename T, int W>
  inline mask<W> operator>=(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] >= p2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator>=(const pack<T, W> &p1, const OTHER_T &v)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (p1[i] >= v) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  template <typename T, int W, typename OTHER_T>
  inline typename
  std::enable_if<std::is_convertible<OTHER_T, T>::value, mask<W>>::type
  operator>=(const OTHER_T &v, const pack<T, W> &p1)
  {
    return pack<T, W>(v) >= p1;
  }

  // binary operator&&() //

  template <int W>
  inline mask<W> operator&&(const mask<W> &m1, const mask<W> &m2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (m1[i] && m2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  // binary operator||() //

  template <int W>
  inline mask<W> operator||(const mask<W> &m1, const mask<W> &m2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = (m1[i] || m2[i]) ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  // unary operator!() //

  template <int W>
  inline mask<W> operator!(const mask<W> &m)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = !m[i] ? 0xFFFFFFFF : 0x00000000;

    return result;
  }

  // unary operator-() //

  template <typename T, int W>
  inline pack<T, W> operator-(const pack<T, W> &p)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = -p[i];

    return result;
  }

  // pack<> math functions ////////////////////////////////////////////////////

  template <typename T, int W>
  inline pack<T, W> abs(const pack<T, W> &p)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = std::abs(p[i]);

    return result;
  }

  template <typename T, int W>
  inline pack<T, W> sqrt(const pack<T, W> &p)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = std::sqrt(p[i]);

    return result;
  }

  template <typename T, int W>
  inline pack<T, W> sin(const pack<T, W> &p)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = std::sin(p[i]);

    return result;
  }

  template <typename T, int W>
  inline pack<T, W> cos(const pack<T, W> &p)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = std::cos(p[i]);

    return result;
  }

  template <typename T, int W>
  inline pack<T, W> tan(const pack<T, W> &p)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = std::tan(p[i]);

    return result;
  }

  template <typename T, int W>
  inline pack<T, W> max(const pack<T, W> &a, const pack<T, W> &b)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = std::max(a[i], b[i]);

    return result;
  }

  template <typename T, int W>
  inline pack<T, W> min(const pack<T, W> &a, const pack<T, W> &b)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = std::min(a[i], b[i]);

    return result;
  }

  // pack<> algorithms ////////////////////////////////////////////////////////

  template <typename T, int W, typename FCN_T>
  inline void foreach(pack<T, W> &p, FCN_T &&fcn)
  {
    #pragma omp simd
    for (int i = 0; i < W; ++i)
      fcn(p[i], i);
  }

  template <int W, typename FCN_T>
  inline void foreach_active(const mask<W> &m, FCN_T &&fcn)
  {
    #pragma omp simd
    for (int i = 0; i < W; ++i)
      if (m[i])
        fcn(i);
  }

  template <typename T, int W, typename FCN_T>
  inline void foreach_active(const mask<W> &m, pack<T, W> &p, FCN_T &&fcn)
  {
    #pragma omp simd
    for (int i = 0; i < W; ++i)
      if (m[i])
        fcn(p[i]);
  }

  template <int W>
  inline bool any(const mask<W> &m)
  {
    bool result = false;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      if (m[i])
        result = true;

    return result;
  }

  template <int W>
  inline bool none(const mask<W> &m)
  {
    return !any(m);
  }

  template <int W>
  inline bool all(const mask<W> &m)
  {
    bool result = true;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      if (!m[i])
        result = false;

    return result;
  }

  template <typename T, int W>
  inline pack<T, W> select(const mask<W> &m,
                           const pack<T, W> &t,
                           const pack<T, W> &f)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i) {
      if (m[i])
        result[i] = t[i];
      else
        result[i] = f[i];
    }

    return result;
  }

  // pack<> memory operations /////////////////////////////////////////////////

  template <typename PACK_T>
  inline PACK_T load(void* _from)
  {
    auto *from = (typename PACK_T::type*) _from;
    PACK_T result;

    #pragma omp simd
    for (int i = 0; i < PACK_T::static_size; ++i)
      result[i] = from[i];

    return result;
  }

  template <typename T, int W, typename OFFSET_T>
  inline void store(const pack<T, W> &p, T* dst, const pack<OFFSET_T, W> &o)
  {
    #pragma omp simd
    for (int i = 0; i < W; ++i)
      dst[o[i]] = p[i];
  }

} // ::psimd
