#ifndef COLOR_H
#define COLOR_H

#include <cstdint>
#include <array>
#include <cassert>
#include <ostream>
#include "immintrin.h"

//Since we fwrite this struct directly and PPM only takes RGB (24 bits)
//we can't allow any padding to be added onto the end
#pragma pack(1)
struct Color24 {
	uint8_t r, g, b;

	inline Color24(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0) : r(r), g(g), b(b){}
	inline uint8_t& operator[](int i){
		switch (i){
			case 0:
				return r;
			case 1:
				return g;
			default:
				return b;
		}
	}
};
#pragma pack(pop)

/*
 * Struct storing a single RGB floating point color
 */
struct Colorf {
	float r, g, b;

	/*
	 * Initialize the RGB values to the same value
	 */
	Colorf(float c = 0);
	/*
	 * Create an RGB color
	 */
	Colorf(float r, float g, float b);
	/*
	 * Normalize the floating point color values to be clamped between 0-1
	 */
	void normalize();
	Colorf normalized() const;
	/*
	 * Compute the luminance of the  color
	 */
	float luminance() const;
	/*
	 * Check if the color is black
	 */
	bool is_black() const;
	/*
	 * Compute and return the sRGB color value for the linear RGB color value
	 */
	Colorf to_sRGB() const;
	Colorf& operator+=(const Colorf &c);
	Colorf& operator-=(const Colorf &c);
	Colorf& operator*=(const Colorf &c);
	Colorf& operator*=(float s);
	Colorf& operator/=(float s);
	float& operator[](int i);
	const float& operator[](int i) const;
	bool has_nans() const;
	/*
	 * Compute the value of e^color, returning a color with value
	 * e^r, e^g, e^b
	 */
	Colorf exp() const;
	/*
	 * Easily convert to the 24bpp and 32bpp color representations
	 */
	operator Color24() const;	
};
Colorf operator+(const Colorf &a, const Colorf &b);
Colorf operator-(const Colorf &a, const Colorf &b);
Colorf operator-(const Colorf &c);
Colorf operator*(const Colorf &a, const Colorf &b);
Colorf operator*(const Colorf &a, float s);
Colorf operator*(float s, const Colorf &a);
Colorf operator/(const Colorf &a, const Colorf &b);
Colorf operator/(const Colorf &c, float s);
bool operator==(const Colorf &a, const Colorf &b);
bool operator!=(const Colorf &a, const Colorf &b);
std::ostream& operator<<(std::ostream &os, const Colorf &c);

// Compute x^y, this just calls pow a bunch, it doesn't look like there's an AVX pow instruction
inline __m256 vpow(__m256 vx, __m256 vy){
	float *x = (float*)&vx;
	float *y = (float*)&vy;
	float z[8] = {0};
	for (int i = 0; i < 8; ++i){
		z[i] = std::pow(x[i], y[i]);
	}
	return _mm256_loadu_ps(z);
}
// Compute sRGB values for some color channel
inline __m256 convert_srgb(__m256 x){
	const static auto a = _mm256_set1_ps(0.055);
	const static auto b = _mm256_rcp_ps(_mm256_set1_ps(2.4));		
	const auto mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.0031308), _CMP_LE_OQ);
	// We need to compute both branches for the sRGB conversion then pick
	// the right one based on the mask
	const auto y = _mm256_mul_ps(x, _mm256_set1_ps(12.92));
	const auto z = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(1.055), vpow(x, b)), a);
	return _mm256_blendv_ps(y, z, mask);
}

/*
 * Floating point color struct storing 8 RGB colors
 */
struct Colorf_8 {
	__m256 r, g, b;

	/*
	 * Initialize the RGB values to the same value
	 */
	inline Colorf_8(float c = 0) : r(_mm256_set1_ps(c)), g(r), b(r){}
	/*
	 * Create an RGB color
	 */
	inline Colorf_8(float r, float g, float b) : r(_mm256_set1_ps(r)), g(_mm256_set1_ps(g)), b(_mm256_set1_ps(b)){}
	inline Colorf_8(Colorf c) : r(_mm256_set1_ps(c.r)), g(_mm256_set1_ps(c.g)), b(_mm256_set1_ps(c.b)){}
	inline Colorf_8(__m256 r, __m256 g, __m256 b) : r(r), g(g), b(b){}
	/*
	 * Normalize the floating point color values to be clamped between 0-1
	 */
	inline void normalize(){
		const auto zero = _mm256_set1_ps(0);
		const auto one = _mm256_set1_ps(1);
		r = _mm256_max_ps(zero, _mm256_min_ps(one, r));
		g = _mm256_max_ps(zero, _mm256_min_ps(one, g));
		b = _mm256_max_ps(zero, _mm256_min_ps(one, b));
	}
	inline Colorf_8 normalized() const {
		const auto zero = _mm256_set1_ps(0);
		const auto one = _mm256_set1_ps(1);
		return Colorf_8{_mm256_max_ps(zero, _mm256_min_ps(one, r)),
			_mm256_max_ps(zero, _mm256_min_ps(one, g)),
			_mm256_max_ps(zero, _mm256_min_ps(one, b))};
	}
};
inline Colorf_8 operator+(const Colorf_8 &a, const Colorf_8 &b){
	return Colorf_8{_mm256_add_ps(a.r, b.r), _mm256_add_ps(a.g, b.g),
		_mm256_add_ps(a.b, b.b)};
}
inline Colorf_8 operator-(const Colorf_8 &a, const Colorf_8 &b){
	return Colorf_8{_mm256_sub_ps(a.r, b.r), _mm256_sub_ps(a.g, b.g),
		_mm256_sub_ps(a.b, b.b)};
}
inline Colorf_8 operator*(const Colorf_8 &a, const Colorf_8 &b){
	return Colorf_8{_mm256_mul_ps(a.r, b.r), _mm256_mul_ps(a.g, b.g),
		_mm256_mul_ps(a.b, b.b)};
}
inline Colorf_8 operator*(const Colorf_8 &a, float s){
	const auto vs = _mm256_set1_ps(s);
	return Colorf_8{_mm256_mul_ps(a.r, vs), _mm256_mul_ps(a.g, vs),
		_mm256_mul_ps(a.b, vs)};
}
inline Colorf_8 operator*(float s, const Colorf_8 &a){
	const auto vs = _mm256_set1_ps(s);
	return Colorf_8{_mm256_mul_ps(a.r, vs), _mm256_mul_ps(a.g, vs),
		_mm256_mul_ps(a.b, vs)};
}
inline Colorf_8 operator/(const Colorf_8 &a, const Colorf_8 &b){
	return Colorf_8{_mm256_div_ps(a.r, b.r), _mm256_div_ps(a.g, b.g),
		_mm256_div_ps(a.b, b.b)};
}
inline Colorf_8 operator/(const Colorf_8 &c, float s){
	const auto vs = _mm256_rcp_ps(_mm256_set1_ps(s));
	return Colorf_8{_mm256_mul_ps(c.r, vs), _mm256_mul_ps(c.g, vs),
		_mm256_mul_ps(c.b, vs)};
}
inline std::ostream& operator<<(std::ostream &os, const Colorf_8 &c){
	os << "Colorf_8:\nr = " << c.r
		<< "\ng = " << c.g
		<< "\nb = " << c.b;
	return os;
}

#endif

