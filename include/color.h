#ifndef COLOR_H
#define COLOR_H

#include <cstdint>
#include <array>
#include <cassert>
#include <ostream>

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

// Compute sRGB values for some color channel
inline tsimd::vfloat convert_srgb(tsimd::vfloat v){
	const float a = 0.055f;
	const float b = 1.f / 2.4f;
	const auto mask = v <= 0.0031308f;
	// We need to compute both branches for the sRGB conversion then pick
	// the right one based on the mask
	const auto y = v * 12.92f;
	const auto z = 1.055f * tsimd::pow(v, b) - a;
	return tsimd::select(mask, z, y);
}

/*
 * Floating point color struct storing N RGB colors
 */
struct ColorfN {
	tsimd::vfloat r, g, b;

	/*
	 * Initialize the RGB values to the same value
	 */
	inline ColorfN(float c = 0) : r(c), g(c), b(c){}
	/*
	 * Create an RGB color
	 */
	inline ColorfN(float r, float g, float b) : r(r), g(g), b(b){}
	inline ColorfN(Colorf c) : r(c.r), g(c.g), b(c.b){}
	inline ColorfN(tsimd::vfloat r, tsimd::vfloat g, tsimd::vfloat b) : r(r), g(g), b(b){}
	/*
	 * Normalize the floating point color values to be clamped between 0-1
	 */
	inline void normalize(){
		const tsimd::vfloat zero(0.f);
		const tsimd::vfloat one(1.f); 
		r = tsimd::max(zero, tsimd::min(one, r));
		g = tsimd::max(zero, tsimd::min(one, g));
		b = tsimd::max(zero, tsimd::min(one, b));
	}
	inline ColorfN normalized() const {
		const tsimd::vfloat zero(0.f);
		const tsimd::vfloat one(1.f); 
		return ColorfN(tsimd::max(zero, tsimd::min(one, r)),
			tsimd::max(zero, tsimd::min(one, g)),
			tsimd::max(zero, tsimd::min(one, b)));
	}
	inline tsimd::vfloat& operator[](int i) {
		switch (i) {
			case 0: return r;
			case 1: return g;
			case 2: return b;
			default: assert(false); return b;
		}
	}
	inline const tsimd::vfloat& operator[](int i) const {
		switch (i) {
			case 0: return r;
			case 1: return g;
			case 2: return b;
			default: assert(false); return b;
		}
	}
};
inline ColorfN operator+(const ColorfN &a, const ColorfN &b){
	return ColorfN{a.r + b.r, a.g + b.g, a.b + b.b};
}
inline ColorfN operator-(const ColorfN &a, const ColorfN &b){
	return ColorfN{a.r - b.r, a.g - b.g, a.b - b.b};
}
inline ColorfN operator*(const ColorfN &a, const ColorfN &b){
	return ColorfN{a.r * b.r, a.g * b.g, a.b * b.b};
}
inline ColorfN operator*(const ColorfN &a, tsimd::vfloat s){
	return ColorfN{a.r * s, a.g * s, a.b * s};
}
inline ColorfN operator*(tsimd::vfloat s, const ColorfN &a){
	return ColorfN{a.r * s, a.g * s, a.b * s};
}
inline ColorfN operator/(const ColorfN &a, const ColorfN &b){
	return ColorfN{a.r / b.r, a.g / b.g, a.b / b.b};
}
inline ColorfN operator/(const ColorfN &c, tsimd::vfloat s){
	const auto vs = 1.f / s;
	return c * vs;
}
inline std::ostream& operator<<(std::ostream &os, const ColorfN &c){
	os << "ColorfN:\nr = " << c.r
		<< "\ng = " << c.g
		<< "\nb = " << c.b;
	return os;
}

#endif

