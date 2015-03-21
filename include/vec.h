#ifndef VEC_H
#define VEC_H

#include <cmath>
#include <ostream>
#include <cfloat>
#include "immintrin.h"

// output operator for debugging vectors
std::ostream& operator<<(std::ostream &os, const __m256 &v);
std::ostream& operator<<(std::ostream &os, const __m256i &v);

template<typename T>
inline T clamp(T x, T min, T max){
	return x < min ? min : x > max ? max : x;
}

// Attempt to solve the quadratic equation. Returns a mask of successful solutions (0xff)
// and stores the computed t values in t0 and t1
__m256 solve_quadratic(const __m256 a, const __m256 b, const __m256 c, __m256 &t0, __m256 &t1);

// A single vec3f
struct Vec3f {
	float x, y, z;

	inline Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z){}
	inline float dot(const Vec3f &v) const {
		return x * v.x + y * v.y + z * v.z;
	}
	inline Vec3f cross(const Vec3f &v) const {
		return Vec3f{y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
	}
	inline float length_sqr() const {
		return x * x + y * y + z * z;
	}
	inline float length() const {
		return std::sqrt(length_sqr());
	}
	inline Vec3f normalized() const {
		const float s = 1.f / length();
		return Vec3f{x * s, y * s, z * s};
	}
};
inline Vec3f operator+(const Vec3f &a, const Vec3f &b){
	return Vec3f{a.x + b.x, a.y + b.y,  a.z + b.z};
}
inline Vec3f operator-(const Vec3f &a, const Vec3f &b){
	return Vec3f{a.x - b.x, a.y - b.y,  a.z - b.z};
}
inline Vec3f operator*(const Vec3f &v, float s){
	return Vec3f{v.x * s, v.y * s, v.z * s};
}
inline Vec3f operator*(float s, const Vec3f &v){
	return v * s;
}
inline Vec3f operator/(const Vec3f &v, float s){
	return v * (1 / s);
}
inline Vec3f operator/(const Vec3f &a, const Vec3f &b){
	return Vec3f{a.x / b.x, a.y / b.y, a.z / b.z};
}
inline Vec3f operator-(const Vec3f &v){
	return Vec3f{-v.x, -v.y, -v.z};
}
inline std::ostream& operator<<(std::ostream &os, const Vec3f &v){
	os << "Vec3f:\n\tx = " << v.x
		<< "\n\ty = " << v.y
		<< "\n\tz = " << v.z;
	return os;
}

// Struct holding 8 vec3f's
struct Vec3f_8 {
	__m256 x, y, z;

	inline Vec3f_8(float x = 0, float y = 0, float z = 0)
		: x(_mm256_set1_ps(x)), y(_mm256_set1_ps(y)), z(_mm256_set1_ps(z)){}
	inline Vec3f_8(Vec3f v)
		: x(_mm256_set1_ps(v.x)), y(_mm256_set1_ps(v.y)), z(_mm256_set1_ps(v.z)){}
	inline Vec3f_8(__m256 x, __m256 y, __m256 z) : x(x), y(y), z(z){}
	// Compute length^2 of all 8 vectors
	inline __m256 length_sqr() const {
		const auto x_sqr = _mm256_mul_ps(x, x);
		const auto y_sqr = _mm256_mul_ps(y, y);
		const auto z_sqr = _mm256_mul_ps(z, z);
		return _mm256_add_ps(x_sqr, _mm256_add_ps(y_sqr, z_sqr));
	}
	// Compute length of all 8 vectors
	inline __m256 length() const {
		return _mm256_sqrt_ps(length_sqr());
	}
	// Normalize all 8 vectors
	inline void normalize(){
		const auto len = length();
		x = _mm256_div_ps(x, len);
		y = _mm256_div_ps(y, len);
		z = _mm256_div_ps(z, len);
	}
	inline __m256 dot(const Vec3f_8 &vb) const {
		const auto a = _mm256_mul_ps(x, vb.x);
		const auto b = _mm256_mul_ps(y, vb.y);
		const auto c = _mm256_mul_ps(z, vb.z);
		return _mm256_add_ps(a, _mm256_add_ps(b, c));
	}
};
inline Vec3f_8 operator+(const Vec3f_8 &a, const Vec3f_8 &b){
	return Vec3f_8{_mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y), _mm256_add_ps(a.z, b.z)};
}
inline Vec3f_8 operator-(const Vec3f_8 &a, const Vec3f_8 &b){
	return Vec3f_8{_mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y),
		_mm256_sub_ps(a.z, b.z)};
}
inline Vec3f_8 operator-(const Vec3f_8 &a){
	const auto neg = _mm256_set1_ps(-1);
	return Vec3f_8{_mm256_mul_ps(a.x, neg), _mm256_mul_ps(a.y, neg),
		_mm256_mul_ps(a.z, neg)};
}
// Scale all components of the vector
inline Vec3f_8 operator*(__m256 s, const Vec3f_8 &v){
	return Vec3f_8{_mm256_mul_ps(s, v.x), _mm256_mul_ps(s, v.y), _mm256_mul_ps(s, v.z)};
}
inline std::ostream& operator<<(std::ostream &os, const Vec3f_8 &v){
	os << "Vec3f_8:\n\tx = " << v.x
		<< "\n\ty = " << v.y
		<< "\n\tz = " << v.z;
	return os;
}

// Struct holding 8 vec2f's
struct Vec2f_8 {
	__m256 x, y;

	inline Vec2f_8(float x = 0, float y = 0) : x(_mm256_set1_ps(x)), y(_mm256_set1_ps(y)){}
	inline Vec2f_8(__m256 x, __m256 y) : x(x), y(y){}
};
inline Vec2f_8 operator-(const Vec2f_8 &a, const Vec2f_8 &b){
	return Vec2f_8{_mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y)};
}
inline Vec2f_8 operator/(const Vec2f_8 &a, const Vec2f_8 &b){
	return Vec2f_8{_mm256_div_ps(a.x, b.x), _mm256_div_ps(a.y, b.y)};
}
inline std::ostream& operator<<(std::ostream &os, const Vec2f_8 &v){
	os << "Vec2f_8:\n\tx = " << v.x
		<< "\n\ty = " << v.y;
	return os;
}


// Packet of 8 rays
struct Ray8 {
	Vec3f_8 o, d;
	__m256 t_min, t_max;

	Ray8(Vec3f_8 o = Vec3f_8{}, Vec3f_8 d = Vec3f_8{}, float t_min = 0, float t_max = INFINITY)
		: o(o), d(d), t_min(_mm256_set1_ps(t_min)), t_max(_mm256_set1_ps(t_max))
	{}
	Vec3f_8 at(__m256 t) const {
		return o + t * d;
	}
};
inline std::ostream& operator<<(std::ostream &os, const Ray8 &r){
	os << "Ray8:\no = " << r.o
		<< "\nd = " << r.d
		<< "\nt_max = " << r.t_max
		<< "\nt_min = " << r.t_min;
	return os;
}

#endif

