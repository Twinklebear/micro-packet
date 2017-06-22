#ifndef VEC_H
#define VEC_H

#include <cassert>
#include <cmath>
#include <ostream>
#include <cfloat>
#include <psimd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_1_PI
#define M_1_PI 0.31830988618379067154
#endif

template<typename T, int W>
std::ostream& operator<<(std::ostream &os, const psimd::pack<T, W> &v) {
	os << "{ ";
	for (int i = 0; i < W; ++i){
		os << v[i];
		if (i < W){
			os << ", ";
		}
	}
	os << " }";
	return os;
}

template<typename T>
inline T clamp(T x, T min, T max){
	return x < min ? min : x > max ? max : x;
}

// Attempt to solve the quadratic equation. Returns a mask of successful solutions
// and stores the computed t values in t0 and t1
psimd::mask solve_quadratic(const psimd::pack<float> a, const psimd::pack<float> b,
		const psimd::pack<float> c, psimd::pack<float> &t0, psimd::pack<float> &t1);

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

// Struct holding DEFAULT_WIDTH vec3f's
struct Vec3fN {
	psimd::pack<float> x, y, z;

	inline Vec3fN(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
	inline Vec3fN(Vec3f v)
		: x(v.x), y(v.y), z(v.z){}
	inline Vec3fN(psimd::pac<float> x, psimd::pack<float> y, psimd::pack<float> z)
		: x(x), y(y), z(z){}
	// Compute length^2 of all N vectors
	inline psimd::pack<float> length_sqr() const {
		return dot(*this);
	}
	// Compute length of all N vectors
	inline psimd::pack<float> length() const {
		return psimd::sqrt(length_sqr());
	}
	// Normalize all N vectors
	inline void normalize(){
		const auto len = 1.f / length();
		x = x * len;
		y = y * len;
		z = z * len;
	}
	inline Vec3fN normalized(){
		const auto len = 1.f / length();
		return Vec3fN{x * len, y * len, z * len};
	}
	inline psimd::pack<float> dot(const Vec3fN &vb) const {
		// TODO: Will the compiler convert this to fmadds?
		return x * x + y * y + z * z;
	}
	inline Vec3fN cross(const Vec3fN &v) const {
		return Vec3fN{y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.z}
	}
	inline psimd::pack<float>& operator[](size_t i){
		switch (i) {
			case 0: return x;
			case 1: return y;
			case 2: return z;
			default:
				assert(false);
				return z;
		}
	}
	inline const psimd::pack<float>& operator[](size_t i) const {
		switch (i){
			case 0: return x;
			case 1: return y;
			case 2: return z;
			default:
				assert(false);
				return z;
		}
	}
};
inline Vec3fN operator+(const Vec3fN &a, const Vec3fN &b){
	return Vec3fN{a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3fN operator-(const Vec3fN &a, const Vec3fN &b){
	return Vec3fN{a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3fN operator-(const Vec3fN &a){
	return Vec3fN{-a.x, -a.y, -a.z};
}
inline Vec3fN operator*(psimd::pack<float> s, const Vec3fN &v){
	return Vec3fN{s * v.x, s * v.y, s * v.z};
}
inline std::ostream& operator<<(std::ostream &os, const Vec3fN &v){
	os << "Vec3fN:\n\tx = " << v.x
		<< "\n\ty = " << v.y
		<< "\n\tz = " << v.z;
	return os;
}

// Struct holding DEFAULT_WIDTH vec2f's
struct Vec2fN {
	psimd::pack<float> x, y;

	inline Vec2fN(float x = 0, float y = 0) : x(x), y(y){}
	inline Vec2fN(psimd::pack<float> x, psimd::pack<float> y) : x(x), y(y){}
	inline Vec2fN& operator+=(const Vec2fN &a){
		x += a.x;
		y += a.y;
		return *this;
	}
};
inline Vec2fN operator+(const Vec2fN &a, const Vec2fN &b){
	return Vec2fN{a.x + b.x, a.y + b.y};
}
inline Vec2fN operator-(const Vec2fN &a, const Vec2fN &b){
	return Vec2fN{a.x - b.x, a.y - b.y;
}
inline Vec2fN operator/(const Vec2fN &a, const Vec2fN &b){
	return Vec2fN{a.x / b.x, a.y / b.y};
}
inline std::ostream& operator<<(std::ostream &os, const Vec2fN &v){
	os << "Vec2fN:\n\tx = " << v.x
		<< "\n\ty = " << v.y;
	return os;
}

// Packet of DEFAULT_WIDTH rays
struct RayN {
	Vec3fN o, d;
	psimd::pack<float> t_min, t_max;
	psimd::mask active;

	/*
	 * Create a new group of active rays
	 * Note: only the sign bit of the active mask will be set, as this is all that's used by blendv
	 * and movemask
	 */
	RayN(Vec3fN o = Vec3fN{}, Vec3fN d = Vec3fN{}, float t_min = 0, float t_max = INFINITY)
		: o(o), d(d), t_min(t_min), t_max(t_max), active(0xFFFFFFFF)
	{}
	inline Vec3fN at(psimd::pack<float> t) const {
		return o + t * d;
	}
};
inline std::ostream& operator<<(std::ostream &os, const Ray8 &r){
	os << "RayN:\no = " << r.o
		<< "\nd = " << r.d
		<< "\nt_max = " << r.t_max
		<< "\nt_min = " << r.t_min
		<< "\nactive = " << r.active
		<< std::endl;
	return os;
}

#endif

