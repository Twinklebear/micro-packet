#include <iostream>
#include <cfloat>
#include "immintrin.h"

// Attempt to solve the quadratic equation. Returns a mask of successful solutions (0xff)
// and stores the computed t values in t0 and t1
__m256 solve_quadratic(const __m256 a, const __m256 b, const __m256 c, __m256 &t0, __m256 &t1){
	auto discrim = _mm256_sub_ps(_mm256_mul_ps(b, b),
			_mm256_mul_ps(_mm256_mul_ps(a, c), _mm256_set1_ps(4.f)));
	auto mask = _mm256_cmp_ps(discrim, _mm256_set1_ps(0), _CMP_LE_OQ);
	// TODO: Finish computing the t0 & t1 for those with mask = 0xff
	return mask;
}

// structure holding 8 vec3's
struct Vec8 {
	__m256 x, y, z;

	Vec8() : x(_mm256_set1_ps(0)), y(_mm256_set1_ps(0)), z(_mm256_set1_ps(0)){}
	Vec8(__m256 x, __m256 y, __m256 z) : x(x), y(y), z(z){}
	// Compute length^2 of all 8 vectors
	__m256 length_sqr() const {
		const auto x_sqr = _mm256_mul_ps(x, x);
		const auto y_sqr = _mm256_mul_ps(y, y);
		const auto z_sqr = _mm256_mul_ps(z, z);
		return _mm256_add_ps(x_sqr, _mm256_add_ps(y_sqr, z_sqr));
	}
	// Compute length of all 8 vectors
	__m256 length() const {
		return _mm256_sqrt_ps(length_sqr());
	}
	__m256 dot(const Vec8 &vb) const {
		const auto a = _mm256_mul_ps(x, vb.x);
		const auto b = _mm256_mul_ps(y, vb.y);
		const auto c = _mm256_mul_ps(z, vb.z);
		return _mm256_add_ps(a, _mm256_add_ps(b, c));
	}
};

// Packet of 8 rays
struct Ray8 {
	Vec8 o, d;
	__m256 t;

	Ray8() : t(_mm256_set1_ps(FLT_MAX)) {}
};

struct Sphere {
	float x, y, z, radius;

	Sphere(float x, float y, float z, float radius) : x(x), y(y), z(z), radius(radius) {}
	// Test 8 rays against the sphere, returns masks for the hits (0xff) and misses (0x00)
	__m256 intersect(const Ray8 &ray){
		// TODO change to handle sphere not at origin
		const auto a = ray.d.length_sqr();
		const auto b = _mm256_mul_ps(ray.d.dot(ray.o), _mm256_set1_ps(2.f));
		const auto c = _mm256_sub_ps(ray.o.dot(ray.o), _mm256_mul_ps(_mm256_set1_ps(radius), _mm256_set1_ps(radius)));

		return _mm256_set1_ps(0);
	}
};

int main(int, char**){
}

