#include <iostream>
#include <array>
#include <iomanip>
#include <cfloat>
#include "immintrin.h"

std::ostream& operator<<(std::ostream &os, const __m256 &v){
	float *f = (float*)&v;
	os << "{ ";
	for (int i = 0; i < 8; ++i){
		os << f[i];
		if (i < 7){
			os << ", ";
		}
	}
	os << " }";
	return os;
}

// Attempt to solve the quadratic equation. Returns a mask of successful solutions (0xff)
// and stores the computed t values in t0 and t1
__m256 solve_quadratic(const __m256 a, const __m256 b, const __m256 c, __m256 &t0, __m256 &t1){
	auto discrim = _mm256_sub_ps(_mm256_mul_ps(b, b),
			_mm256_mul_ps(_mm256_mul_ps(a, c), _mm256_set1_ps(4.f)));
	auto solved = _mm256_cmp_ps(discrim, _mm256_set1_ps(0), _CMP_GT_OQ);
	std::cout << "discrim = " << discrim << "\n";
	std::cout << "solved = " << solved << "\n";
	// Test for the case where none of the equations can be solved (eg. none hit)
	// TODO: Is this & necessary?
	if ((_mm256_movemask_ps(solved) & 255) == 0){
		return solved;
	}
	// Compute +/-sqrt(discrim), setting -discrim where we have b < 0
	discrim = _mm256_sqrt_ps(discrim);
	auto neg_discrim = _mm256_mul_ps(discrim, _mm256_set1_ps(-1.f));
	// Blend the discriminants to pick the right +/- value
	// Find mask for this with b < 0 set to 1
	auto mask = _mm256_cmp_ps(b, _mm256_set1_ps(0), _CMP_LT_OQ);
	discrim = _mm256_blendv_ps(discrim, neg_discrim, mask);
	auto q = _mm256_mul_ps(_mm256_set1_ps(-0.5f), _mm256_add_ps(b, discrim));
	auto x = _mm256_div_ps(q, a);
	auto y = _mm256_div_ps(c, q);
	// Find which elements have t0 > t1 and compute mask so we can swap them
	mask = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
	t0 = _mm256_blendv_ps(x, y, mask);
	t1 = _mm256_blendv_ps(y, x, mask);
	std::cout << "t0 = " << t0
		<< "\nt1 = " << t1
		<< std::endl;
	return solved;
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

std::ostream& operator<<(std::ostream &os, const Vec8 &v){
	os << "Vec8:\n\tx = " << v.x
		<< "\n\ty = " << v.y
		<< "\n\tz = " << v.z
		<< "\n";
	return os;
}

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
		const auto c = _mm256_sub_ps(ray.o.dot(ray.o), _mm256_mul_ps(_mm256_set1_ps(radius),
					_mm256_set1_ps(radius)));

		return _mm256_set1_ps(0);
	}
};

int main(int, char**){
	std::array<float, 8> a, b, c;
	a.fill(2);
	b.fill(4);
	c.fill(-4);
	__m256 va = _mm256_loadu_ps(a.data());
	__m256 vb = _mm256_loadu_ps(b.data());
	__m256 vc = _mm256_loadu_ps(c.data());
	__m256 t0 = _mm256_set1_ps(0);
	__m256 t1 = _mm256_set1_ps(0);
	__m256 solved = solve_quadratic(va, vb, vc, t0, t1);
	const int solve_mask = (_mm256_movemask_ps(solved) & 255);
	std::cout << "Solve mask: 0x" << std::hex << solve_mask << std::dec << "\n";
	std::cout << "t0 = " << t0
		<< "\nt1 = " << t1
		<< std::endl;
}

