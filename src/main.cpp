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
	// Test for the case where none of the equations can be solved (eg. none hit)
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
	return solved;
}

// Struct holding 8 vec3's
struct Vec8 {
	__m256 x, y, z;

	Vec8(float x = 0, float y = 0, float z = 0)
		: x(_mm256_set1_ps(x)), y(_mm256_set1_ps(y)), z(_mm256_set1_ps(z))
	{}
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
	__m256 t_max, t_min;

	Ray8(Vec8 o, Vec8 d) : o(o), d(d), t_max(_mm256_set1_ps(FLT_MAX)),
		t_min(_mm256_set1_ps(0))
	{}
};

struct Sphere {
	float x, y, z, radius;

	Sphere(float x, float y, float z, float radius) : x(x), y(y), z(z), radius(radius) {}
	// Test 8 rays against the sphere, returns masks for the hits (0xff) and misses (0x00)
	__m256 intersect(Ray8 &ray) const {
		// TODO change to handle sphere not at origin
		const auto a = ray.d.length_sqr();
		const auto b = _mm256_mul_ps(ray.d.dot(ray.o), _mm256_set1_ps(2.f));
		const auto c = _mm256_sub_ps(ray.o.dot(ray.o), _mm256_mul_ps(_mm256_set1_ps(radius),
					_mm256_set1_ps(radius)));
		// Solve the quadratic equation and store the mask of potential hits
		// We'll update this mask as we discard other potential hits, eg. due to
		// the hit being beyond the ray's t value
		std::cout << "a = " << a
			<< "\nb = " << b
			<< "\nc = " << c << std::endl;
		__m256 t0, t1;
		auto hits = solve_quadratic(a, b, c, t0, t1);
		std::cout << "solved quadratic:"
			<< "\nhits = " << hits
			<< "\nt0 = " << t0
			<< "\nt1 = " << t1 << std::endl;
		// We want t0 to hold the nearest t value that is greater than ray.t_min
		auto swap_t = _mm256_cmp_ps(t0, ray.t_min, _CMP_GT_OQ);
		t0 = _mm256_blendv_ps(t0, t1, swap_t);
		// Check which rays are within the ray's t range
		auto in_range = _mm256_and_ps(_mm256_cmp_ps(t0, ray.t_min, _CMP_GT_OQ),
				_mm256_cmp_ps(t0, ray.t_max, _CMP_LT_OQ));
		hits = _mm256_and_ps(hits, in_range);
		// Check if all rays miss the sphere
		if ((_mm256_movemask_ps(hits) & 255) == 0){
			return hits;
		}
		// Update t values for rays that did hit
		ray.t_max = _mm256_blendv_ps(ray.t_max, t0, hits);
		return hits;
	}
};

int main(int, char**){
	const auto sphere = Sphere{0, 0, 0, 1};
	auto packet = Ray8{Vec8{0, 0, -2}, Vec8{0, 0, 1}};
	auto hits = sphere.intersect(packet);
	std::cout << "Expect all hits, hits = " << hits << std::endl;

	packet = Ray8{Vec8{0, 0, -2}, Vec8{0, 0, -1}};
	hits = sphere.intersect(packet);
	std::cout << "Expect all miss, hits = " << hits << std::endl;
}

