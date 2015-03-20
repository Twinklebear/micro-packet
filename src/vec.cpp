#include "vec.h"

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

