#include "vec.h"


psimd::mask<> solve_quadratic(const psimd::pack<float> a, const psimd::pack<float> b,
		const psimd::pack<float> c, psimd::pack<float> &t0, psimd::pack<float> &t1)
{
	// TODO: I'm curious if the compiler will figure out these can be fmsubs and fmadds and so on
	auto discrim = b * b - 4.f * a * c;
	auto solvable = discrim > 0.f;
	// Test for the case where none of the equations can be solved (eg. none hit)
	if (psimd::none(solvable)) {
		return solvable;
	}
	// Compute +/-sqrt(discrim), setting -discrim where we have b < 0
	discrim = psimd::sqrt(discrim);
	auto neg_discrim = -discrim;
	// Blend the discriminants to pick the right +/- value
	// Find mask for this with b < 0 set to 1
	auto mask = b < 0.f;
	discrim = psimd::select(mask, neg_discrim, discrim);
	auto q = -0.5f * (b + discrim);
	auto x = q / a;
	auto y = c / q;
	// Find which elements have t0 > t1 and compute mask so we can swap them
	mask = x > y;
	t0 = psimd::select(mask, y, x);
	t1 = psimd::select(mask, x, y);
	return solvable;
}

