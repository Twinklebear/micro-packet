#include <iostream>
#include <algorithm>
#include "vec.h"
#include "ld_sampler.h"

static void sample2d(int n, uint32_t x, uint32_t y, float *s0, float *s1, int offset = 0);
static void sample02(uint32_t n, const uint32_t x, const uint32_t y, float *s0, float *s1);
static float van_der_corput(uint32_t n, uint32_t scramble);
static float sobol2(uint32_t n, uint32_t scramble);
/*
 * Round x up to the nearest power of 2
 * Based off Stephan Brumme's method
 * http://bits.stephan-brumme.com/roundUpToNextPowerOfTwo.html
 */
static inline uint32_t round_up_pow2(uint32_t x);

// We set start's y coord so that we'll report we don't have any samples until a block is selected
LDSampler::LDSampler(uint32_t sp, uint32_t block_dim) : spp(std::max(round_up_pow2(sp), uint32_t{8})),
	block_dim(block_dim), samples_taken(0), start({0, block_dim}), current({0, 0})
{
	if (sp < 8){
		std::cout << "Warning: LDSampler only supports taking 8 or more samples per pixel,"
			<< " will take " << spp << "spp\n";
	}
	else if (sp != spp){
		std::cout << "Warning: LDSampler only takes power of 2 samples per pixel, rounded up to"
			<< " take " << spp << "spp\n";
	}
}
void LDSampler::select_block(const std::pair<uint32_t, uint32_t> &b){
	start = b;
	current = b;
	samples_taken = 0;
}
bool LDSampler::has_samples() const {
	return current.second != start.second + block_dim;
}
psimd::mask<> LDSampler::sample(std::mt19937 &rng, Vec2fN &samples){
	if (!has_samples()){
		return psimd::mask<>(0);
	}
	PSIMD_ALIGN(16) float x[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
	PSIMD_ALIGN(16) float y[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

	// Take at most 8 samples per sampling pass since that's how many we can
	// fit into a packet
	int n = spp - samples_taken;
	if (n > 8){
		n = 8;
	}
	sample2d(n, distrib(rng), distrib(rng), x, y, samples_taken);
	std::shuffle(x, x + n, rng);
	std::shuffle(y, y + n, rng);
	samples.x = psimd::load<psimd::pack<float>>(x);
	samples.y = psimd::load<psimd::pack<float>>(y);
	// We use -1 to signal that there is no sample to be taken for the lane, so
	// compute mask of those samples which shouldn't be used
	const auto active = samples.x > -0.5f;
	samples += Vec2fN{static_cast<float>(current.first), static_cast<float>(current.second)};

	samples_taken += n;
	// We're done sampling this pixel, move to the next one
	if (samples_taken >= spp){
		samples_taken = 0;
		++current.first;
		if (current.first == start.first + block_dim){
			current.first = start.first;
			++current.second;
		}
	}
	return active;
}
void sample2d(int n, uint32_t x, uint32_t y, float *s0, float *s1, int offset){
	for (int i = 0; i < n; ++i){
		sample02(i + offset, x, y, s0 + i, s1 + i);
	}
}
void sample02(uint32_t n, const uint32_t x, const uint32_t y, float *s0, float *s1){
	*s0 = van_der_corput(n, x);
	*s1 = sobol2(n, y);
}
float van_der_corput(uint32_t n, uint32_t scramble){
	n = (n << 16) | (n >> 16);
	n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
	n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
	n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
	n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
	n ^= scramble;
	return ((n >> 8) & 0xffffff) / float{1 << 24};
}
float sobol2(uint32_t n, uint32_t scramble){
	for (uint32_t i = uint32_t{1} << 31; n != 0; n >>= 1, i ^= i >> 1){
		if (n & 0x1){
			scramble ^= i;
		}
	}
	return ((scramble >> 8) & 0xffffff) / float{1 << 24};
}
inline uint32_t round_up_pow2(uint32_t x){
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

