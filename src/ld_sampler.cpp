#include <algorithm>
#include "vec.h"
#include "ld_sampler.h"

static void sample2d(int n, uint32_t x, uint32_t y, float *s0, float *s1, int offset = 0);
static void sample02(uint32_t n, const uint32_t x, const uint32_t y, float *s0, float *s1);
static float van_der_corput(uint32_t n, uint32_t scramble);
static float sobol2(uint32_t n, uint32_t scramble);

// We set start's y coord so that we'll report we don't have any samples until a block is selected
LDSampler::LDSampler(uint32_t spp, uint32_t block_dim) : spp(spp), block_dim(block_dim),
	samples_taken(0), start({0, block_dim}), current({0, 0}){}
void LDSampler::select_block(const std::pair<uint32_t, uint32_t> &b){
	start = b;
	current = b;
	samples_taken = 0;
}
bool LDSampler::has_samples() const {
	return current.second != start.second + block_dim;
}
__m256 LDSampler::sample(std::mt19937 &rng, Vec2f_8 &samples){
	if (!has_samples()){
		return _mm256_set1_ps(0.f);
	}
	// Cases to handle:
	// - Our samples per pixel are less than 8 so we need to sample multiple pixels
	//   TODO: Is it worth supporting this case?
	// - Our samples per pixel are == 8 so we just take samples for one pixel
	// - Our samples per pixel are > 8 so we need to resume sampling and track how
	// 	 many we took in the current pixel. We may also end up spilling into another pixel
	// 	 if the sampling rate is > 8 but not a multiple of it.
	CACHE_ALIGN float x[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
	CACHE_ALIGN float y[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

	// Take at most 8 samples per sampling pass since that's how many we can
	// fit into a packet
	int n = spp - samples_taken;
	if (n > 8){
		n = 8;
	}
	sample2d(n, distrib(rng), distrib(rng), x, y, samples_taken);
	std::shuffle(x, x + n, rng);
	std::shuffle(y, y + n, rng);
	samples.x = _mm256_load_ps(x);
	samples.y = _mm256_load_ps(y);
	// We use -1 to signal that there is no sample to be taken for the lane, so
	// compute mask of those samples which shouldn't be used
	const auto active = _mm256_cmp_ps(samples.x, _mm256_set1_ps(-0.5f), _CMP_GT_OQ);
	samples += Vec2f_8{static_cast<float>(current.first), static_cast<float>(current.second)};

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

