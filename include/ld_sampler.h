#ifndef LD_SAMPLER_H
#define LD_SAMPLER_H

#include <random>
#include <cstdint>
#include "vec.h"

/*
 * A low discrepancy sampler based on (0, 2) sequences, as described in PBRT
 */
class LDSampler {
	uint32_t spp, block_dim;
	// Number of samples we've taken so far in the current pixel, since
	// we may be taking more than 8 samples per pixel and will need to resume
	uint32_t samples_taken;
	// The current block being sampled
	std::pair<uint32_t, uint32_t> start, current;
	std::uniform_int_distribution<uint32_t> distrib;

public:
	/*
	 * Creat a new Low Discrepancy sampler to take spp samples per pixel over
	 * blocks of pixels that are block_dim x block_dim
	 */
	LDSampler(uint32_t spp, uint32_t block_dim);
	/*
	 * Select a new block to start sampling
	 */
	void select_block(const std::pair<uint32_t, uint32_t> &b);
	/*
	 * Check if the sampler has more samples left to take
	 */
	bool has_samples() const;
	/*
	 * Compute up to 8 pixel samples in the block being sampled and return them
	 * Note: fewer than 8 samples may be generated if the sampler runs out of
	 * samples to take over the block, in which case some lanes will be masked
	 * off in the active mask returned and the masked off components will
	 * have (-1, -1) as the pixel sample position
	 */
	tsimd::vmask sample(std::mt19937 &rng, Vec2fN &samples);
};

#endif

