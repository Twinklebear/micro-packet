#ifndef BLOCK_QUEUE_H
#define BLOCK_QUEUE_H

#include <vector>
#include <cstdint>
#include <utility>

/*
 * Queue that hands out blocks of pixels to be rendered in Z-order
 */
class BlockQueue {
	// Dimensions of a single block
	uint32_t block_dim;
	// TODO: should be atomic when multithreading is added
	int next_block;
	// Block starting positions
	std::vector<std::pair<uint32_t, uint32_t>> blocks;

public:
	/*
	 * Construct a new block queue to  the image of [imgw, imgh]
	 * pixels with blocks of size [block_dim, block_dim]
	 */
	BlockQueue(uint32_t block_dim, uint32_t imgw, uint32_t imgh);
	/*
	 * Get the next block in the queue, returns (-1, -1) if all blocks
	 * have been taken
	 */
	std::pair<uint32_t, uint32_t> next();
	std::pair<uint32_t, uint32_t> end();
	uint32_t get_block_dim() const;
};

#endif

