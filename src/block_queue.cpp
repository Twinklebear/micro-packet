#include <algorithm>
#include <iostream>
#include "block_queue.h"

// Fabian Giesen's Morton code generation
// See: http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
static uint32_t part1_by1(uint32_t x){
	x &= 0x0000ffff;
	x = (x ^ (x << 8)) & 0x00ff00ff;
	x = (x ^ (x << 4)) & 0x0f0f0f0f;
	x = (x ^ (x << 2)) & 0x33333333;
	x = (x ^ (x << 1)) & 0x55555555;
	return x;
}
static uint32_t morton2(uint32_t x, uint32_t y){
	return (part1_by1(y) << 1) + part1_by1(x);
}

BlockQueue::BlockQueue(uint32_t block_dim, uint32_t imgw, uint32_t imgh)
	: block_dim(block_dim), next_block(0)
{
	if (imgw % block_dim != 0 || imgh % block_dim != 0){
		std::cout << "BlockQueue WARNING: blocks don't evenly partition the image\n";
	}
	blocks.resize(imgw * imgh / (block_dim * block_dim), std::make_pair(0, 0));
	int blocks_per_row = imgw / block_dim;
	int b = 0;
	std::generate(blocks.begin(), blocks.end(),
		[&](){
			int i = b++;
			return std::make_pair(i % blocks_per_row, i / blocks_per_row);
		});
	std::sort(blocks.begin(), blocks.end(),
		[](const std::pair<uint32_t, uint32_t> &a, const std::pair<uint32_t, uint32_t> &b){
			return morton2(a.first, a.second) < morton2(b.first, b.second);
		});
}
std::pair<uint32_t, uint32_t> BlockQueue::next(){
	if (next_block >= blocks.size()){
		return end();
	}
	const auto b = blocks[next_block++];
	return std::make_pair(b.first * block_dim, b.second * block_dim);
}
std::pair<uint32_t, uint32_t> BlockQueue::end(){
	return std::make_pair(-1, -1);
}

