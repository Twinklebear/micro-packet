#ifndef RENDER_TARGET_H
#define RENDER_TARGET_H

#include <string>
#include <vector>
#include <atomic>
#include <memory>
#include "vec.h"
#include "color.h"

/*
 * A pixel stored in the image being rendered to track pixel
 * luminance and weight for reconstruction
 * Because we need to deal with multi-thread synchronization the
 * rgb values and weights are stored as atomics
 */
struct Pixel {
	float r, g, b, weight;

	Pixel();
	Pixel(const Pixel &p);
};

/*
 * The render target where pixel data is stored for the rendered scene
 * along with for some reason a depth buffer is required for proj1?
 */
class RenderTarget {
	uint32_t width, height;
	std::vector<Pixel> pixels;

public:
	/*
	 * Create a render target with width * height pixels
	 */
	RenderTarget(uint32_t width, uint32_t height);
	/*
	 * Write a color samples to the image, the mask will specify
	 * which color should actually be stored (0xff to store)
	 */
	void write_samples(const Vec2f_8 &p, const Colorf_8 &c, __m256 mask);
	//Save the image or depth buffer to the desired file
	bool save_image(const std::string &file) const;
	uint32_t get_width() const;
	uint32_t get_height() const;
	/*
	 * Get a snapshot of the color buffer at the moment
	 * stored in img
	 */
	void get_colorbuf(std::vector<Color24> &img) const;

private:
	/*
	 * Save color data as a PPM image to the file, data should be
	 * RGB8 data and have width * height elements
	 */
	bool save_ppm(const std::string &file, const uint8_t *data) const;
	/*
	 * Save color data as a BMP image to the file, data should be
	 * RGBA8 data and have width * height elements
	 * The image data should be flipped appropriately already for the BMP
	 * file format (eg. starting at bottom left)
	 */
	bool save_bmp(const std::string &file, const uint8_t *data) const;
};

#endif

