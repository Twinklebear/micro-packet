#include <array>
#include <atomic>
#include <string>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <memory>
#include <cstdio>
#include "immintrin.h"
#include "render_target.h"

/*
 * Convenient wrapper for BMP header information for a 24bpp BMP
 */
#pragma pack(1)
struct BMPHeader {
	std::array<uint8_t, 2> header;
	uint32_t file_size;
	// 4 reserved bytes we don't care about
	uint32_t dont_care = 0;
	// Offset in the file to the pixel array
	uint32_t px_array = 54;
	uint32_t header_size = 40;
	std::array<int32_t, 2> dims;
	uint16_t color_planes = 1;
	uint16_t bpp = 24;
	uint32_t compression = 0;
	uint32_t img_size;
	std::array<int32_t, 2> res;
	uint32_t color_palette = 0;
	uint32_t important_colors = 0;

	BMPHeader(uint32_t img_size, int32_t w, int32_t h)
		: file_size(54 + img_size), img_size(img_size)
	{
		// Workaround hacks to init header, dims and res because MSVC
		// is too slow to implement C++11/14
		header[0] = 'B';
		header[1] = 'M';
		dims[0] = w;
		dims[1] = h;
		res[0] = 2835;
		res[1] = 2835;
	}
};

Pixel::Pixel() : r(0), g(0), b(0), weight(0){}
Pixel::Pixel(const Pixel &p) : r(p.r), g(p.g), b(p.b), weight(p.weight){}

RenderTarget::RenderTarget(uint32_t width, uint32_t height)
	: width(width), height(height), pixels(width * height){}
void RenderTarget::write_samples(const Vec2f_8 &p, const Colorf_8 &c, __m256 mask){
	// Compute the discrete pixel coordinates which the sample hits
	const auto img_p = p - Vec2f_8{0.5};
	const auto *img_x = (const float*)&img_p.x;
	const auto *img_y = (const float*)&img_p.y;
	const auto *cr = (const float*)&c.r;
	const auto *cg = (const float*)&c.g;
	const auto *cb = (const float*)&c.b;
	const auto write_mask = _mm256_movemask_ps(mask);
	for (int i = 0, mask = 1; i < 8; ++i, mask <<= 1){
		if (write_mask & mask){
			int ix = clamp(static_cast<int>(img_x[i]), 0, static_cast<int>(width) - 1);
			int iy = clamp(static_cast<int>(img_y[i]), 0, static_cast<int>(height) - 1);
			Pixel &p = pixels[iy * width + ix];
			p.r += cr[i];
			p.g += cg[i];
			p.b += cb[i];
			p.weight += 1;
		}
	}
}
bool RenderTarget::save_image(const std::string &file) const {
	// Compute the correct image from the saved pixel data and write
	// it to the desired file
	std::string file_ext = file.substr(file.rfind(".") + 1);
	if (file_ext == "ppm"){
		std::vector<Color24> img(width * height);
		get_colorbuf(img);
		return save_ppm(file, &img[0].r);
	}
	if (file_ext == "bmp"){
		std::vector<Color24> img(width * height);
		get_colorbuf(img);
		//Do y-flip for BMP since BMP starts at the bottom-left
		for (uint32_t y = 0; y < height / 2; ++y){
			Color24 *a = &img[y * width];
			Color24 *b = &img[(height - y - 1) * width];
			for (uint32_t x = 0; x < width; ++x){
				std::swap(a[x], b[x]);
			}
		}
		// We also need to convert to BGRA order for BMP
		for (auto &c : img){
			std::swap(c.r, c.b);
		}
		return save_bmp(file, &img[0].r);
	}
	std::cout << "Unsupported output image format: " << file_ext << std::endl;
	return false;
}
uint32_t RenderTarget::get_width() const {
	return width;
}
uint32_t RenderTarget::get_height() const {
	return height;
}
void RenderTarget::get_colorbuf(std::vector<Color24> &img) const { 
	// Compute the correct image from the saved pixel data
	img.resize(width * height);
	for (uint32_t y = 0; y < height; ++y){
		for (uint32_t x = 0; x < width; ++x){
			const Pixel &p = pixels[y * width + x];
			if (p.weight != 0){
				Colorf c{p.r, p.g, p.b};
				c /= p.weight;
				c.normalize();
				img[y * width + x] = c.to_sRGB();
			}
		}
	}
}
bool RenderTarget::save_ppm(const std::string &file, const uint8_t *data) const {
	FILE *fp = fopen(file.c_str(), "wb");
	if (!fp){
		std::cerr << "RenderTarget::save_ppm Error: failed to open file "
			<< file << std::endl;
		return false;
	}
	fprintf(fp, "P6\n%d %d\n255\n", static_cast<int>(width), static_cast<int>(height));
	if (fwrite(data, 1, 3 * width * height, fp) != 3 * width * height){
		fclose(fp);
		return false;
	}
	fclose(fp);
	return true;
}
bool RenderTarget::save_bmp(const std::string &file, const uint8_t *data) const {
	FILE *fp = fopen(file.c_str(), "wb");
	if (!fp){
		std::cerr << "RenderTarget::save_bmp Error: failed to open file "
			<< file << std::endl;
		return false;
	}
	uint32_t w = width, h = height;
	BMPHeader bmp_header{4 * w * h, static_cast<int32_t>(w),
		static_cast<int32_t>(h)};
	if (fwrite(&bmp_header, sizeof(BMPHeader), 1, fp) != 1){
		fclose(fp);
		return false;
	}
	// Write each row follwed by any necessary padding
	uint32_t padding = (w * 3) % 4;
	for (uint32_t r = 0; r < h; ++r){
		if (fwrite(data + 3 * w * r, 1, 3 * w, fp) != 3 * w){
			fclose(fp);
			return false;
		}
		if (padding != 0){
			if (fwrite(data, 1, padding, fp) != padding){
				fclose(fp);
				return false;
			}
		}

	}
	fclose(fp);
	return true;
}

