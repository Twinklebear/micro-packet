#include "immintrin.h"
#include "vec.h"
#include "color.h"
#include "render_target.h"
#include "camera.h"
#include "sphere.h"

int main(int, char**){
	const uint32_t width = 800;
	const uint32_t height = 600;
	const auto sphere = Sphere{Vec3f{0}, 1.25};
	const auto camera = PerspectiveCamera{Vec3f{0, 0, -3}, Vec3f{0, 0, 0}, Vec3f{0, 1, 0},
		60.f, static_cast<float>(width) / height};
	auto target = RenderTarget{width, height};
	const auto img_dim = Vec2f_8{static_cast<float>(width), static_cast<float>(height)};
	// TODO: Z-order 4x4 blocks
	const int n_pixels = width * height;
	for (int i = 0; i < n_pixels; i += 8){
		std::array<float, 8> pixel_x, pixel_y;
		for (int j = 0; j < 8; ++j){
			pixel_x[j] = 0.5 + i % width + j;
			pixel_y[j] = 0.5 + i / width;
		}
		const auto samples = Vec2f_8{_mm256_loadu_ps(pixel_x.data()),
			_mm256_loadu_ps(pixel_y.data())};
		Ray8 packet;
		camera.generate_rays(packet, samples / img_dim);

		auto hits = sphere.intersect(packet);
		target.write_samples(samples, Colorf_8{1}, hits);
	}
	target.save_image("out.bmp");
}

