#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <array>
#include <iomanip>
#include <cfloat>
#include "immintrin.h"
#include "vec.h"
#include "color.h"
#include "render_target.h"
#include "camera.h"

struct Sphere {
	float x, y, z, radius;

	Sphere(float x, float y, float z, float radius) : x(x), y(y), z(z), radius(radius) {}
	// Test 8 rays against the sphere, returns masks for the hits (0xff) and misses (0x00)
	__m256 intersect(Ray8 &ray) const {
		const auto center = Vec3f_8{x, y, z};
		const auto d = center - ray.o;
		const auto a = ray.d.length_sqr();
		const auto b = _mm256_mul_ps(ray.d.dot(d), _mm256_set1_ps(-2.f));
		const auto c = _mm256_sub_ps(d.dot(d), _mm256_mul_ps(_mm256_set1_ps(radius),
					_mm256_set1_ps(radius)));
		// Solve the quadratic equation and store the mask of potential hits
		// We'll update this mask as we discard other potential hits, eg. due to
		// the hit being beyond the ray's t value
		__m256 t0, t1;
		auto hits = solve_quadratic(a, b, c, t0, t1);
		// We want t0 to hold the nearest t value that is greater than ray.t_min
		auto swap_t = _mm256_cmp_ps(t0, ray.t_min, _CMP_LT_OQ);
		t0 = _mm256_blendv_ps(t0, t1, swap_t);
		// Check which rays are within the ray's t range
		auto in_range = _mm256_and_ps(_mm256_cmp_ps(t0, ray.t_min, _CMP_GT_OQ),
				_mm256_cmp_ps(t0, ray.t_max, _CMP_LT_OQ));
		hits = _mm256_and_ps(hits, in_range);
		// Check if all rays miss the sphere
		if (_mm256_movemask_ps(hits) == 0){
			return hits;
		}
		// Update t values for rays that did hit
		ray.t_max = _mm256_blendv_ps(ray.t_max, t0, hits);
		return hits;
	}
};

int main(int, char**){
	const uint32_t width = 800;
	const uint32_t height = 600;
	const auto sphere = Sphere{0, 0, 0, 1.25};
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

