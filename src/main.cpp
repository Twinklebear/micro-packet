#include <iostream>
#include <vector>
#include <memory>
#include "geometry.h"
#include "immintrin.h"
#include "vec.h"
#include "color.h"
#include "render_target.h"
#include "camera.h"
#include "sphere.h"
#include "plane.h"
#include "diff_geom.h"
#include "material.h"
#include "light.h"
#include "occlusion_tester.h"
#include "scene.h"

int main(int, char**){
	const uint32_t width = 800;
	const uint32_t height = 600;
	const auto scene = Scene{
		{
			std::make_shared<Sphere>(Vec3f{0}, 0.5f, 0),
			std::make_shared<Plane>(Vec3f{0, -0.5, 0}, Vec3f{0, 1, 0}, 1)
		},
		{
			std::make_shared<LambertianMaterial>(Colorf{1, 0, 0}),
			std::make_shared<LambertianMaterial>(Colorf{0, 0, 1})
		},
		PointLight{Vec3f{1, 1, -2}, Colorf{50}}
	};

	const auto camera = PerspectiveCamera{Vec3f{0, 0, -3}, Vec3f{0, 0, 0}, Vec3f{0, 1, 0},
		60.f, static_cast<float>(width) / height};
	auto target = RenderTarget{width, height};
	const auto img_dim = Vec2f_8{static_cast<float>(width), static_cast<float>(height)};

	// TODO: Z-order 4x4 blocks?
	const int n_pixels = width * height;
	for (int i = 0; i < n_pixels; i += 8){
		std::array<float, 8> pixel_x, pixel_y;
		for (int j = 0; j < 8; ++j){
			pixel_x[j] = 0.5f + i % width + j;
			pixel_y[j] = 0.5f + i / width;
		}
		const auto samples = Vec2f_8{_mm256_loadu_ps(pixel_x.data()),
			_mm256_loadu_ps(pixel_y.data())};
		Ray8 packet;
		camera.generate_rays(packet, samples / img_dim);

		DiffGeom8 dg;
		auto hits = scene.intersect(packet, dg);
		// If we hit something, shade it
		if (_mm256_movemask_ps(hits) != 0){
			auto color = Colorf_8{0};
			// This is pretty ugly. TODO: Better approach to shading??
			for (uint32_t i = 0; i < scene.materials.size(); ++i){
				// Is there a better way to just get the bits of one register casted to another?
				// it seems like the or operations don't let you mix either
				const auto use_mat = _mm256_cmpeq_epi32(dg.material_id, _mm256_set1_epi32(i));
				auto shade_mask = *(__m256*)&use_mat;
				if (_mm256_movemask_ps(shade_mask) != 0){
					const auto w_o = -packet.d;
					Vec3f_8 w_i{0};
					OcclusionTester occlusion;
					const auto li = scene.light.sample(dg.point, w_i, occlusion);
					// We just need to flip the sign bit to change occluded mask to unoccluded mask since
					// only the sign bit is used by movemask and blendv
					auto unoccluded = _mm256_xor_ps(occlusion.occluded(scene), _mm256_set1_ps(-0.f));
					if (_mm256_movemask_ps(unoccluded) != 0){
						const auto c = scene.materials[i]->shade(w_o, w_i) * li
							* _mm256_max_ps(w_i.dot(dg.normal), _mm256_set1_ps(0.f));
						shade_mask = _mm256_and_ps(shade_mask, unoccluded);
						color.r = _mm256_blendv_ps(color.r, c.r, shade_mask);
						color.g = _mm256_blendv_ps(color.g, c.g, shade_mask);
						color.b = _mm256_blendv_ps(color.b, c.b, shade_mask);
					}
				}
			}
			target.write_samples(samples, color, hits);
		}
	}
	target.save_image("out.bmp");
}

