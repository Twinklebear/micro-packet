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

int main(int, char**){
	const uint32_t width = 800;
	const uint32_t height = 600;
	const std::vector<std::shared_ptr<Material>> materials{
		std::make_shared<LambertianMaterial>(Colorf{1, 0, 0}),
		std::make_shared<LambertianMaterial>(Colorf{0, 0, 1})
	};

	const auto sphere = Sphere{Vec3f{0}, 0.5, 0};
	const auto plane = Plane{Vec3f{0, -2, 0}, Vec3f{0, 1, 0}, 1};
	const auto light = PointLight{Vec3f{1, 1, -2}, Colorf{50}};
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

		DiffGeom8 dg;
		auto hits = plane.intersect(packet, dg);
		hits = _mm256_or_ps(hits, sphere.intersect(packet, dg));
		// If we hit something, shade it
		if (_mm256_movemask_ps(hits) != 0){
			auto color = Colorf_8{0};
			// This is pretty ugly. TODO: Better approach to shading??
			for (uint32_t i = 0; i < materials.size(); ++i){
				// Is there a better way to just get the bits of one register casted to another?
				// it seems like the or operations don't let you mix either
				const auto use_mat = _mm256_cmpeq_epi32(dg.material_id, _mm256_set1_epi32(i));
				const auto shade_mask = *(__m256*)&use_mat;
				if (_mm256_movemask_ps(shade_mask) != 0){
					const auto w_o = -packet.d;
					Vec3f_8 w_i{0};
					const auto li = light.sample(dg.point, w_i);
					// Just flat material at the moment so this doesn't matter (we also don't have lights :P)
					const auto c = materials[i]->shade(w_o, w_i) * li
						* _mm256_max_ps(w_i.dot(dg.normal), _mm256_set1_ps(0.f));
					color.r = _mm256_blendv_ps(color.r, c.r, shade_mask);
					color.g = _mm256_blendv_ps(color.g, c.g, shade_mask);
					color.b = _mm256_blendv_ps(color.b, c.b, shade_mask);
				}
			}
			target.write_samples(samples, color, hits);
		}
	}
	target.save_image("out.bmp");
}

