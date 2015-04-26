#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <memory>
#include "geometry.h"
#include "immintrin.h"
#include "vec.h"
#include "mat3.h"
#include "color.h"
#include "render_target.h"
#include "camera.h"
#include "sphere.h"
#include "plane.h"
#include "diff_geom.h"
#include "material.h"
#include "light.h"
#include "occlusion_tester.h"
#include "block_queue.h"
#include "ld_sampler.h"
#include "scene.h"

int main(int, char**){
	const uint32_t width = 800;
	const uint32_t height = 600;
	const auto scene = Scene{
		{
			std::make_shared<Sphere>(Vec3f{0}, 0.5f, 0),
			std::make_shared<Plane>(Vec3f{0, -0.5f, 0.5f}, Vec3f{0, 1, 0}, 1)
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

	// TODO: Proper seeding
	std::mt19937 rng;
	const uint32_t block_dim = 4;
	auto block_queue = BlockQueue{block_dim, width, height};
	auto sampler = LDSampler{8, block_dim};
	for (auto block = block_queue.next(); block != block_queue.end(); block = block_queue.next()){
		sampler.select_block(block);
		while (sampler.has_samples()){
			auto samples = Vec2f_8{0, 0};
			Ray8 packet;
			packet.active = sampler.sample(rng, samples);
			camera.generate_rays(packet, samples / img_dim);

			DiffGeom8 dg;
			auto hits = scene.intersect(packet, dg);
			// If we hit something, shade it, otherwise use the background color (black)
			auto color = Colorf_8{0};
			if (_mm256_movemask_ps(hits) != 0){
				// How does ISPC find the unique values for its foreach_unique loop? Would like to do that
				// if it will be nicer than this
				std::array<int32_t, 8> mat_ids;
				_mm256_storeu_si256((__m256i*)mat_ids.data(), dg.material_id);
				// std::unique just removes consecutive repeated elements, so sort things first so we
				// don't get something like -1, 0, -1 or such
				std::sort(std::begin(mat_ids), std::end(mat_ids));
				std::unique(std::begin(mat_ids), std::end(mat_ids));
				for (const auto &i : mat_ids){
					if (i == -1){
						continue;
					}
					const auto use_mat = _mm256_cmpeq_epi32(dg.material_id, _mm256_set1_epi32(i));
					auto shade_mask = _mm256_castsi256_ps(use_mat);
					if (_mm256_movemask_ps(shade_mask) != 0){
						const auto w_o = -packet.d;
						Vec3f_8 w_i{0};
						// Setup occlusion tester and set active ray mask to just be those with
						// corresponding to tests from hits with the material id being shaded
						OcclusionTester occlusion;
						const auto li = scene.light.sample(dg.point, w_i, occlusion);
						occlusion.rays.active = shade_mask;
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
			}
			target.write_samples(samples, color, packet.active);
		}
	}
	target.save_image("out.bmp");
}

