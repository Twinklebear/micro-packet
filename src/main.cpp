#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <memory>
#include <tsimd.h>
#include "geometry.h"
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
#include "block_queue.h"
#include "ld_sampler.h"
#include "scene.h"

void render(const Scene &scene, const PerspectiveCamera &camera, const Vec2fN img_dim, RenderTarget &target,
			BlockQueue &block_queue){
	std::random_device rand_device;
	std::mt19937 rng(rand_device());
	auto sampler = LDSampler{64, block_queue.get_block_dim()};
	for (auto block = block_queue.next(); block != block_queue.end(); block = block_queue.next()){
		sampler.select_block(block);
		while (sampler.has_samples()){
			auto samples = Vec2fN{0, 0};
			RayN packet;
			packet.active = sampler.sample(rng, samples);
			camera.generate_rays(packet, samples / img_dim);

			DiffGeomN dg;
			auto hits = scene.intersect(packet, dg);
			// If we hit something, shade it, otherwise use the background color (black)
			auto color = ColorfN{0};
			if (tsimd::any(hits)) {
				// How does ISPC find the unique values for its foreach_unique loop? Would like to do that
				// if it will be nicer than this
				std::array<int32_t, 8> mat_ids;
				tsimd::store(dg.material_id, static_cast<void*>(mat_ids.data()));
				// std::unique just removes consecutive repeated elements, so sort things first so we
				// don't get something like -1, 0, -1 or such
				std::sort(std::begin(mat_ids), std::end(mat_ids));
				auto id_end = std::unique(std::begin(mat_ids), std::end(mat_ids));
				for (auto it = std::begin(mat_ids); it != id_end; ++it){
					if (*it == -1){
						continue;
					}
					auto shade_mask = dg.material_id == *it;
					if (tsimd::any(shade_mask)) {
						const auto w_o = -packet.d;
						Vec3fN w_i{0};
						// Setup occlusion tester and set active ray mask to just be those with
						// corresponding to tests from hits with the material id being shaded
						OcclusionTester occlusion;
						const auto li = scene.light.sample(dg.point, w_i, occlusion);
						occlusion.rays.active = shade_mask;
						// We just need to flip the sign bit to change occluded mask to unoccluded mask since
						// only the sign bit is used by movemask and blendv
						auto unoccluded = !occlusion.occluded(scene);
						if (tsimd::any(unoccluded)) {
							const auto c = scene.materials[*it]->shade(w_o, w_i) * li
								* tsimd::max(w_i.dot(dg.normal), tsimd::vfloat(0.f));
							shade_mask = shade_mask && unoccluded;

							for (size_t i = 0; i < 3; ++i) {
								color[i] = tsimd::select(shade_mask, c[i], color[i]);
							}
						}
					}
				}
			}
			target.write_samples(samples, color, packet.active);
		}
	}
}

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
	const auto img_dim = Vec2fN{static_cast<float>(width), static_cast<float>(height)};
	const uint32_t block_dim = 8;
	auto block_queue = BlockQueue{block_dim, width, height};

	render(scene, camera, img_dim, target, block_queue);

	target.save_image("out.bmp");
}

