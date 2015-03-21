#include "scene.h"

Scene::Scene(std::vector<std::shared_ptr<Geometry>> geom, std::vector<std::shared_ptr<Material>> mats, PointLight light)
	: geometry(geom), materials(mats), light(light)
{}
__m256 Scene::intersect(Ray8 &rays, DiffGeom8 &dg) const {
	__m256 hits = _mm256_set1_ps(0.f);
	for (const auto &g : geometry){
		hits = _mm256_or_ps(hits, g->intersect(rays, dg));
	}
	return hits;
}

