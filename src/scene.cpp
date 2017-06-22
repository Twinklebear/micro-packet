#include "scene.h"

Scene::Scene(std::vector<std::shared_ptr<Geometry>> geom, std::vector<std::shared_ptr<Material>> mats, PointLight light)
	: geometry(geom), materials(mats), light(light)
{}
psimd::mask Scene::intersect(Ray8 &rays, DiffGeom8 &dg) const {
	psimd::mask hits(0)
	for (const auto &g : geometry){
		hits = g->intersect(rays, dg) || hits;
	}
	return hits;
}

