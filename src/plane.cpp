#include "plane.h"

Plane::Plane(Vec3f pos, Vec3f normal, int material_id) : pos(pos), normal(normal.normalized()), material_id(material_id){}
tsimd::vmask Plane::intersect(RayN &ray, DiffGeomN &dg) const {
	const auto vpos = Vec3fN{pos};
	const auto vnorm = Vec3fN{normal};
	const auto t = (vpos - ray.o).dot(vnorm) / ray.d.dot(vnorm);

	auto hits = t > ray.t_min && t < ray.t_max && ray.active;
	// Check if all rays miss the sphere
	if (tsimd::none(hits)) {
		return hits;
	}

	// Update t values for rays that did hit
	ray.t_max = tsimd::select(hits, t, ray.t_max);
	const auto point = ray.at(ray.t_max);
	for (size_t i = 0; i < 3; ++i) {
		dg.point[i] = tsimd::select(hits, point[i], dg.point[i]);
		dg.normal[i] = tsimd::select(hits, vnorm[i], dg.normal[i]);
	}
	dg.material_id = tsimd::select(hits, tsimd::pack<int>(material_id), dg.material_id);
	return hits;
}

