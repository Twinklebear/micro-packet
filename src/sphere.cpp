#include "sphere.h"

Sphere::Sphere(Vec3f pos, float radius, int material_id) : pos(pos), radius(radius), material_id(material_id){}
psimd::mask Sphere::intersect(RayN &ray, DiffGeomN &dg) const {
	const auto center = Vec3fN{pos};
	const auto d = center - ray.o;
	const auto a = ray.d.length_sqr();
	const auto b = -2.f * ray.d.dot(d);
	const auto c = d.dot(d) - radius * radius;
	// Solve the quadratic equation and store the mask of potential hits
	// We'll update this mask as we discard other potential hits, eg. due to
	// the hit being beyond the ray's t value
	psimd::pack<float> t0(0);
	psimd::pack<float> t1(0);
	auto hits = solve_quadratic(a, b, c, t0, t1);

	// We want t0 to hold the nearest t value that is greater than ray.t_min
	auto swap_t = t0 < ray.t_min;
	t0 = psimd::select(swap_t, t1, t0);

	// Check which hits are within the ray's t range
	auto in_range = t0 > ray.t_min && t0 < ray.t_max && ray.active && hits;
	// Check if all rays miss the sphere
	if (none(hits)) {
		return hits;
	}

	// Update t values for rays that did hit
	ray.t_max = psimd::select(hits, t, ray.t_max);
	const auto point = ray.at(ray.t_max);
	const auto normal = (point - center).normalized();
	for (size_t i = 0; i < 3; ++i) {
		dg.point[i] = psimd::select(hits, point[i], dg.point[i]);
		dg.normal[i] = psimd::select(hits, normal[i], dg.normal[i]);
	}
	dg.material_id = psimd::select(hits, material_id, dg.material_id);
	return hits;
}

