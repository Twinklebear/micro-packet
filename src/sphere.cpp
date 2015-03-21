#include "sphere.h"

Sphere::Sphere(Vec3f pos, float radius, int material_id) : pos(pos), radius(radius), material_id(material_id){}
__m256 Sphere::intersect(Ray8 &ray, DiffGeom8 &dg) const {
	const auto center = Vec3f_8{pos};
	const auto d = center - ray.o;
	const auto a = ray.d.length_sqr();
	const auto b = _mm256_mul_ps(ray.d.dot(d), _mm256_set1_ps(-2.f));
	const auto c = _mm256_sub_ps(d.dot(d), _mm256_mul_ps(_mm256_set1_ps(radius),
				_mm256_set1_ps(radius)));
	// Solve the quadratic equation and store the mask of potential hits
	// We'll update this mask as we discard other potential hits, eg. due to
	// the hit being beyond the ray's t value
	__m256 t0 = _mm256_set1_ps(0);
	__m256 t1 = _mm256_set1_ps(0);
	auto hits = solve_quadratic(a, b, c, t0, t1);
	// We want t0 to hold the nearest t value that is greater than ray.t_min
	auto swap_t = _mm256_cmp_ps(t0, ray.t_min, _CMP_LT_OQ);
	t0 = _mm256_blendv_ps(t0, t1, swap_t);
	// Check which hits are within the ray's t range
	auto in_range = _mm256_and_ps(_mm256_cmp_ps(t0, ray.t_min, _CMP_GT_OQ),
			_mm256_cmp_ps(t0, ray.t_max, _CMP_LT_OQ));
	hits = _mm256_and_ps(hits, in_range);
	// Check if all rays miss the sphere
	if (_mm256_movemask_ps(hits) == 0){
		return hits;
	}
	// Update t values for rays that did hit
	ray.t_max = _mm256_blendv_ps(ray.t_max, t0, hits);

	const auto point = ray.at(ray.t_max);
	dg.point.x = _mm256_blendv_ps(dg.point.x, point.x, hits);
	dg.point.y = _mm256_blendv_ps(dg.point.y, point.y, hits);
	dg.point.z = _mm256_blendv_ps(dg.point.z, point.z, hits);
	const auto normal = point - center;
	dg.normal.x = _mm256_blendv_ps(dg.normal.x, normal.x, hits);
	dg.normal.y = _mm256_blendv_ps(dg.normal.y, normal.y, hits);
	dg.normal.z = _mm256_blendv_ps(dg.normal.z, normal.z, hits);
	dg.normal.normalize();
	// Is there some cleaner way to do this (including without the type punning?) and a blendv_epi32 instruction?
	dg.material_id = _mm256_blendv_epi8(dg.material_id, _mm256_set1_epi32(material_id), *(__m256i*)&hits);
	return hits;
}
