#include "plane.h"

Plane::Plane(Vec3f pos, Vec3f normal) : pos(pos), normal(normal){}
__m256 Plane::intersect(Ray8 &ray) const {
	const auto vpos = Vec3f_8{pos};
	const auto vnorm = Vec3f_8{normal};
	const auto t = (vpos - ray.o).dot(vnorm) / ray.d.dot(vnorm);
	auto hits = _mm256_and_ps(_mm256_cmp_ps(t, ray.t_min, _CMP_GT_OQ),
			_mm256_cmp_ps(t, ray.t_max, _CMP_LT_OQ));
	// Check if all rays miss the sphere
	if (_mm256_movemask_ps(hits) == 0){
		return hits;
	}
	// Update t values for rays that did hit
	ray.t_max = _mm256_blendv_ps(ray.t_max, t, hits);
	return hits;
}

