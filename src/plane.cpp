#include "plane.h"

Plane::Plane(Vec3f pos, Vec3f normal, int material_id) : pos(pos), normal(normal.normalized()), material_id(material_id){}
__m256 Plane::intersect(Ray8 &ray, DiffGeom8 &dg) const {
	const auto vpos = Vec3f_8{pos};
	const auto vnorm = Vec3f_8{normal};
	const auto t = _mm256_div_ps((vpos - ray.o).dot(vnorm), ray.d.dot(vnorm));
	auto hits = _mm256_and_ps(_mm256_cmp_ps(t, ray.t_min, _CMP_GT_OQ),
			_mm256_cmp_ps(t, ray.t_max, _CMP_LT_OQ));
	hits = _mm256_and_ps(hits, ray.active);
	// Check if all rays miss the sphere
	if (_mm256_movemask_ps(hits) == 0){
		return hits;
	}
	// Update t values for rays that did hit
	ray.t_max = _mm256_blendv_ps(ray.t_max, t, hits);
	const auto point = ray.at(ray.t_max);
	dg.point.x = _mm256_blendv_ps(dg.point.x, point.x, hits);
	dg.point.y = _mm256_blendv_ps(dg.point.y, point.y, hits);
	dg.point.z = _mm256_blendv_ps(dg.point.z, point.z, hits);
	dg.normal.x = _mm256_blendv_ps(dg.normal.x, vnorm.x, hits);
	dg.normal.y = _mm256_blendv_ps(dg.normal.y, vnorm.y, hits);
	dg.normal.z = _mm256_blendv_ps(dg.normal.z, vnorm.z, hits);
	// There's no blendv_epi32 in AVX/AVX2 so we have to resort to some hacky casting
	dg.material_id = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(dg.material_id),
			_mm256_castsi256_ps(_mm256_set1_epi32(material_id)), hits));
	return hits;
}

