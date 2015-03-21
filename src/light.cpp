#include "light.h"
#include "occlusion_tester.h"

PointLight::PointLight(Vec3f pos, Colorf intensity) : pos(pos), intensity(intensity){}
Colorf_8 PointLight::sample(const Vec3f_8 &p, Vec3f_8 &w_i, OcclusionTester &occlusion) const {
	occlusion.set_points(p, pos);
	w_i = (pos - p);
	const auto dist_sqr = w_i.length_sqr();
	w_i.normalize();
	return Colorf_8{intensity} / dist_sqr;
}

