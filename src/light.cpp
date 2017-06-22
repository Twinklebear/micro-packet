#include "light.h"
#include "occlusion_tester.h"

PointLight::PointLight(Vec3f pos, Colorf intensity) : pos(pos), intensity(intensity){}
ColorfN PointLight::sample(const Vec3fN &p, Vec3fN &w_i, OcclusionTester &occlusion) const {
	occlusion.set_points(p, pos);
	w_i = (pos - p);
	const auto dist_sqr = w_i.length_sqr();
	w_i.normalize();
	return ColorfN{intensity} / dist_sqr;
}

