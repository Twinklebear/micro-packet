#ifndef LIGHT_H
#define LIGHT_H

#include "vec.h"
#include "color.h"

struct PointLight {
	Vec3f pos;
	Colorf intensity;

	inline PointLight(Vec3f pos, Colorf intensity) : pos(pos), intensity(intensity){}
	inline Colorf_8 sample(const Vec3f_8 &p, Vec3f_8 &w_i) const {
		w_i = (pos - p);
		const auto dist_sqr = w_i.length_sqr();
		w_i.normalize();
		return Colorf_8{intensity} / dist_sqr;
	}
};

#endif

