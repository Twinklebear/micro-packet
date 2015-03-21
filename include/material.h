#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec.h"

struct Material {
	virtual Colorf_8 shade(const Vec3f_8 &w_o, const Vec3f_8 &w_i) const = 0;
};

struct LambertianMaterial : Material {
	Colorf color;

	inline LambertianMaterial(Colorf color) : color(color){}
	inline Colorf_8 shade(const Vec3f_8&, const Vec3f_8&) const {
		return Colorf_8{color * M_1_PI};
	}
};

#endif

