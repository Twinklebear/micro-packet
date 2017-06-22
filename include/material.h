#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec.h"
#include "color.h"

struct Material {
	virtual ColorfN shade(const Vec3fN &w_o, const Vec3fN &w_i) const = 0;
};

struct LambertianMaterial : Material {
	Colorf color;

	inline LambertianMaterial(Colorf color) : color(color){}
	inline ColorfN shade(const Vec3fN&, const Vec3fN&) const {
		return ColorfN{color * static_cast<float>(M_1_PI)};
	}
};

#endif

