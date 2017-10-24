#ifndef DIFF_GEOM_H
#define DIFF_GEOM_H

#include "vec.h"

/*
 * Structure storing N differential geomtry entries
 */
struct DiffGeomN {
	Vec3fN point, normal;
	tsimd::vint material_id;

	DiffGeomN() : point(0), normal(0), material_id(-1){}
};

#endif

