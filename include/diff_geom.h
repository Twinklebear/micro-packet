#ifndef DIFF_GEOM_H
#define DIFF_GEOM_H

#include "vec.h"

/*
 * Structure storing 8 differential geomtry entries
 */
struct DiffGeom8 {
	Vec3f_8 point, normal;
	__m256i material_id;

	DiffGeom8() : point(0), normal(0), material_id(_mm256_set1_epi32(-1)){}
};

#endif

