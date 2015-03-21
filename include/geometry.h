#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "immintrin.h"
#include "diff_geom.h"

struct Geometry {
	/*
	 * Test a ray packet for intersection against the object
	 */
	virtual __m256 intersect(Ray8 &ray, DiffGeom8 &dg) const = 0;
};

#endif

