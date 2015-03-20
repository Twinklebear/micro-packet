#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "immintrin.h"

struct Geometry {
	/*
	 * Test a ray packet for intersection against the object
	 */
	virtual __m256 intersect(Ray8 &ray) const = 0;
};

#endif

