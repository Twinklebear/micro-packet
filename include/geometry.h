#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "diff_geom.h"

struct Geometry {
	/*
	 * Test a ray packet for intersection against the object
	 */
	virtual psimd::mask<> intersect(RayN &ray, DiffGeomN &dg) const = 0;
};

#endif

