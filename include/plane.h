#ifndef PLANE_H
#define PLANE_H

#include "vec.h"
#include "geometry.h"

/*
 * An infinite plane
 */
struct Plane : Geometry {
	Vec3f pos, normal;
	int material_id;

	Plane(Vec3f pos, Vec3f normal, int material_id);
	tsimd::vmask intersect(RayN &ray, DiffGeomN &dg) const override;
};

#endif

