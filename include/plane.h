#ifndef PLANE_H
#define PLANE_H

#include "vec.h"
#include "geometry.h"

/*
 * An infinite plane
 */
struct Plane : Geometry {
	Vec3f pos, normal;

	Plane(Vec3f pos, Vec3f normal);
	__m256 intersect(Ray8 &ray) const override;
};

#endif

