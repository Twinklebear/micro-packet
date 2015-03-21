#ifndef SPHERE_H
#define SPHERE_H

#include "vec.h"
#include "geometry.h"

struct Sphere : Geometry {
	Vec3f pos;
	float radius;
	int material_id;

	Sphere(Vec3f pos, float radius, int material_id);
	/*
	 * Test 8 rays against the sphere, returns masks for the hits (0xff) and misses (0x00)
	 */
	__m256 intersect(Ray8 &ray, DiffGeom8 &dg) const override;
};

#endif

