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
	psimd::mask intersect(RayN &ray, DiffGeomN &dg) const override;
};

#endif

