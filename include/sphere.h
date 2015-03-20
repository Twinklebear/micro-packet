#ifndef SPHERE_H
#define SPHERE_H

#include "vec.h"
#include "geometry.h"

struct Sphere : Geometry {
	Vec3f pos;
	float radius;

	Sphere(Vec3f pos, float radius);
	/*
	 * Test 8 rays against the sphere, returns masks for the hits (0xff) and misses (0x00)
	 */
	__m256 intersect(Ray8 &ray) const override;
};

#endif

