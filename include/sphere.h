#ifndef SPHERE_H
#define SPHERE_H

#include "immintrin.h"

struct Sphere {
	float x, y, z, radius;

	Sphere(float x, float y, float z, float radius);
	/*
	 * Test 8 rays against the sphere, returns masks for the hits (0xff) and misses (0x00)
	 */
	__m256 intersect(Ray8 &ray) const;
};

#endif

