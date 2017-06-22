#ifndef LIGHT_H
#define LIGHT_H

#include "vec.h"
#include "color.h"

struct OcclusionTester;

struct PointLight {
	Vec3f pos;
	Colorf intensity;

	PointLight(Vec3f pos, Colorf intensity);
	/*
	 * Sample incident illumination from the light source at p
	 * returns the incident direction of the light and an occlusion tester
	 * that can be used to check if the light is visble
	 */
	ColorfN sample(const Vec3fN &p, Vec3fN &w_i, OcclusionTester &occlusion) const;
};

#endif

